"""Bet execution engine.

Runs the three-phase fill strategy for a single signal and then monitors
resting orders, applying the live line-shift cancellation rules.

Phase 1 — Immediate sweep: take any lay liquidity in [MIN_PRICE, MAX_PRICE].
Phase 2 — Resting orders: distribute the remaining stake at MIN_PRICE.
Phase 3 — Monitor: poll statuses (500ms), check line shift (30s), cancel/stop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import database as db
from config import config
from smarkets import Market, SmarketsClient, SmarketsError

logger = logging.getLogger("pongbot.executor")

# Monitor tuning (per spec)
STATUS_POLL_SECONDS = 0.5
LINE_CHECK_SECONDS = 30
MONITOR_MAX_SECONDS = 2 * 60 * 60  # 2 hours
ORDERBOOK_POLL_SECONDS = 0.2  # 200ms during active execution


@dataclass
class OrderRecord:
    fill_id: int
    order_id: str | None
    market_id: str
    contract_id: str
    line: float
    price: float
    stake: float
    status: str  # unmatched / matched / cancelled
    phase: int


@dataclass
class ExecutionResult:
    matched_eur: float = 0.0
    matched_orders: int = 0
    resting_eur: float = 0.0
    line: float | None = None
    line_ok: bool = True
    orders: list[OrderRecord] = field(default_factory=list)


class BetExecutor:
    def __init__(self, client: SmarketsClient, notifier, signal, signal_id: int):
        self.client = client
        self.notifier = notifier
        self.signal = signal
        self.signal_id = signal_id
        self.stake_cap = signal.stake_eur(config.unit_size)
        self.orders: list[OrderRecord] = []

    # --- Line filter ----------------------------------------------------
    def _line_passes(self, market: Market) -> bool:
        if market.line is None:
            return False
        if self.signal.direction == "OVER":
            return market.line <= config.max_line_over
        return market.line >= config.min_line_under

    def _target_contract(self, market: Market):
        side = "over" if self.signal.direction == "OVER" else "under"
        for c in market.contracts:
            if c.side == side:
                return c
        return None

    @property
    def _placed_count(self) -> int:
        return len([o for o in self.orders if o.status != "cancelled"])

    @property
    def _committed_stake(self) -> float:
        return sum(o.stake for o in self.orders if o.status != "cancelled")

    # --- Entry point ----------------------------------------------------
    async def run(self) -> None:
        try:
            event = await self.client.find_event(
                self.signal.player1,
                self.signal.player2,
                minutes_until=self.signal.minutes_until,
            )
        except SmarketsError as exc:
            await self.notifier.error(self.signal, str(exc))
            return

        event_id = str(event.get("id"))

        markets = await self.client.wait_for_markets(
            event_id,
            minutes_until=self.signal.minutes_until,
            line_filter=self._line_passes,
        )

        if not markets:
            await self._handle_no_passing_markets(event_id)
            return

        db.mark_signal_executed(self.signal_id, True)

        # Phase 1 + 2
        await self._phase1_sweep(markets)
        await self._phase2_resting(markets)

        # Phase 3
        result = await self._phase3_monitor(event_id, markets)
        await self.notifier.execution_complete(
            self.signal,
            matched_eur=result.matched_eur,
            matched_orders=result.matched_orders,
            resting_eur=result.resting_eur,
            line=result.line,
            line_ok=result.line_ok,
        )

    async def _handle_no_passing_markets(self, event_id: str) -> None:
        """Distinguish 'no markets at all' from 'markets exist but line OOB'."""
        try:
            all_markets = await self.client.get_markets(event_id)
        except SmarketsError:
            all_markets = []

        oob = [m for m in all_markets if m.line is not None and not self._line_passes(m)]
        if oob:
            # Report the line closest to the bound for clarity.
            if self.signal.direction == "OVER":
                worst = max(oob, key=lambda m: m.line)
                await self.notifier.line_rejected(
                    self.signal, line=worst.line, limit=config.max_line_over, kind="over"
                )
            else:
                worst = min(oob, key=lambda m: m.line)
                await self.notifier.line_rejected(
                    self.signal, line=worst.line, limit=config.min_line_under, kind="under"
                )
        else:
            await self.notifier.error(
                self.signal, "No over/under markets appeared on Smarkets"
            )

    # --- Phase 1: immediate sweep --------------------------------------
    async def _phase1_sweep(self, markets: list[Market]) -> None:
        for market in markets:
            if self._committed_stake >= self.stake_cap or self._placed_count >= config.max_orders:
                break
            contract = self._target_contract(market)
            if contract is None:
                continue
            try:
                book = await self.client.get_order_book(market.id, contract.id)
            except SmarketsError as exc:
                logger.warning("order book error: %s", exc)
                continue

            price = book.best_lay_price
            if price is None:
                continue
            if not (config.min_price <= price <= config.max_price):
                continue
            if book.best_lay_volume < config.min_liquidity:
                continue

            remaining = self.stake_cap - self._committed_stake
            take = min(book.best_lay_volume, remaining)
            if take < config.min_liquidity and take < remaining:
                continue

            await self._place(market, contract, price=price, stake=take, phase=1)
            await asyncio.sleep(ORDERBOOK_POLL_SECONDS)

    # --- Phase 2: resting orders ---------------------------------------
    async def _phase2_resting(self, markets: list[Market]) -> None:
        remaining = self.stake_cap - self._committed_stake
        if remaining <= 0:
            return

        # Contracts still available to rest on (respecting MAX_ORDERS).
        contracts = []
        for market in markets:
            c = self._target_contract(market)
            if c is not None:
                contracts.append((market, c))
        if not contracts:
            return

        slots = max(1, min(len(contracts), config.max_orders - self._placed_count))
        if slots <= 0:
            return
        per = remaining / slots

        for market, contract in contracts[:slots]:
            if self._placed_count >= config.max_orders:
                break
            if per <= 0:
                break
            await self._place(market, contract, price=config.min_price, stake=per, phase=2)

    async def _place(self, market: Market, contract, *, price: float, stake: float, phase: int):
        try:
            result = await self.client.place_back_order(
                market.id, contract.id, decimal_price=price, stake_eur=stake
            )
        except SmarketsError as exc:
            logger.warning("place order failed: %s", exc)
            return

        # Phase-1 sweeps take available liquidity -> treat as matched.
        status = result.status
        if phase == 1 and status not in ("cancelled", "rejected"):
            status = "matched"

        fill_id = db.insert_fill(
            self.signal_id,
            market_id=market.id,
            contract_id=contract.id,
            order_id=result.order_id,
            price=price,
            stake=stake,
            line_number=market.line if market.line is not None else 0.0,
            status=status,
        )
        if status == "matched":
            db.update_fill_status(fill_id, "matched", matched=True)

        self.orders.append(
            OrderRecord(
                fill_id=fill_id,
                order_id=result.order_id,
                market_id=market.id,
                contract_id=contract.id,
                line=market.line if market.line is not None else 0.0,
                price=price,
                stake=stake,
                status=status,
                phase=phase,
            )
        )
        logger.info(
            "Placed phase-%d %s €%.2f @ %.2f (line %.1f) -> %s",
            phase, self.signal.direction, stake, price, self.orders[-1].line, status,
        )

    # --- Phase 3: monitor ----------------------------------------------
    def _line_shifted_against(self, order: OrderRecord, current_line: float) -> bool:
        thr = config.line_cancel_threshold
        if self.signal.direction == "OVER":
            # line drops more than threshold below order line
            return current_line < order.line - thr
        # UNDER: line rises more than threshold above order line
        return current_line > order.line + thr

    async def _phase3_monitor(self, event_id: str, markets: list[Market]) -> ExecutionResult:
        start = time.monotonic()
        last_line_check = 0.0
        ref_line = markets[0].line if markets and markets[0].line is not None else None
        line_ok = True

        while True:
            now = time.monotonic()

            # Stop after 2 hours: cancel remaining unmatched resting orders.
            if now - start >= MONITOR_MAX_SECONDS:
                await self._cancel_all_unmatched("2 hour timeout")
                break

            # Poll order statuses.
            await self._refresh_statuses()

            matched_total = sum(o.stake for o in self.orders if o.status == "matched")
            if matched_total >= 0.98 * self.stake_cap:
                break

            # No more unmatched orders -> done.
            if not any(o.status == "unmatched" for o in self.orders):
                break

            # Line check every 30s.
            if now - last_line_check >= LINE_CHECK_SECONDS:
                last_line_check = now
                line_ok = await self._check_lines(event_id, ref_line)

            await asyncio.sleep(STATUS_POLL_SECONDS)

        matched = [o for o in self.orders if o.status == "matched"]
        resting = [o for o in self.orders if o.status == "unmatched"]
        return ExecutionResult(
            matched_eur=sum(o.stake for o in matched),
            matched_orders=len(matched),
            resting_eur=sum(o.stake for o in resting),
            line=ref_line,
            line_ok=line_ok,
            orders=self.orders,
        )

    async def _refresh_statuses(self) -> None:
        for order in self.orders:
            if order.status != "unmatched" or not order.order_id:
                continue
            try:
                status = await self.client.get_order_status(order.order_id)
            except SmarketsError:
                continue
            if status != order.status:
                order.status = status
                db.update_fill_status(
                    order.fill_id, status, matched=(status == "matched")
                )

    async def _check_lines(self, event_id: str, ref_line: float | None) -> bool:
        """Check every unmatched order against the live line; cancel if shifted.

        Returns whether the reference line is still within bounds.
        """
        for order in [o for o in self.orders if o.status == "unmatched"]:
            try:
                current = await self.client.get_current_line(event_id, order.market_id)
            except SmarketsError:
                continue
            if current is None:
                continue
            if self._line_shifted_against(order, current):
                await self._cancel(order, current)

        # Recompute bound status of reference line.
        if ref_line is None:
            return True
        if self.signal.direction == "OVER":
            return ref_line <= config.max_line_over
        return ref_line >= config.min_line_under

    async def _cancel(self, order: OrderRecord, new_line: float) -> None:
        try:
            await self.client.cancel_order(order.order_id)
        except SmarketsError as exc:
            logger.warning("cancel failed for %s: %s", order.order_id, exc)
            return
        order.status = "cancelled"
        reason = (
            f"line moved {abs(new_line - order.line):g} points "
            f"({order.line:g} -> {new_line:g})"
        )
        db.update_fill_status(order.fill_id, "cancelled", cancel_reason=reason)
        await self.notifier.order_cancelled_line(
            self.signal,
            order_stake=order.stake,
            order_line=order.line,
            new_line=new_line,
            threshold=config.line_cancel_threshold,
        )

    async def _cancel_all_unmatched(self, reason: str) -> None:
        for order in [o for o in self.orders if o.status == "unmatched"]:
            try:
                await self.client.cancel_order(order.order_id)
            except SmarketsError:
                pass
            order.status = "cancelled"
            db.update_fill_status(order.fill_id, "cancelled", cancel_reason=reason)
