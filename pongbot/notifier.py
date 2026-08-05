"""Discord DM notifier.

Every notification is sent privately to ``BOT_OWNER_USER_ID``. Nothing is ever
posted in a server channel. In paper mode every message is prefixed ``[PAPER]``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("pongbot.notifier")


class Notifier:
    def __init__(self, client, owner_user_id: int, *, paper_mode: bool = True):
        self.client = client
        self.owner_user_id = owner_user_id
        self.paper_mode = paper_mode
        self._user = None

    async def _get_user(self):
        if self._user is None:
            try:
                self._user = await self.client.fetch_user(self.owner_user_id)
            except Exception as exc:  # noqa: BLE001
                logger.error("Could not fetch owner user %s: %s", self.owner_user_id, exc)
                return None
        return self._user

    async def send(self, message: str) -> None:
        """Send a DM to the owner. Never raises."""
        if self.paper_mode:
            message = f"[PAPER]\n{message}"
        try:
            user = await self._get_user()
            if user is None:
                logger.warning("No owner user; DM not sent:\n%s", message)
                return
            await user.send(message)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to send DM: %s\n%s", exc, message)

    # --- Templated notifications ---------------------------------------
    async def signal_received(self, signal, stake_eur: float) -> None:
        t = f"T-{signal.minutes_until} minutes" if signal.minutes_until is not None else "T-? minutes"
        await self.send(
            "⚡ Signal received\n"
            f"Match: {signal.player1} vs {signal.player2}\n"
            f"Direction: {signal.direction} | {signal.units:g}u | €{stake_eur:.0f}\n"
            f"League: {signal.league.value}\n"
            f"{t}\n"
            "Searching Smarkets..."
        )

    async def execution_complete(
        self, signal, *, matched_eur: float, matched_orders: int,
        resting_eur: float, line: float | None, line_ok: bool,
    ) -> None:
        line_str = (
            f"{line:g} {'✅ within bounds' if line_ok else '⚠️ out of bounds'}"
            if line is not None else "n/a"
        )
        await self.send(
            "✅ Execution complete\n"
            f"Match: {signal.player1} vs {signal.player2} | {signal.direction}\n"
            f"Matched: €{matched_eur:.0f} across {matched_orders} orders\n"
            f"Resting: €{resting_eur:.0f} still open\n"
            f"Line: {line_str}"
        )

    async def order_cancelled_line(
        self, signal, *, order_stake: float, order_line: float,
        new_line: float, threshold: float,
    ) -> None:
        moved = abs(new_line - order_line)
        await self.send(
            "⚠️ Order cancelled — line moved\n"
            f"Match: {signal.player1} vs {signal.player2} | {signal.direction}\n"
            f"Order: €{order_stake:.0f} @ {order_line:g}\n"
            f"Line shifted: {order_line:g} → {new_line:g} (moved {moved:g} points)\n"
            f"Threshold: {threshold:g} points — cancelled"
        )

    async def signal_skipped(self, signal, reason: str) -> None:
        await self.send(
            "❌ Signal skipped\n"
            f"Match: {signal.player1} vs {signal.player2}\n"
            f"Reason: {reason}"
        )

    async def line_rejected(self, signal, *, line: float, limit: float, kind: str) -> None:
        if kind == "over":
            reason = f"Line {line:g} exceeds maximum {limit:g}"
        else:
            reason = f"Line {line:g} below minimum {limit:g}"
        await self.send(
            "❌ Signal skipped\n"
            f"Match: {signal.player1} vs {signal.player2} | {signal.direction}\n"
            f"Reason: {reason}"
        )

    async def error(self, signal, issue: str) -> None:
        match = (
            f"{signal.player1} vs {signal.player2}" if signal else "Unknown match"
        )
        await self.send(
            "⚠️ Error\n"
            f"Match: {match}\n"
            f"Issue: {issue}"
        )

    async def morning_report(self, date_str: str, counts: dict) -> None:
        bar = "━" * 23
        await self.send(
            f"🌙 OVERNIGHT REPORT — {date_str}\n"
            f"{bar}\n"
            f"Signals received:  {counts['signals_received']}\n"
            f"Executed:          {counts['executed']}\n"
            f"Skipped (no market): {counts['skipped_no_market']}\n"
            f"Skipped (line OOB):  {counts['skipped_line_oob']}\n"
            f"Orders cancelled (line shift): {counts['orders_cancelled_line']}\n"
            f"Orders placed:     {counts['orders_placed']}\n"
            f"Total matched:     €{counts['total_matched']:.0f}\n"
            f"{bar}"
        )
