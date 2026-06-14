"""Async Smarkets REST API client (v3).

Authentication uses an API token passed as ``Authorization: Token {token}``.

Notes on units / conversions
-----------------------------
* Stakes are handled in euros internally and converted to pence for the API
  (multiply by 100), per spec.
* Prices in this bot are expressed as decimal odds (e.g. 1.50 .. 2.50).
  Smarkets expresses price as implied-probability per 10,000, so a decimal
  odd ``d`` maps to ``round(10000 / d)``. Helpers below do the conversion.

Because live Smarkets access is approved separately, network behaviour is
defensive: every call retries on transient errors and backs off for 5s on a
429 rate-limit response, per spec.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger("pongbot.smarkets")

BASE_URL = "https://api.smarkets.com/v3"
RATE_LIMIT_BACKOFF = 5.0  # seconds, per spec
TABLE_TENNIS_TYPE = "table_tennis_match"


# --- Conversions --------------------------------------------------------
def decimal_to_smarkets_price(decimal_odds: float) -> int:
    """Decimal odds -> Smarkets price (implied probability per 10,000)."""
    return int(round(10000.0 / decimal_odds))


def smarkets_price_to_decimal(price: int) -> float:
    if not price:
        return 0.0
    return round(10000.0 / price, 4)


def eur_to_pence(eur: float) -> int:
    return int(round(eur * 100))


# --- Data structures ----------------------------------------------------
@dataclass
class Contract:
    id: str
    name: str
    # "over" or "under", derived from the contract/market name
    side: str | None = None


@dataclass
class Market:
    id: str
    event_id: str
    name: str
    line: float | None
    contracts: list[Contract]


@dataclass
class OrderBook:
    contract_id: str
    # Best price available to BACK (i.e. best resting LAY) as decimal odds.
    best_lay_price: float | None
    # Volume available at that price, in euros.
    best_lay_volume: float
    raw: dict | None = None


@dataclass
class OrderResult:
    order_id: str | None
    status: str  # unmatched / matched / cancelled / rejected / paper
    raw: dict | None = None


class SmarketsError(RuntimeError):
    pass


class SmarketsClient:
    def __init__(self, api_token: str, *, paper_mode: bool = True):
        self.api_token = api_token
        self.paper_mode = paper_mode
        self._session: aiohttp.ClientSession | None = None
        # Monotonic counter for synthetic paper order ids.
        self._paper_seq = 0

    async def __aenter__(self) -> "SmarketsClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def connect(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Token {self.api_token}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=20),
            )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # --- Low-level request with retry / 429 backoff --------------------
    async def _request(self, method: str, path: str, *, retries: int = 4, **kwargs):
        await self.connect()
        assert self._session is not None
        url = f"{BASE_URL}{path}"
        attempt = 0
        while True:
            attempt += 1
            try:
                async with self._session.request(method, url, **kwargs) as resp:
                    if resp.status == 429:
                        logger.warning("Smarkets 429 rate limit on %s; backing off %.0fs",
                                       path, RATE_LIMIT_BACKOFF)
                        await asyncio.sleep(RATE_LIMIT_BACKOFF)
                        continue
                    if resp.status >= 500:
                        raise SmarketsError(f"Server error {resp.status} on {path}")
                    text = await resp.text()
                    if resp.status >= 400:
                        raise SmarketsError(f"HTTP {resp.status} on {path}: {text}")
                    if not text:
                        return {}
                    return await resp.json()
            except (aiohttp.ClientError, SmarketsError, asyncio.TimeoutError) as exc:
                if attempt > retries:
                    raise SmarketsError(f"{method} {path} failed after {retries} retries: {exc}")
                backoff = min(2 ** attempt, 16)
                logger.warning("Smarkets request error (%s); retry %d in %ds", exc, attempt, backoff)
                await asyncio.sleep(backoff)

    # --- Event discovery ------------------------------------------------
    @staticmethod
    def _names_match(query_names: list[str], event_name: str) -> bool:
        ev = event_name.lower()
        # Require each player's surname (first token) to appear.
        for full in query_names:
            surname = full.lower().split()[0].strip(".") if full.split() else full.lower()
            if surname and surname not in ev:
                return False
        return True

    async def _search_events_once(self, player1: str, player2: str) -> dict | None:
        params = {"type": TABLE_TENNIS_TYPE, "state": "upcoming,live", "limit": 200}
        data = await self._request("GET", "/events/", params=params)
        for ev in data.get("events", []) or []:
            if self._names_match([player1, player2], ev.get("name", "")):
                return ev
        return None

    async def find_event(
        self,
        player1: str,
        player2: str,
        *,
        minutes_until: int | None,
        timeout_minutes: int = 90,
    ) -> dict:
        """Poll Smarkets until the matching table-tennis event appears.

        Polling cadence (per spec):
          * >30 min to start  -> every 30s
          * 10-30 min         -> every 10s
          * <10 min           -> every 2s
        Times out after ``timeout_minutes`` and raises ``SmarketsError``.
        """
        deadline = time.monotonic() + timeout_minutes * 60
        remaining = minutes_until if minutes_until is not None else 60

        while time.monotonic() < deadline:
            try:
                ev = await self._search_events_once(player1, player2)
            except SmarketsError as exc:
                logger.warning("Event search error: %s", exc)
                ev = None
            if ev:
                logger.info("Found event %s (%s)", ev.get("id"), ev.get("name"))
                return ev

            if remaining > 30:
                interval = 30
            elif remaining > 10:
                interval = 10
            else:
                interval = 2
            await asyncio.sleep(interval)
            remaining = max(0, remaining - interval / 60.0)

        raise SmarketsError("Event not found on Smarkets after 90 minutes")

    # --- Market discovery ----------------------------------------------
    @staticmethod
    def _extract_line(name: str) -> float | None:
        import re

        m = re.search(r"(\d+(?:\.\d+)?)", name or "")
        return float(m.group(1)) if m else None

    @staticmethod
    def _is_over_under_market(name: str) -> bool:
        low = (name or "").lower()
        return "over" in low or "under" in low or "total" in low

    @staticmethod
    def _contract_side(name: str) -> str | None:
        low = (name or "").lower()
        if "over" in low:
            return "over"
        if "under" in low:
            return "under"
        return None

    async def _get_event_markets(self, event_id: str) -> list[Market]:
        data = await self._request("GET", f"/events/{event_id}/markets/")
        markets: list[Market] = []
        for m in data.get("markets", []) or []:
            name = m.get("name", "")
            if not self._is_over_under_market(name):
                continue
            mid = str(m.get("id"))
            contracts = await self._get_contracts(mid)
            markets.append(
                Market(
                    id=mid,
                    event_id=str(event_id),
                    name=name,
                    line=self._extract_line(name),
                    contracts=contracts,
                )
            )
        return markets

    async def _get_contracts(self, market_id: str) -> list[Contract]:
        data = await self._request("GET", f"/markets/{market_id}/contracts/")
        contracts = []
        for c in data.get("contracts", []) or []:
            name = c.get("name", "")
            contracts.append(
                Contract(id=str(c.get("id")), name=name, side=self._contract_side(name))
            )
        return contracts

    async def get_markets(self, event_id: str) -> list[Market]:
        """Public wrapper: all over/under markets for an event (with contracts)."""
        return await self._get_event_markets(event_id)

    async def get_current_line(self, event_id: str, market_id: str) -> float | None:
        """Re-read an event's markets and return the current line for a market.

        Models a live-shifting line: if the line for ``market_id`` is no longer
        offered, fall back to the closest active over/under line so monitoring
        can detect a shift.
        """
        markets = await self._get_event_markets(event_id)
        for m in markets:
            if m.id == str(market_id):
                return m.line
        # Market gone; return the active line nearest in name, if any.
        lines = [m.line for m in markets if m.line is not None]
        return lines[0] if lines else None

    async def wait_for_markets(
        self,
        event_id: str,
        *,
        minutes_until: int | None,
        line_filter,
        timeout_minutes: int = 90,
    ) -> list[Market]:
        """Poll for over/under markets to appear and return those passing the
        line filter the moment they're available.

        Cadence: every 5s until <5 min to start, then every 1s.
        ``line_filter(market)`` -> bool decides acceptance.
        """
        deadline = time.monotonic() + timeout_minutes * 60
        remaining = minutes_until if minutes_until is not None else 60

        while time.monotonic() < deadline:
            try:
                markets = await self._get_event_markets(event_id)
            except SmarketsError as exc:
                logger.warning("Market fetch error: %s", exc)
                markets = []

            passing = [m for m in markets if line_filter(m)]
            if passing:
                return passing

            interval = 5 if remaining > 5 else 1
            await asyncio.sleep(interval)
            remaining = max(0, remaining - interval / 60.0)

        return []

    # --- Order book -----------------------------------------------------
    async def get_order_book(self, market_id: str, contract_id: str) -> OrderBook:
        """Return best available lay price/volume for a contract.

        We back by consuming resting LAY liquidity, so the best price to back
        at is the best offered lay.
        """
        data = await self._request(
            "GET", f"/markets/{market_id}/quotes/"
        )
        quotes = (data.get("quotes") or {}).get(str(contract_id), {})
        # Smarkets quote structure: {"offers": [...], "bids": [...]} per side.
        # "offers" are available to back (someone laying).
        offers = quotes.get("offers") or quotes.get("offer") or []
        if not offers:
            return OrderBook(contract_id=contract_id, best_lay_price=None,
                             best_lay_volume=0.0, raw=quotes)
        best = offers[0]
        price = best.get("price")
        decimal_price = smarkets_price_to_decimal(price) if price else None
        # quantity is in pence -> euros
        volume = (best.get("quantity") or 0) / 100.0
        return OrderBook(
            contract_id=contract_id,
            best_lay_price=decimal_price,
            best_lay_volume=volume,
            raw=quotes,
        )

    # --- Order placement / management ----------------------------------
    async def place_back_order(
        self,
        market_id: str,
        contract_id: str,
        *,
        decimal_price: float,
        stake_eur: float,
    ) -> OrderResult:
        """Place a BACK (buy) order. ``decimal_price`` is the worst acceptable
        price; the exchange matches at best available.
        """
        if self.paper_mode:
            self._paper_seq += 1
            oid = f"paper-{self._paper_seq}"
            logger.info(
                "[PAPER] BACK order contract=%s €%.2f @ %.2f", contract_id, stake_eur, decimal_price
            )
            return OrderResult(order_id=oid, status="unmatched", raw={"paper": True})

        payload = {
            "market_id": str(market_id),
            "contract_id": str(contract_id),
            "side": "buy",  # back
            "price": decimal_to_smarkets_price(decimal_price),
            "quantity": eur_to_pence(stake_eur),
            "type": "limit",
        }
        data = await self._request("POST", "/orders/", json=payload)
        order = data.get("order", data)
        return OrderResult(
            order_id=str(order.get("id")) if order.get("id") else None,
            status=order.get("status", "unmatched"),
            raw=data,
        )

    async def cancel_order(self, order_id: str) -> bool:
        if self.paper_mode:
            logger.info("[PAPER] cancel order %s", order_id)
            return True
        await self._request("DELETE", f"/orders/{order_id}/")
        return True

    async def get_order_status(self, order_id: str) -> str:
        """Return one of: unmatched / matched / cancelled."""
        if self.paper_mode:
            # In paper mode we never receive real matches; treat as unmatched.
            return "unmatched"
        data = await self._request("GET", f"/orders/{order_id}/")
        order = data.get("order", data)
        status = order.get("status", "unmatched")
        # Normalise Smarkets statuses to the three we track.
        remaining = order.get("quantity_remaining", order.get("remaining"))
        filled = order.get("quantity_filled", order.get("filled"))
        if status in ("cancelled", "voided", "expired"):
            return "cancelled"
        if (remaining == 0) or status == "matched" or status == "filled":
            return "matched"
        if filled and remaining == 0:
            return "matched"
        return "unmatched"
