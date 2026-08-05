"""Read a plays.txt file and execute each signal at the right time.

Usage:
    python3 run_plays.py plays.txt

File format (one play per line):
    LEAGUE | HH:MM AM/PM | Player1 vs Player2 | DIRECTION | NUu

Examples:
    TT ELITE | 5:05 PM | Kolek M. vs Kolodziej K. | UNDER | 1U
    TT ELITE | 5:40 PM | Smolka K. vs Gajda R. | OVER | 1U
    TT CUP   | 4:00 PM | Guzy K. vs Gola B. | UNDER | 2U
    TT ELITE | 6:55 PM | Warzecha M. vs Barton A. | UNDER | 1.0U

Lines starting with # are comments and are ignored.
Blank lines are ignored.

The bot holds each play until 10 minutes before its scheduled time,
then starts searching Smarkets and places orders as per the full spec.
DMs are sent for every signal, execution, and any errors.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from datetime import datetime, timedelta, timezone

import discord

import database as db
from config import config
from executor import BetExecutor
from notifier import Notifier
from parser import League, ParsedSignal, ParseError
from smarkets import SmarketsClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("pongbot.run_plays")

# How many minutes before match time to start searching Smarkets.
SEARCH_LEAD_MINUTES = 10


# ---------------------------------------------------------------------------
# Parse the simple pipe-delimited format
# ---------------------------------------------------------------------------

_LEAGUE_MAP = {
    "tt elite":   League.ELITE,
    "tt elite series": League.ELITE,
    "tt cup":     League.CUP,
    "tt cup series": League.CUP,
    "czech liga pro": League.LIGA_PRO,
    "liga pro":   League.LIGA_PRO,
}

_TIME_RE = re.compile(
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})\s*(?P<ampm>AM|PM)", re.IGNORECASE
)
_UNITS_RE = re.compile(r"(?P<units>\d+(?:\.\d+)?)\s*[Uu]")


def _classify_league(raw: str) -> League:
    low = raw.strip().lower()
    for key, league in _LEAGUE_MAP.items():
        if key in low:
            return league
    # Fallback to keyword matching (matches parser.py logic)
    if "elite" in low:
        return League.ELITE
    if "czech" in low or "liga" in low or "pro" in low:
        return League.LIGA_PRO
    if "cup" in low:
        return League.CUP
    return League.UNKNOWN


def _parse_match_time(time_str: str, now: datetime) -> datetime | None:
    """Parse 'H:MM AM/PM' into a UTC datetime (assumes same day or next day)."""
    m = _TIME_RE.search(time_str)
    if not m:
        return None
    hour = int(m.group("hour"))
    minute = int(m.group("minute"))
    ampm = m.group("ampm").upper()
    if ampm == "PM" and hour != 12:
        hour += 12
    if ampm == "AM" and hour == 12:
        hour = 0

    # Times on the dashboard are in the local timezone of the user's machine;
    # but we work in UTC internally. We use system local time to compute
    # minutes_until and let asyncio.sleep handle the wait.
    local_now = datetime.now()
    candidate = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate < local_now - timedelta(minutes=5):
        # Probably tomorrow (e.g. a 12:55 AM play listed late at night)
        candidate += timedelta(days=1)
    return candidate


def parse_plays_file(path: str) -> list[ParsedSignal]:
    """Parse the pipe-delimited plays file into ParsedSignal objects."""
    signals: list[ParsedSignal] = []
    now = datetime.now()

    with open(path, encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 5:
                logger.warning("Line %d: expected 5 fields, got %d — skipping: %r",
                               lineno, len(parts), line)
                continue

            league_raw, time_raw, match_raw, direction_raw, units_raw = parts[:5]

            # Direction
            direction = direction_raw.strip().upper()
            if direction not in ("OVER", "UNDER"):
                logger.warning("Line %d: unknown direction %r — skipping", lineno, direction_raw)
                continue

            # Units
            um = _UNITS_RE.search(units_raw)
            if not um:
                logger.warning("Line %d: can't parse units from %r — skipping", lineno, units_raw)
                continue
            units = float(um.group("units"))

            # Players
            vs_match = re.split(r"\s+vs\.?\s+", match_raw, flags=re.IGNORECASE, maxsplit=1)
            if len(vs_match) != 2:
                logger.warning("Line %d: can't split players from %r — skipping", lineno, match_raw)
                continue
            player1, player2 = vs_match

            # League
            league = _classify_league(league_raw)
            skip_reason: str | None = None
            if league == League.CUP:
                skip_reason = "TT Cup — no Smarkets market"
            elif league == League.UNKNOWN:
                skip_reason = f"Unrecognised league: {league_raw!r}"

            # Match time → minutes until
            match_time = _parse_match_time(time_raw, now)
            if match_time is None:
                logger.warning("Line %d: can't parse time from %r — skipping", lineno, time_raw)
                continue
            minutes_until = max(0, int((match_time - now).total_seconds() / 60))

            raw_message = line
            signals.append(
                ParsedSignal(
                    player1=player1.strip(),
                    player2=player2.strip(),
                    direction=direction,
                    units=units,
                    minutes_until=minutes_until,
                    league=league,
                    raw_message=raw_message,
                    skip_reason=skip_reason,
                )
            )
            logger.info(
                "Parsed: %s vs %s | %s | %.1fu | T-%d min | %s",
                player1.strip(), player2.strip(), direction, units,
                minutes_until, league.value,
            )

    return signals


# ---------------------------------------------------------------------------
# Execution runner (no Discord bot needed — uses a minimal Discord client
# just for sending DMs)
# ---------------------------------------------------------------------------

class DMClient(discord.Client):
    """Minimal Discord client used only to send DMs to the owner."""

    def __init__(self, signals: list[ParsedSignal]):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self._signals = signals
        self._ready = asyncio.Event()

    async def on_ready(self) -> None:
        logger.info("Discord connected as %s", self.user)
        self._ready.set()

    async def run_all(self) -> None:
        await self._ready.wait()

        notifier = Notifier(self, config.bot_owner_user_id, paper_mode=config.paper_mode)
        smarkets = SmarketsClient(
            username=config.smarkets_username,
            password=config.smarkets_password,
            session_token=config.smarkets_api_token,
            paper_mode=config.paper_mode,
        )
        await smarkets.connect()

        try:
            tasks = []
            seen: set[str] = set()

            for signal in self._signals:
                # Skip leagues with no market
                if signal.skip_reason is not None:
                    await notifier.signal_skipped(signal, signal.skip_reason)
                    continue

                # Duplicate detection
                key = signal.match_key
                if key in seen:
                    logger.info("Duplicate skipped: %s", key)
                    continue
                seen.add(key)

                tasks.append(
                    asyncio.create_task(
                        self._run_one(signal, notifier, smarkets)
                    )
                )

            if tasks:
                await asyncio.gather(*tasks)
            else:
                logger.info("No executable signals.")
        finally:
            await smarkets.close()
            await self.close()

    async def _run_one(
        self,
        signal: ParsedSignal,
        notifier: Notifier,
        smarkets: SmarketsClient,
    ) -> None:
        """Wait until SEARCH_LEAD_MINUTES before match time, then execute."""
        try:
            wait_minutes = max(0, (signal.minutes_until or 0) - SEARCH_LEAD_MINUTES)
            if wait_minutes > 0:
                logger.info(
                    "%s vs %s: waiting %d min before searching Smarkets",
                    signal.player1, signal.player2, wait_minutes,
                )
                await asyncio.sleep(wait_minutes * 60)

            # Recalculate actual minutes_until at execution time
            # so Smarkets polling cadence is accurate
            signal.minutes_until = min(signal.minutes_until or 0, SEARCH_LEAD_MINUTES)

            stake = signal.stake_eur(config.unit_size)
            signal_id = db.insert_signal(signal, stake)
            await notifier.signal_received(signal, stake)

            executor = BetExecutor(smarkets, notifier, signal, signal_id)
            await executor.run()

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error running signal %s vs %s", signal.player1, signal.player2)
            await notifier.error(signal, f"Unexpected error: {exc}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 run_plays.py plays.txt")
        sys.exit(1)

    plays_file = sys.argv[1]
    signals = parse_plays_file(plays_file)

    if not signals:
        print("No valid signals found in file.")
        sys.exit(0)

    print(f"\nLoaded {len(signals)} signal(s):")
    for s in signals:
        status = f"SKIP ({s.skip_reason})" if s.skip_reason else f"T-{s.minutes_until}min"
        print(f"  {s.player1} vs {s.player2} | {s.direction} | {s.units:g}u | {s.league.value} | {status}")

    print(f"\nRunning in {'PAPER' if config.paper_mode else 'LIVE'} mode.")
    print("DMs will be sent to your Discord. Press Ctrl+C to abort.\n")

    client = DMClient(signals)

    async def _run():
        async with client:
            await asyncio.gather(
                client.start(config.discord_token),
                client.run_all(),
            )

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\nAborted.")


if __name__ == "__main__":
    main()
