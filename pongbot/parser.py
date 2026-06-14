"""Parse PongBot Discord signal messages into structured signals.

Expected message format (3 lines)::

    Kolodziej K. vs Rutkowski M. | UNDER | 1u
    1:30 PM EDT (63 Minutes)
    TT ELITE SERIES · 1:30 PM
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class League(str, Enum):
    ELITE = "TT Elite Series"
    LIGA_PRO = "Czech Liga Pro"
    CUP = "TT Cup"
    UNKNOWN = "Unknown"


# Leagues we actually place bets on.
EXECUTABLE_LEAGUES = {League.ELITE, League.LIGA_PRO}


@dataclass
class ParsedSignal:
    player1: str
    player2: str
    direction: str  # "OVER" or "UNDER"
    units: float
    minutes_until: int | None
    league: League
    raw_message: str

    # Set when parsing produced an actionable signal but the league
    # has no Smarkets market or is unrecognised.
    skip_reason: str | None = None

    @property
    def is_executable(self) -> bool:
        return self.skip_reason is None and self.league in EXECUTABLE_LEAGUES

    def stake_eur(self, unit_size: float) -> float:
        return self.units * unit_size

    @property
    def match_key(self) -> str:
        """Stable key for duplicate detection, order-independent on players."""
        names = sorted([self.player1.lower().strip(), self.player2.lower().strip()])
        return f"{names[0]}|{names[1]}|{self.direction}"


class ParseError(ValueError):
    """Raised when a message cannot be parsed into a signal at all."""


# --- Regexes ------------------------------------------------------------
_HEADER_RE = re.compile(
    r"^(?P<p1>.+?)\s+vs\.?\s+(?P<p2>.+?)\s*\|\s*"
    r"(?P<dir>OVER|UNDER)\s*\|\s*"
    r"(?P<units>\d+(?:\.\d+)?)\s*u\b",
    re.IGNORECASE,
)
_MINUTES_RE = re.compile(r"\((?P<minutes>\d+)\s*Minutes?\)", re.IGNORECASE)


def _classify_league(message: str) -> League:
    text = message.lower()
    if "elite" in text:
        return League.ELITE
    if "czech" in text or "liga" in text or "pro" in text:
        return League.LIGA_PRO
    if "cup" in text:
        return League.CUP
    return League.UNKNOWN


def parse_signal(message: str) -> ParsedSignal:
    """Parse a raw Discord message into a ``ParsedSignal``.

    Raises ``ParseError`` if the header line (players/direction/units) is
    not present. League/skip handling is set on the returned object so the
    caller can DM the user with a reason.
    """
    if not message or not message.strip():
        raise ParseError("Empty message")

    header = _HEADER_RE.search(message)
    if not header:
        raise ParseError("Could not find 'P1 vs P2 | DIRECTION | Nu' header")

    player1 = header.group("p1").strip()
    player2 = header.group("p2").strip()
    direction = header.group("dir").upper()
    units = float(header.group("units"))

    minutes_match = _MINUTES_RE.search(message)
    minutes_until = int(minutes_match.group("minutes")) if minutes_match else None

    league = _classify_league(message)

    skip_reason: str | None = None
    if league == League.CUP:
        skip_reason = "TT Cup — no Smarkets market"
    elif league == League.UNKNOWN:
        skip_reason = "Unrecognised league — no Smarkets market mapping"

    return ParsedSignal(
        player1=player1,
        player2=player2,
        direction=direction,
        units=units,
        minutes_until=minutes_until,
        league=league,
        raw_message=message,
        skip_reason=skip_reason,
    )
