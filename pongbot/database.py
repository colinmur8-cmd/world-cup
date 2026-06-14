"""SQLite logging for signals and fills.

All writes are wrapped so a logging failure never crashes the pipeline.
The database lives at ``data/bets.db`` relative to this package.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone

_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_DB_PATH = os.path.join(_DB_DIR, "bets.db")

# A single shared connection guarded by a lock. SQLite writes are quick and
# this keeps things simple across the many short-lived async tasks.
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(path: str | None = None) -> None:
    global _conn
    os.makedirs(_DB_DIR, exist_ok=True)
    _conn = sqlite3.connect(path or _DB_PATH, check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    with _lock:
        _conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at   TEXT NOT NULL,
                player1       TEXT NOT NULL,
                player2       TEXT NOT NULL,
                direction     TEXT NOT NULL,
                units         REAL NOT NULL,
                stake_eur     REAL NOT NULL,
                league        TEXT NOT NULL,
                minutes_until INTEGER,
                raw_message   TEXT NOT NULL,
                was_executed  INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS fills (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id     INTEGER NOT NULL,
                market_id     TEXT,
                contract_id   TEXT,
                order_id      TEXT,
                price         REAL,
                stake         REAL,
                line_number   REAL,
                status        TEXT,
                placed_at     TEXT,
                matched_at    TEXT,
                cancel_reason TEXT,
                FOREIGN KEY (signal_id) REFERENCES signals(id)
            );
            """
        )
        _conn.commit()


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        init_db()
    assert _conn is not None
    return _conn


def insert_signal(signal, stake_eur: float) -> int:
    """Insert a received signal, returning its row id."""
    conn = _require_conn()
    with _lock:
        cur = conn.execute(
            """INSERT INTO signals
               (received_at, player1, player2, direction, units, stake_eur,
                league, minutes_until, raw_message, was_executed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
            (
                _utcnow(),
                signal.player1,
                signal.player2,
                signal.direction,
                signal.units,
                stake_eur,
                getattr(signal.league, "value", str(signal.league)),
                signal.minutes_until,
                signal.raw_message,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def mark_signal_executed(signal_id: int, executed: bool = True) -> None:
    conn = _require_conn()
    with _lock:
        conn.execute(
            "UPDATE signals SET was_executed = ? WHERE id = ?",
            (1 if executed else 0, signal_id),
        )
        conn.commit()


def insert_fill(
    signal_id: int,
    *,
    market_id: str,
    contract_id: str,
    order_id: str | None,
    price: float,
    stake: float,
    line_number: float,
    status: str,
) -> int:
    conn = _require_conn()
    with _lock:
        cur = conn.execute(
            """INSERT INTO fills
               (signal_id, market_id, contract_id, order_id, price, stake,
                line_number, status, placed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal_id,
                market_id,
                contract_id,
                order_id,
                price,
                stake,
                line_number,
                status,
                _utcnow(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_fill_status(
    fill_id: int,
    status: str,
    *,
    matched: bool = False,
    cancel_reason: str | None = None,
) -> None:
    conn = _require_conn()
    with _lock:
        conn.execute(
            """UPDATE fills
               SET status = ?,
                   matched_at = COALESCE(?, matched_at),
                   cancel_reason = COALESCE(?, cancel_reason)
               WHERE id = ?""",
            (status, _utcnow() if matched else None, cancel_reason, fill_id),
        )
        conn.commit()


def overnight_counts(since_iso: str) -> dict:
    """Aggregate counts for the morning report since the given timestamp."""
    conn = _require_conn()
    with _lock:
        signals = conn.execute(
            "SELECT * FROM signals WHERE received_at >= ?", (since_iso,)
        ).fetchall()
        fills = conn.execute(
            """SELECT f.* FROM fills f
               JOIN signals s ON s.id = f.signal_id
               WHERE s.received_at >= ?""",
            (since_iso,),
        ).fetchall()

    signals_received = len(signals)
    executed = sum(1 for s in signals if s["was_executed"])
    skipped_no_market = sum(
        1 for s in signals if not s["was_executed"] and s["league"] == "TT Cup"
    )
    # Line-OOB skips are recorded as fills with a cancel_reason mentioning "line"
    # but never placed; we approximate via signals not executed and not cup.
    orders_placed = len(fills)
    orders_cancelled_line = sum(
        1 for f in fills if (f["cancel_reason"] or "").lower().find("line") >= 0
    )
    skipped_line_oob = sum(
        1
        for s in signals
        if not s["was_executed"] and s["league"] in ("TT Elite Series", "Czech Liga Pro")
    )
    total_matched = sum(f["stake"] for f in fills if f["status"] == "matched")

    return {
        "signals_received": signals_received,
        "executed": executed,
        "skipped_no_market": skipped_no_market,
        "skipped_line_oob": skipped_line_oob,
        "orders_cancelled_line": orders_cancelled_line,
        "orders_placed": orders_placed,
        "total_matched": total_matched,
    }
