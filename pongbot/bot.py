"""PongBot Smarkets Executor — Discord listener and entry point.

Reads the configured PongBot channel, parses each message as a bet signal and
runs the execution pipeline in its own async task so simultaneous signals never
block each other. Sends all notifications via DM to the bot owner. Runs a daily
overnight report at 08:00 UTC.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time as dtime, timedelta, timezone

import discord

import database as db
from config import config
from executor import BetExecutor
from notifier import Notifier
from parser import ParseError, parse_signal
from smarkets import SmarketsClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("pongbot.bot")


class PongBot(discord.Client):
    def __init__(self, **kwargs):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, **kwargs)

        self.notifier: Notifier | None = None
        self.smarkets = SmarketsClient(config.smarkets_api_token, paper_mode=config.paper_mode)
        # Active match keys for duplicate detection.
        self._active: set[str] = set()
        self._report_task: asyncio.Task | None = None

    async def setup_hook(self) -> None:
        db.init_db()
        self.notifier = Notifier(self, config.bot_owner_user_id, paper_mode=config.paper_mode)
        await self.smarkets.connect()
        self._report_task = self.loop.create_task(self._overnight_report_loop())

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (paper_mode=%s)", self.user, config.paper_mode)

    async def on_message(self, message: discord.Message) -> None:
        # Only the configured channel; ignore our own messages.
        if message.channel.id != config.pongbot_channel_id:
            return
        if self.user and message.author.id == self.user.id:
            return
        # Never react or post in the channel — just process.
        self.loop.create_task(self._handle_signal(message.content))

    async def _handle_signal(self, content: str) -> None:
        """Process a single signal end-to-end. Never raises."""
        signal = None
        try:
            try:
                signal = parse_signal(content)
            except ParseError as exc:
                logger.warning("Unparseable message: %s", exc)
                return  # Not a valid signal; nothing to DM about.

            stake = signal.stake_eur(config.unit_size)
            signal_id = db.insert_signal(signal, stake)

            # Skip leagues with no Smarkets market (TT Cup / unknown).
            if signal.skip_reason is not None:
                await self.notifier.signal_skipped(signal, signal.skip_reason)
                return

            # Duplicate detection by player names (+direction).
            key = signal.match_key
            if key in self._active:
                logger.info("Duplicate signal ignored: %s", key)
                return
            self._active.add(key)

            try:
                await self.notifier.signal_received(signal, stake)
                executor = BetExecutor(self.smarkets, self.notifier, signal, signal_id)
                await executor.run()
            finally:
                self._active.discard(key)

        except Exception as exc:  # noqa: BLE001 — never crash on a bad signal
            logger.exception("Error processing signal")
            if self.notifier:
                await self.notifier.error(signal, f"Unexpected error: {exc}")

    # --- Daily overnight report ----------------------------------------
    async def _overnight_report_loop(self) -> None:
        await self.wait_until_ready()
        while not self.is_closed():
            await asyncio.sleep(self._seconds_until_0800_utc())
            try:
                since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
                counts = db.overnight_counts(since)
                date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if self.notifier:
                    await self.notifier.morning_report(date_str, counts)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to send overnight report")

    @staticmethod
    def _seconds_until_0800_utc() -> float:
        now = datetime.now(timezone.utc)
        target = datetime.combine(now.date(), dtime(8, 0), tzinfo=timezone.utc)
        if now >= target:
            target += timedelta(days=1)
        return (target - now).total_seconds()

    async def close(self) -> None:
        await self.smarkets.close()
        await super().close()


def main() -> None:
    bot = PongBot()
    bot.run(config.discord_token)


if __name__ == "__main__":
    main()
