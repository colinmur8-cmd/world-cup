"""Load and validate configuration from the environment / .env file.

Every tunable lives in .env. This module loads it once at import time,
coerces types, validates required secrets and exposes a single ``config``
object that the rest of the bot imports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load .env from the package directory regardless of the current working dir
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_ENV_PATH)
# Also fall back to a .env in the CWD (e.g. when launched by PM2)
load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _get_str(name: str, *, required: bool = False, default: str = "") -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Invalid float for {name!r}: {raw!r}") from exc


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Invalid integer for {name!r}: {raw!r}") from exc


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    # --- Secrets / identifiers (required) -------------------------------
    discord_token: str = field(default_factory=lambda: _get_str("DISCORD_TOKEN", required=True))
    pongbot_channel_id: int = field(
        default_factory=lambda: int(_get_str("PONGBOT_CHANNEL_ID", required=True))
    )
    # Smarkets auth. The REST API is session-based: log in with your
    # account credentials to mint a Session-Token. A pre-minted token may
    # be supplied directly instead (it will be used until it expires).
    smarkets_username: str = field(default_factory=lambda: _get_str("SMARKETS_USERNAME"))
    smarkets_password: str = field(default_factory=lambda: _get_str("SMARKETS_PASSWORD"))
    smarkets_api_token: str = field(default_factory=lambda: _get_str("SMARKETS_API_TOKEN"))
    bot_owner_user_id: int = field(
        default_factory=lambda: int(_get_str("BOT_OWNER_USER_ID", required=True))
    )

    # --- Stake / safety limits ------------------------------------------
    unit_size: float = field(default_factory=lambda: _get_float("UNIT_SIZE", 200.0))
    min_price: float = field(default_factory=lambda: _get_float("MIN_PRICE", 1.50))
    max_price: float = field(default_factory=lambda: _get_float("MAX_PRICE", 2.50))
    min_liquidity: float = field(default_factory=lambda: _get_float("MIN_LIQUIDITY", 10.0))
    max_orders: int = field(default_factory=lambda: _get_int("MAX_ORDERS", 20))
    max_line_over: float = field(default_factory=lambda: _get_float("MAX_LINE_OVER", 84.5))
    min_line_under: float = field(default_factory=lambda: _get_float("MIN_LINE_UNDER", 59.5))
    line_cancel_threshold: float = field(
        default_factory=lambda: _get_float("LINE_CANCEL_THRESHOLD", 20.0)
    )

    # --- Mode ------------------------------------------------------------
    paper_mode: bool = field(default_factory=lambda: _get_bool("PAPER_MODE", True))

    def validate(self) -> None:
        if self.min_price < 1.0:
            raise ConfigError("MIN_PRICE must be >= 1.0 (decimal odds)")
        if self.max_price < self.min_price:
            raise ConfigError("MAX_PRICE must be >= MIN_PRICE")
        if self.unit_size <= 0:
            raise ConfigError("UNIT_SIZE must be positive")
        # Need either login credentials or a pre-minted session token.
        if not self.smarkets_api_token and not (
            self.smarkets_username and self.smarkets_password
        ):
            raise ConfigError(
                "Set SMARKETS_USERNAME + SMARKETS_PASSWORD (or SMARKETS_API_TOKEN)"
            )


# Singleton used across the package.
config = Config()
config.validate()
