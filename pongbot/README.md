# PongBot Smarkets Executor

A Python bot that runs 24/7 on a Linux VPS. It reads a Discord channel where
PongBot posts table-tennis betting signals, finds the corresponding markets on
the Smarkets exchange and automatically places back orders up to a configured
stake limit.

> Smarkets API access must be approved separately. Run in **paper mode**
> (`PAPER_MODE=true`, the default) until you are ready to place real orders.

## Layout

```
pongbot/
├── bot.py          # Discord listener, entry point
├── parser.py       # Signal parsing
├── smarkets.py     # Smarkets API client
├── executor.py     # Fill logic and line monitoring
├── database.py     # SQLite logging
├── notifier.py     # Discord DM sender
├── config.py       # Load and validate .env
├── requirements.txt
├── .env.example
└── data/
    └── bets.db     # created at runtime
```

## Setup

```bash
cd pongbot
cp .env.example .env      # fill in tokens / IDs
pip install -r requirements.txt
python3 bot.py
```

## Deployment (Ubuntu 22.04 + PM2)

```bash
pip install discord.py aiohttp python-dotenv
pm2 start bot.py --interpreter python3 --name pongbot
pm2 startup
pm2 save
```

## Behaviour summary

- **Signal parsing** — `P1 vs P2 | DIRECTION | Nu`, plus minutes-to-start and
  league line. League map: `elite` → TT Elite Series, `czech`/`liga`/`pro` →
  Czech Liga Pro (both executed); `cup` → TT Cup (skipped). Anything else is
  skipped with a DM.
- **Stake** — `units × UNIT_SIZE` (default €200/unit).
- **Line filter** — OVER never placed above `MAX_LINE_OVER` (84.5); UNDER never
  below `MIN_LINE_UNDER` (59.5). Resting orders are cancelled if the live line
  moves more than `LINE_CANCEL_THRESHOLD` (20) points against the order.
- **Execution** — Phase 1 sweeps available lay liquidity in
  `[MIN_PRICE, MAX_PRICE]`; Phase 2 rests the remainder evenly at `MIN_PRICE`;
  Phase 3 monitors statuses (500ms) and the live line (30s), stops at 98% of the
  cap, and cancels everything after 2 hours.
- **Notifications** — all sent via DM to `BOT_OWNER_USER_ID`. Nothing is ever
  posted in a server channel. Paper mode prefixes every DM with `[PAPER]`.
- **Resilience** — each signal runs in its own task with a catch-all; duplicate
  matches are ignored; Smarkets 429s back off 5s; Discord reconnects
  automatically.

## Paper mode

With `PAPER_MODE=true` the full pipeline runs but no real orders are placed:
phase-1 sweeps are treated as matched for the summary, every DM is prefixed
`[PAPER]`, and the database records what *would* have been placed.
