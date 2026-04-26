# Equity Momentum Rotation

This repository contains a research backtest of an equity momentum
strategy, evaluated on historical U.S. equity data (Stooq) and benchmarked
against a buy-and-hold benchmark.

## Strategy

- Signal: Multi-period momentum (configurable lookback windows)
- Universe: U.S. equities (Stooq parquet cache)
- Portfolio: Top-N ranked equities, equal-weighted
- Rebalance: Every N trading days (biweekly by default)
- Costs: Applied on rebalance days, proportional to turnover
- Risk-off: Optional absolute momentum filter

## Performance

Results vary based on configuration, universe, and time period.
Run `python -m research.run_backtest` to generate metrics for your local data.

## Correctness Guarantees

This project includes unit tests that lock:

- Exact momentum math (no indexing drift)
- Rebalance timing (no lookahead)
- Transaction cost application
- Absolute momentum risk-off behavior

Any unintended logic change will fail tests.

## Disclaimer

This is a research project, not investment advice.

## Data Setup (Required)

This project expects historical equity price data in **Stooq parquet format**.
The data directory is intentionally **not committed to git**.

Each parquet file should contain daily OHLCV data for a single symbol and
be indexed by date.

---

### How to Obtain the Data

You can download historical U.S. equity data from **Stooq**:

- Stooq database: https://stooq.com/db/h/

Download the daily U.S. equities dataset and extract the CSV files you want
to use (e.g. large-cap U.S. stocks).

---

### Convert CSV Files to Parquet

Example conversion script:

```python
import pandas as pd
from pathlib import Path

csv_dir = Path("stooq_csv")
out_dir = Path("data_cache/stooq")
out_dir.mkdir(parents=True, exist_ok=True)

for csv in csv_dir.glob("*.csv"):
    df = pd.read_csv(csv)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    df = df.sort_index()

    out_path = out_dir / f"{csv.stem}.parquet"
    df.to_parquet(out_path)

    print(f"Wrote {out_path}")
```

### Verify Data Load

Run:

```bash
python -m research.run_backtest
```

## Intentionally Excluded From Git

The following are intentionally local-only and should not be committed:

- Market data cache in `data_cache/`
- Local secrets such as `.env`
- Runtime logs in `logs/`
- Generated research outputs (CSV/JSON/TXT summaries)
- Machine-local caches such as `.pytest_cache/` and notebook checkpoints

Generated outputs can be retained locally for analysis, but should be treated as
artifacts rather than source code.

## Live Alpaca Rebalance (Safety-First)

The live runner executes one rebalance cycle and is conservative by default.

- Default mode is `DRY_RUN=true` (plan only, no live order submits).
- Buy orders are submitted as **notional dollar** limit orders.
- Sell orders use **exact held qty** from Alpaca positions (fractional safe).
- Orders are only allowed during the configured ET execution window.

### Required environment variables

Set these in `.env` (or your shell):

```bash
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets

DRY_RUN=true
FIRST_RUN_LIQUIDATE_ALL=false
EXECUTION_WINDOW_START_ET=09:50
EXECUTION_WINDOW_END_ET=10:05
MAX_DEPLOYMENT_PCT=0.60
MIN_TRADE_NOTIONAL=10
BUY_LIMIT_BUFFER_BPS=10
SELL_LIMIT_BUFFER_BPS=10
MAX_ORDER_COUNT=40
MAX_POSITIONS=12
```

### Dry run (recommended default)

```bash
python3 -m live.run_alpaca_live_trader --dry-run --verbose
```

This computes targets, reconciles current vs target holdings, writes audit artifacts,
and submits no orders.

### First-run liquidation mode

Use this when the account has legacy positions that must be cleared before strategy deployment.

```bash
FIRST_RUN_LIQUIDATE_ALL=true DRY_RUN=false python3 -m live.run_alpaca_live_trader --live --verbose
```

Behavior:

- Plans full exits for all existing equity positions.
- Uses exact held qty for each sell (fractional-safe).
- Logs liquidation progress and stores `first_run_liquidation_done` in state.

After liquidation is confirmed, disable `FIRST_RUN_LIQUIDATE_ALL`.

### Live mode

```bash
DRY_RUN=false FIRST_RUN_LIQUIDATE_ALL=false python3 -m live.run_alpaca_live_trader --live --verbose
```

Execution order is sells first, then buys. Open-order idempotency checks prevent
duplicate submit for the same symbol/direction in the same cycle.

### Audit artifacts

Each cycle writes files under `logs/`:

- `live_run_<run_id>.log` (structured event logs)
- `rebalance_summary_<run_id>.json` and `.txt`
- `rebalance_positions_<run_id>.csv`
- `rebalance_orders_<run_id>.csv`
- `live_execution_<run_id>.csv` (live mode only)
