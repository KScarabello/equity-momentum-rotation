# Equity Momentum Rotation (12–1)

This repository contains a research backtest of a classic 12–1 equity momentum
strategy, evaluated on historical U.S. equity data (Stooq) and benchmarked
against SPY buy-and-hold.

## Strategy

- Signal: 12–1 momentum (12-month lookback, 1-month skip)
- Universe: U.S. equities (Stooq parquet cache)
- Portfolio: Top-N ranked equities, equal-weighted
- Rebalance: Every N trading days (biweekly by default)
- Costs: Applied on rebalance days, proportional to turnover
- Risk-off: Optional absolute momentum filter

## Results (2009–2025)

- Strategy CAGR: ~21%
- SPY CAGR: ~14%
- Excess return: ~7% annualized
- Sharpe: Higher than SPY
- Drawdown: Comparable

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
