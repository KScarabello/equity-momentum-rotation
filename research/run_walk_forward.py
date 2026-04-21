#!/usr/bin/env python3
"""
Run walk-forward validation for the equity momentum strategy.

Run:
    python -m research.run_walk_forward
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from research.walk_forward_momentum import WalkForwardConfig, walk_forward_validate
from research.data_stooq import list_stooq_parquets, _infer_symbol_from_filename

STOOQ_DIR = Path("data_cache/stooq")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize index/columns and require close/volume fields."""
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})

    for a in ["adj close", "adjclose", "adj_close"]:
        if a in df.columns and "close" not in df.columns:
            df["close"] = df[a]

    required = ["close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing}. Have: {list(df.columns)}"
        )

    return df.copy()


def fetch_ohlcv(symbol: str, stooq_dir: Path = STOOQ_DIR) -> pd.DataFrame:
    """Load one symbol from Stooq parquet cache."""
    fp = stooq_dir / f"{symbol}.US.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing parquet for {symbol}: {fp}")
    df = pd.read_parquet(fp)
    return _normalize_ohlcv(df)


def build_universe_from_stooq(
    stooq_dir: Path = STOOQ_DIR, limit: int | None = None
) -> list[str]:
    files = list_stooq_parquets(stooq_dir, limit=limit)
    syms: list[str] = []
    for fp in files:
        sym = _infer_symbol_from_filename(fp)
        if sym and sym not in syms:
            syms.append(sym)
    return syms


def main():
    cfg = WalkForwardConfig(
        train_years=3,
        test_months=6,
        step_months=6,
        positions=12,
        universe_top_n=800,
        rebalance_weekday=0,
        starting_cash=100_000.0,
        liq_lookback=60,
        mom_3m=63,
        mom_6m=126,
        mom_12m=252,
        w_3m=0.60,
        w_6m=0.30,
        w_12m=0.10,
        veto_if_12m_return_below=0.0,
        market_symbol="SPY",
        market_sma_days=200,
        risk_on_buffer=0.0,
        cost_bps=0.0,
    )

    print("[INFO] Loading universe from parquet cache")
    symbols = build_universe_from_stooq(STOOQ_DIR)
    print(f"[INFO] Universe loaded from cache: {len(symbols)} symbols")

    print("[INFO] Loading market proxy OHLCV")
    market_df = fetch_ohlcv(cfg.market_symbol)

    print(f"[INFO] Loading OHLCV for {len(symbols)} symbols")
    symbol_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            symbol_dfs[sym] = fetch_ohlcv(sym)
        except Exception as e:
            print(f"[WARN] Skipping {sym}: {e}")

    if not symbol_dfs:
        raise ValueError("No symbol data loaded successfully.")

    print(f"[INFO] Loaded OHLCV for {len(symbol_dfs)} symbols")
    print("[INFO] Running walk-forward validation")

    results_df, equity_oos, debug = walk_forward_validate(
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=cfg,
        return_debug=True,
    )

    print("\n=== WALK-FORWARD RESULTS (PER WINDOW) ===")
    print(results_df.to_string(index=False))

    print("\n=== OVERALL OUT-OF-SAMPLE EQUITY ===")
    print(f"Start equity: {equity_oos.iloc[0]:.2f}")
    print(f"End equity:   {equity_oos.iloc[-1]:.2f}")

    print("\n=== OOS EQUITY DIAGNOSTICS ===")
    print(f"First date in equity series: {equity_oos.index.min().date()}")
    print(f"Last date in equity series:  {equity_oos.index.max().date()}")
    print(f"Number of rows: {len(equity_oos)}")
    print(f"Backtest OOS date range: {equity_oos.index.min().date()} -> {equity_oos.index.max().date()}")
    print(f"First rebalance date:   {debug['first_rebalance_date']}")
    print(f"Last rebalance date:    {debug['last_rebalance_date']}")
    print(f"Number of rebalances:   {debug['num_rebalances']}")
    print(f"Number of trades:       {debug['total_trades']}")
    print(f"Total selected symbols across all rebalances: {debug['total_selected_symbols_across_rebalances']}")
    print(f"Number of days with non-zero exposure: {debug['non_zero_exposure_days']}")

    print(f"Min equity: {equity_oos.min():.2f}")
    print(f"Max equity: {equity_oos.max():.2f}")
    print(f"Unique equity values: {equity_oos.nunique()}")

    print("\nFirst 10 equity values:")
    print(equity_oos.head(10).to_string())

    print("\nLast 10 equity values:")
    print(equity_oos.tail(10).to_string())

    results_df.to_csv("walk_forward_windows.csv", index=False)
    equity_oos.to_csv("walk_forward_equity_oos.csv", header=True)

    print("\n[INFO] Saved outputs")
    print("  - walk_forward_windows.csv")
    print("  - walk_forward_equity_oos.csv")


if __name__ == "__main__":
    main()
