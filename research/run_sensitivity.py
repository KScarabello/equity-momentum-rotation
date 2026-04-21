#!/usr/bin/env python3
"""
Run sensitivity analysis for the equity momentum strategy.

Sweeps cost_bps and slippage_bps to show how performance degrades
under realistic friction assumptions.

Run:
    python -m research.run_sensitivity
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from research.walk_forward_momentum import WalkForwardConfig, run_sensitivity
from research.data_stooq import list_stooq_parquets, _infer_symbol_from_filename

STOOQ_DIR = Path("data_cache/stooq")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
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
        raise ValueError(f"Missing required columns {missing}. Have: {list(df.columns)}")

    return df.copy()


def fetch_ohlcv(symbol: str, stooq_dir: Path = STOOQ_DIR) -> pd.DataFrame:
    fp = stooq_dir / f"{symbol}.US.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing parquet for {symbol}: {fp}")
    return _normalize_ohlcv(pd.read_parquet(fp))


def load_data(market_symbol: str = "SPY") -> tuple[dict, pd.DataFrame]:
    print("[INFO] Loading universe from parquet cache")
    files = list_stooq_parquets(STOOQ_DIR)
    symbols = []
    for fp in files:
        sym = _infer_symbol_from_filename(fp)
        if sym and sym not in symbols:
            symbols.append(sym)
    print(f"[INFO] Universe loaded from cache: {len(symbols)}")

    print("[INFO] Loading market proxy OHLCV")
    market_df = fetch_ohlcv(market_symbol)

    print(f"[INFO] Loading OHLCV for {len(symbols)} symbols...")
    symbol_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            symbol_dfs[sym] = fetch_ohlcv(sym)
        except Exception as e:
            print(f"[WARN] Skipping {sym}: {e}")

    if not symbol_dfs:
        raise ValueError("No symbol data loaded successfully.")

    print(f"[INFO] Loaded OHLCV for {len(symbol_dfs)} symbols")
    return symbol_dfs, market_df


def main() -> None:
    print("[INFO] Loading data")
    symbol_dfs, market_df = load_data()

    print("[INFO] Running sensitivity analysis...")
    cfg = WalkForwardConfig()

    df = run_sensitivity(
        symbol_dfs,
        market_df,
        cfg,
        cost_bps_list=[0, 5, 10],
        slippage_bps_list=[0, 2, 5],
    )

    print("\n=== SENSITIVITY RESULTS ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
