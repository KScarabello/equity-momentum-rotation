#!/usr/bin/env python3
"""
Run a position-count sweep for the equity momentum strategy.

Compares walk-forward out-of-sample performance for different numbers of
held positions while keeping signal generation, weighting, and trading logic unchanged.

Run:
    python -m research.run_position_sweep
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from research.walk_forward_momentum import WalkForwardConfig, walk_forward_validate
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


def load_data(market_symbol: str = "SPY") -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    print("[INFO] Loading universe from parquet cache")
    files = list_stooq_parquets(STOOQ_DIR)
    symbols: list[str] = []
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
        except Exception as exc:
            print(f"[WARN] Skipping {sym}: {exc}")

    if not symbol_dfs:
        raise ValueError("No symbol data loaded successfully.")

    print(f"[INFO] Loaded OHLCV for {len(symbol_dfs)} symbols")
    return symbol_dfs, market_df


def _summarize_equity_curve(equity_oos: pd.Series) -> dict[str, float]:
    rets = equity_oos.pct_change().dropna()
    years = len(equity_oos) / 252.0
    cagr = (
        (equity_oos.iloc[-1] / equity_oos.iloc[0]) ** (1.0 / max(years, 1e-9)) - 1.0
    )
    sharpe = rets.mean() / rets.std() * np.sqrt(252.0) if rets.std() > 0 else 0.0
    max_drawdown = (equity_oos / equity_oos.cummax() - 1.0).min()
    return {
        "final_equity": float(equity_oos.iloc[-1]),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "max_drawdown_pct": round(float(max_drawdown) * 100, 2),
    }


def main() -> None:
    position_list = [8, 10, 12, 15, 20]

    print("[INFO] Loading data")
    symbol_dfs, market_df = load_data()

    rows = []
    for positions in position_list:
        print(f"[INFO] Running walk-forward for positions={positions}...")
        cfg = WalkForwardConfig(
            positions=positions,
            cost_bps=5.0,
            slippage_bps=2.0,
            rebalance_interval_weeks=2,
            min_rebalance_weight_change=0.0,
        )

        results_df, equity_oos = walk_forward_validate(
            symbol_dfs=symbol_dfs,
            market_df=market_df,
            cfg=cfg,
        )

        summary = _summarize_equity_curve(equity_oos)
        avg_turnover = (
            results_df["avg_turnover"].mean()
            if "avg_turnover" in results_df.columns
            else float("nan")
        )
        total_costs = (
            results_df["total_costs"].sum()
            if "total_costs" in results_df.columns
            else float("nan")
        )

        rows.append(
            {
                "positions": positions,
                **summary,
                "avg_turnover_pct": round(float(avg_turnover) * 100, 2),
                "total_costs": round(float(total_costs), 2),
            }
        )

    df = pd.DataFrame(rows)

    print("\n=== POSITION SWEEP RESULTS ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
