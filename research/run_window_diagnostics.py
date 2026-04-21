#!/usr/bin/env python3
"""
Window-level diagnostics for the equity momentum walk-forward strategy.

For each OOS window, computes:
  - strategy metrics (cagr, sharpe, max_drawdown)
  - market performance (total return, vol, drawdown)
  - relative performance (strategy return vs market)
  - participation ratio (% of days strategy was invested)
  - concentration of losses (sum of worst 5 daily returns)
  - regime label (bull / bear)

Results are sorted by CAGR ascending so worst windows appear first.

Run:
    python -m research.run_window_diagnostics
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from research.walk_forward_momentum import WalkForwardConfig, walk_forward_validate
from research.data_stooq import list_stooq_parquets, _infer_symbol_from_filename

STOOQ_DIR = Path("data_cache/stooq")


# ---------------------------------------------------------------------------
# Data loading (shared pattern)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-window diagnostic helpers
# ---------------------------------------------------------------------------

def _window_strategy_metrics(eq: pd.Series) -> dict[str, float]:
    """Compute CAGR, Sharpe, max drawdown from a price/equity series."""
    if len(eq) < 2:
        return {"cagr": float("nan"), "sharpe": float("nan"), "max_dd": float("nan"),
                "total_return": float("nan")}
    rets = eq.pct_change().dropna()
    years = max(len(eq) / 252.0, 1e-9)
    total_return = float(eq.iloc[-1] / eq.iloc[0]) - 1.0
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years)) - 1.0
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252.0)) if rets.std() > 0 else 0.0
    max_dd = float((eq / eq.cummax() - 1.0).min())
    return {"cagr": cagr, "sharpe": sharpe, "max_dd": max_dd, "total_return": total_return}


def _window_market_metrics(market_df: pd.DataFrame, ws: pd.Timestamp, we: pd.Timestamp) -> dict[str, float]:
    """Market return, vol, and drawdown over the window."""
    mkt = market_df.loc[(market_df.index >= ws) & (market_df.index <= we), "close"]
    if len(mkt) < 2:
        return {"mkt_return": float("nan"), "mkt_vol": float("nan"), "mkt_drawdown": float("nan")}
    rets = mkt.pct_change().dropna()
    total_return = float(mkt.iloc[-1] / mkt.iloc[0]) - 1.0
    vol = float(rets.std() * np.sqrt(252.0))
    max_dd = float((mkt / mkt.cummax() - 1.0).min())
    return {"mkt_return": total_return, "mkt_vol": vol, "mkt_drawdown": max_dd}


def _participation_ratio(eq: pd.Series, threshold: float = 1e-4) -> float:
    """
    Fraction of days where the strategy had non-trivial exposure.
    Proxy: daily returns with abs value above threshold indicate holdings were present.
    """
    if len(eq) < 2:
        return float("nan")
    rets = eq.pct_change().dropna().abs()
    return float((rets > threshold).mean())


def _worst5_sum(eq: pd.Series) -> float:
    """Sum of the 5 worst daily returns in the window."""
    if len(eq) < 2:
        return float("nan")
    rets = eq.pct_change().dropna()
    worst5 = rets.nsmallest(5)
    return float(worst5.sum())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Best-config settings from prior sweeps
    cfg = WalkForwardConfig(
        positions=12,
        rebalance_interval_weeks=2,
        cost_bps=5.0,
        slippage_bps=2.0,
        min_rebalance_weight_change=0.0,
        w_3m=0.60,
        w_6m=0.30,
        w_12m=0.10,
        market_sma_days=200,
        risk_on_buffer=0.0,
        min_exposure=0.25,
        max_exposure=1.0,
        exposure_slope=0.0,
        stability_lookback_periods=1,
    )

    print("[INFO] Loading data")
    symbol_dfs, market_df = load_data()

    # Normalize market_df index for slicing
    market_df = market_df.sort_index()
    if not isinstance(market_df.index, pd.DatetimeIndex):
        market_df.index = pd.to_datetime(market_df.index)

    print("[INFO] Running walk-forward")
    results_df, equity_oos = walk_forward_validate(
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=cfg,
    )

    print("[INFO] Computing per-window diagnostics...")
    rows = []

    for _, wrow in results_df.iterrows():
        ws = pd.Timestamp(wrow["window_start"])
        we = pd.Timestamp(wrow["window_end"])

        # Slice stitched equity curve to this window
        eq_window = equity_oos.loc[(equity_oos.index >= ws) & (equity_oos.index <= we)]

        strat = _window_strategy_metrics(eq_window)
        mkt = _window_market_metrics(market_df, ws, we)

        rel_perf = (
            strat["total_return"] - mkt["mkt_return"]
            if np.isfinite(strat["total_return"]) and np.isfinite(mkt["mkt_return"])
            else float("nan")
        )
        participation = _participation_ratio(eq_window)
        worst5 = _worst5_sum(eq_window)
        regime = "bear" if mkt["mkt_return"] < 0 else "bull"

        rows.append({
            "window_start":     ws.strftime("%Y-%m-%d"),
            "window_end":       we.strftime("%Y-%m-%d"),
            "cagr_pct":         round(strat["cagr"] * 100, 2),
            "sharpe":           round(strat["sharpe"], 3),
            "max_dd_pct":       round(strat["max_dd"] * 100, 2),
            "mkt_return_pct":   round(mkt["mkt_return"] * 100, 2),
            "mkt_vol_ann_pct":  round(mkt["mkt_vol"] * 100, 2),
            "mkt_drawdown_pct": round(mkt["mkt_drawdown"] * 100, 2),
            "rel_perf_pct":     round(rel_perf * 100, 2),
            "participation_pct":round(participation * 100, 1),
            "worst5_sum_pct":   round(worst5 * 100, 2),
            "regime":           regime,
        })

    diag_df = (
        pd.DataFrame(rows)
        .sort_values("cagr_pct", ascending=True)
        .reset_index(drop=True)
    )

    print("\n=== WINDOW DIAGNOSTICS ===")
    print(diag_df.to_string(index=False))

    # Summary: average stats split by regime
    print("\n--- Regime summary (averages) ---")
    summary = (
        diag_df.groupby("regime")[["cagr_pct", "sharpe", "rel_perf_pct", "participation_pct", "worst5_sum_pct"]]
        .mean()
        .round(2)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
