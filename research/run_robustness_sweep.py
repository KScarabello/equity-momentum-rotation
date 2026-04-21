#!/usr/bin/env python3
"""
Run a small robustness sweep for the walk-forward momentum strategy.

This is a robustness study (12 runs), not a parameter optimization workflow.
It reuses the existing walk-forward engine and evaluates OOS equity only.

Run:
    python -m research.run_robustness_sweep
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd

from research.run_walk_forward import build_universe_from_stooq, fetch_ohlcv, STOOQ_DIR
from research.walk_forward_momentum import WalkForwardConfig, walk_forward_validate


# Robustness grid (fixed by request)
POSITIONS_LIST = [10, 12, 14]
REBALANCE_DAYS_LIST = [10, 15]  # approx: 10d~2w, 15d~3w
MOMENTUM_WEIGHTS_LIST = [
    (0.6, 0.3, 0.1),
    (0.5, 0.3, 0.2),
]

# Baseline marker for summary/deltas
BASELINE = {
    "positions": 12,
    "rebalance_days": 15,
    "weights": (0.6, 0.3, 0.1),
}

OUT_CSV = Path("research/robustness_results.csv")


def _weights_to_str(w: tuple[float, float, float]) -> str:
    return f"{w[0]:.1f}/{w[1]:.1f}/{w[2]:.1f}"


def _rebalance_days_to_interval_weeks(rebalance_days: int) -> int:
    # Use existing weekly-anchor engine, approximating trading-day cadence.
    mapping = {10: 2, 15: 3}
    if rebalance_days not in mapping:
        raise ValueError(f"Unsupported rebalance_days={rebalance_days}. Expected one of {list(mapping)}")
    return mapping[rebalance_days]


def _compute_metrics(equity: pd.Series) -> dict:
    if equity.empty:
        raise ValueError("OOS equity series is empty.")

    returns = equity.pct_change().dropna()
    if returns.empty:
        raise ValueError("OOS equity series has no return observations.")

    final_multiple = float(equity.iloc[-1] / equity.iloc[0])
    cagr = float(final_multiple ** (252 / len(returns)) - 1)
    vol = float(returns.std() * sqrt(252))

    std = float(returns.std())
    sharpe = float((returns.mean() / std) * sqrt(252)) if std > 0 else float("nan")

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = float(drawdown.min())

    return {
        "final_multiple": final_multiple,
        "cagr": cagr,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "rows_in_oos": int(len(equity)),
        "oos_start": str(equity.index.min().date()),
        "oos_end": str(equity.index.max().date()),
    }


def _build_cfg(positions: int, rebalance_days: int, weights: tuple[float, float, float]) -> WalkForwardConfig:
    interval_weeks = _rebalance_days_to_interval_weeks(rebalance_days)
    w3, w6, w12 = weights

    return WalkForwardConfig(
        train_years=3,
        test_months=6,
        step_months=6,
        positions=positions,
        universe_top_n=800,
        rebalance_weekday=0,
        rebalance_interval_weeks=interval_weeks,
        starting_cash=100_000.0,
        liq_lookback=60,
        mom_3m=63,
        mom_6m=126,
        mom_12m=252,
        w_3m=w3,
        w_6m=w6,
        w_12m=w12,
        veto_if_12m_return_below=0.0,
        market_symbol="SPY",
        market_sma_days=200,
        risk_on_buffer=0.0,
        cost_bps=5.0,
        slippage_bps=2.0,
        min_exposure=0.25,
        max_exposure=1.0,
        exposure_slope=0.0,
        require_positive_sma_slope=True,
        sma_slope_lookback=20,
        stability_lookback_periods=1,
        min_rebalance_weight_change=0.0,
    )


def _load_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    print("[INFO] Loading universe from parquet cache")
    symbols = build_universe_from_stooq(STOOQ_DIR)
    print(f"[INFO] Universe loaded from cache: {len(symbols)}")

    print("[INFO] Loading market proxy OHLCV")
    market_df = fetch_ohlcv("SPY")

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


def _print_row(label: str, row: pd.Series) -> None:
    print(label)
    print(row.to_string())
    print()


def main() -> None:
    symbol_dfs, market_df = _load_data()

    grid = list(product(POSITIONS_LIST, REBALANCE_DAYS_LIST, MOMENTUM_WEIGHTS_LIST))
    total_runs = len(grid)
    rows: list[dict] = []

    for i, (positions, rebalance_days, weights) in enumerate(grid, start=1):
        wstr = _weights_to_str(weights)
        print(f"[{i}/{total_runs}] Running positions={positions} rebalance_days={rebalance_days} weights={wstr}")

        cfg = _build_cfg(positions, rebalance_days, weights)

        results_df, equity_oos, debug = walk_forward_validate(
            symbol_dfs=symbol_dfs,
            market_df=market_df,
            cfg=cfg,
            return_debug=True,
        )

        if equity_oos.empty:
            raise RuntimeError(
                f"Empty OOS equity for positions={positions}, rebalance_days={rebalance_days}, weights={wstr}"
            )

        metrics = _compute_metrics(equity_oos)

        rows.append(
            {
                "positions": positions,
                "rebalance_days": rebalance_days,
                "momentum_weights": wstr,
                "trade_count": int(debug.get("total_trades", 0)),
                "non_zero_exposure_days": int(debug.get("non_zero_exposure_days", 0)),
                **metrics,
            }
        )

    df = pd.DataFrame(rows)

    baseline_mask = (
        (df["positions"] == BASELINE["positions"])
        & (df["rebalance_days"] == BASELINE["rebalance_days"])
        & (df["momentum_weights"] == _weights_to_str(BASELINE["weights"]))
    )
    if baseline_mask.sum() != 1:
        raise RuntimeError("Baseline row not found uniquely in robustness results.")

    baseline = df.loc[baseline_mask].iloc[0]

    df["cagr_delta_vs_baseline"] = df["cagr"] - float(baseline["cagr"])
    df["sharpe_delta_vs_baseline"] = df["sharpe"] - float(baseline["sharpe"])
    # Positive delta means less severe drawdown than baseline (improvement).
    df["max_dd_delta_vs_baseline"] = df["max_drawdown"] - float(baseline["max_drawdown"])

    print("\n=== ROBUSTNESS RESULTS (FULL TABLE) ===")
    print(df.to_string(index=False))

    print("\n=== ROBUSTNESS RESULTS (SORTED BY SHARPE DESC) ===")
    print(df.sort_values("sharpe", ascending=False).to_string(index=False))

    print("\n=== ROBUSTNESS RESULTS (SORTED BY CAGR DESC) ===")
    print(df.sort_values("cagr", ascending=False).to_string(index=False))

    print("\n=== ROBUSTNESS SUMMARY ===")
    _print_row("Baseline:", baseline)
    _print_row("Best Sharpe:", df.loc[df["sharpe"].idxmax()])
    _print_row("Best CAGR:", df.loc[df["cagr"].idxmax()])
    _print_row("Worst Sharpe:", df.loc[df["sharpe"].idxmin()])
    _print_row("Worst CAGR:", df.loc[df["cagr"].idxmin()])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[INFO] Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
