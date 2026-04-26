#!/usr/bin/env python3
"""
Run cost sensitivity analysis for the walk-forward momentum strategy.

This script varies only cost_bps while keeping all other reference settings fixed.
It reuses the existing walk-forward pipeline and evaluates in-memory OOS equity.

Run:
    python3 -m research.run_cost_sensitivity
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import pandas as pd

from research.run_walk_forward import STOOQ_DIR, build_universe_from_stooq, fetch_ohlcv
from research.walk_forward_momentum import WalkForwardConfig, walk_forward_validate


COST_BPS_LIST = [0, 5, 10, 20]
OUT_CSV = Path("research/cost_sensitivity_results.csv")


def _build_baseline_cfg(cost_bps: int) -> WalkForwardConfig:
    # Reference settings, varying only cost_bps.
    return WalkForwardConfig(
        train_years=3,
        test_months=6,
        step_months=6,
        positions=12,
        universe_top_n=800,
        rebalance_weekday=0,
        rebalance_interval_weeks=3,  # 15 trading days approx
        starting_cash=100_000.0,
        liq_lookback=60,
        mom_3m=63,
        mom_6m=126,
        mom_12m=252,
        w_3m=0.6,
        w_6m=0.3,
        w_12m=0.1,
        veto_if_12m_return_below=0.0,
        market_symbol="SPY",
        market_sma_days=200,
        risk_on_buffer=0.0,
        cost_bps=float(cost_bps),
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


def _compute_metrics(equity: pd.Series) -> dict:
    if equity.empty:
        raise ValueError("OOS equity series is empty.")

    returns = equity.pct_change().dropna()
    if returns.empty:
        raise ValueError("OOS equity has no return observations.")

    final_multiple = float(equity.iloc[-1] / equity.iloc[0])
    cagr = float(final_multiple ** (252 / len(returns)) - 1)
    vol = float(returns.std() * sqrt(252))

    std = float(returns.std())
    sharpe = float((returns.mean() / std) * sqrt(252)) if std > 0 else float("nan")

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = float(drawdown.min())

    return {
        "final_multiple": final_multiple,
        "cagr": cagr,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "rows_in_oos": int(len(equity)),
        "oos_start": str(equity.index.min().date()),
        "oos_end": str(equity.index.max().date()),
    }


def _choose_baseline_row(df: pd.DataFrame) -> pd.Series:
    if (df["cost_bps"] == 5).any():
        return df.loc[df["cost_bps"] == 5].iloc[0]
    idx = (df["cost_bps"] - 5).abs().idxmin()
    return df.loc[idx]


def main() -> None:
    symbol_dfs, market_df = _load_data()

    rows: list[dict] = []
    total_runs = len(COST_BPS_LIST)

    for i, cost_bps in enumerate(COST_BPS_LIST, start=1):
        print(f"[{i}/{total_runs}] Running cost_bps={cost_bps}")

        cfg = _build_baseline_cfg(cost_bps)
        _, equity_oos, _ = walk_forward_validate(
            symbol_dfs=symbol_dfs,
            market_df=market_df,
            cfg=cfg,
            return_debug=True,
        )

        if equity_oos.empty:
            raise RuntimeError(f"Empty OOS equity for cost_bps={cost_bps}")

        metrics = _compute_metrics(equity_oos)
        rows.append(
            {
                "cost_bps": int(cost_bps),
                **metrics,
            }
        )

    df = pd.DataFrame(rows)

    print("\n=== COST SENSITIVITY RESULTS ===")
    print(df.to_string(index=False))

    print("\n=== SORTED BY SHARPE DESC ===")
    print(df.sort_values("sharpe", ascending=False).to_string(index=False))

    print("\n=== SORTED BY CAGR DESC ===")
    print(df.sort_values("cagr", ascending=False).to_string(index=False))

    print("\n=== COST SENSITIVITY SUMMARY ===")
    baseline = _choose_baseline_row(df)
    best_sharpe = df.loc[df["sharpe"].idxmax()]
    worst_sharpe = df.loc[df["sharpe"].idxmin()]

    print("Baseline (cost_bps=5 or closest):")
    print(baseline.to_string())
    print()

    print("Best Sharpe:")
    print(best_sharpe.to_string())
    print()

    print("Worst Sharpe:")
    print(worst_sharpe.to_string())
    print()

    row_0 = df.loc[df["cost_bps"] == 0]
    row_20 = df.loc[df["cost_bps"] == 20]
    if row_0.empty or row_20.empty:
        print("CAGR degradation: N/A (missing cost_bps=0 or cost_bps=20)")
        print("Sharpe degradation: N/A (missing cost_bps=0 or cost_bps=20)")
    else:
        cagr_drop = float(row_0.iloc[0]["cagr"] - row_20.iloc[0]["cagr"])
        sharpe_drop = float(row_0.iloc[0]["sharpe"] - row_20.iloc[0]["sharpe"])
        print(f"CAGR degradation: {cagr_drop * 100:.2f}%")
        print(f"Sharpe degradation: {sharpe_drop:.4f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[INFO] Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
