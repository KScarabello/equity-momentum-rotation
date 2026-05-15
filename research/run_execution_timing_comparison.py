#!/usr/bin/env python3
"""
Compare walk-forward results under close execution versus next-open execution.

Run:
    python -m research.run_execution_timing_comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.live_trading_config import build_baseline_cfg
from research.data_stooq import load_stooq_ohlcv_bundle
from research.walk_forward_momentum import WalkForwardConfig, walk_forward_validate

STOOQ_DIR = Path("data_cache/stooq")
OUT_FULL = Path("research/execution_timing_results.csv")
OUT_WINDOWS = Path("research/execution_timing_window_results.csv")
OUT_GAPS = Path("research/execution_timing_gap_diagnostics.csv")


def _compute_full_metrics(equity: pd.Series) -> dict[str, float]:
    eq = equity.dropna().sort_index()
    if len(eq) < 2:
        return {
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
        }

    rets = eq.pct_change().dropna()
    ann_factor = 252.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (ann_factor / max(len(eq) - 1, 1)) - 1.0
    sharpe = (
        (rets.mean() * ann_factor) / (rets.std() * np.sqrt(ann_factor))
        if rets.std() > 0
        else 0.0
    )
    max_drawdown = float((eq / eq.cummax() - 1.0).min())
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "total_return": total_return,
    }


def _compute_benchmark_return(market_close: pd.Series, equity_index: pd.Index) -> float:
    bench = market_close.reindex(equity_index).ffill().dropna()
    if len(bench) < 2:
        return 0.0
    return float(bench.iloc[-1] / bench.iloc[0] - 1.0)


def _run_variant(
    symbol_dfs: dict[str, pd.DataFrame],
    market_df: pd.DataFrame,
    cfg: WalkForwardConfig,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    return walk_forward_validate(
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=cfg,
        return_debug=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare close versus next-open execution timing using walk-forward validation."
    )
    parser.add_argument("--stooq-dir", type=Path, default=STOOQ_DIR)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--limit-symbols", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    base_cfg = build_baseline_cfg()

    print("[INFO] Loading OHLCV bundle with required open prices...")
    symbol_dfs = load_stooq_ohlcv_bundle(
        stooq_dir=args.stooq_dir,
        limit_symbols=args.limit_symbols,
        start=args.start,
        end=args.end,
        require_open=True,
    )

    market_symbol = base_cfg.market_symbol
    if market_symbol not in symbol_dfs:
        if "SPY" in symbol_dfs:
            print(
                f"[WARN] Configured market_symbol={market_symbol!r} not found; falling back to 'SPY'"
            )
            market_symbol = "SPY"
        else:
            raise ValueError(
                f"Configured market_symbol={market_symbol!r} not found in the Stooq OHLCV bundle"
            )
    market_df = symbol_dfs[market_symbol]

    comparison_rows: list[dict[str, Any]] = []
    window_tables: list[pd.DataFrame] = []
    next_open_gap_df = pd.DataFrame()

    for execution_price in ["close", "next_open"]:
        cfg = WalkForwardConfig(**{**base_cfg.__dict__, "execution_price": execution_price})
        print(f"[INFO] Running walk-forward with execution_price={execution_price}")

        results_df, equity_oos, debug = _run_variant(symbol_dfs, market_df, cfg)
        full_metrics = _compute_full_metrics(equity_oos)
        benchmark_return = _compute_benchmark_return(market_df["close"], equity_oos.index)

        comparison_rows.append(
            {
                "execution_price": execution_price,
                "cagr": full_metrics["cagr"],
                "sharpe": full_metrics["sharpe"],
                "max_drawdown": full_metrics["max_drawdown"],
                "turnover": float(results_df["turnover"].sum()) if "turnover" in results_df.columns else 0.0,
                "total_return": full_metrics["total_return"],
                "benchmark_return": benchmark_return,
                "skipped_execution_symbols": int(debug.get("skipped_execution_symbols", 0)),
                "skipped_execution_rebalances": int(debug.get("skipped_execution_rebalances", 0)),
            }
        )

        window_tables.append(results_df.assign(execution_price=execution_price))

        if execution_price == "next_open":
            next_open_gap_df = pd.DataFrame(debug.get("gap_records", []))

    full_df = pd.DataFrame(comparison_rows)
    windows_df = pd.concat(window_tables, ignore_index=True) if window_tables else pd.DataFrame()

    print("\n=== EXECUTION TIMING COMPARISON ===")
    print(full_df.to_string(index=False))

    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(OUT_FULL, index=False)
    windows_df.to_csv(OUT_WINDOWS, index=False)
    next_open_gap_df.to_csv(OUT_GAPS, index=False)

    print("\n[INFO] Wrote outputs")
    print(f"  - {OUT_FULL}")
    print(f"  - {OUT_WINDOWS}")
    print(f"  - {OUT_GAPS}")


if __name__ == "__main__":
    main()