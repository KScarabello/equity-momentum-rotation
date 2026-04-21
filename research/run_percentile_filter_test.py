#!/usr/bin/env python3
"""
Compare baseline momentum strategy vs a percentile-based momentum eligibility filter.

Filtered rule:
- At each rebalance, after momentum scoring, only keep names at/above the
  configured cross-sectional percentile threshold (default top 20%).
- If fewer than N names are eligible, hold fewer names and keep residual cash.

Run:
    python3 -m research.run_percentile_filter_test
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.run_walk_forward import STOOQ_DIR, build_universe_from_stooq, fetch_ohlcv
from research.walk_forward_momentum import (
    WalkForwardConfig,
    _ensure_datetime_index,
    _month_delta,
    _normalize_cols,
    _year_delta,
    run_weekly_portfolio,
)


WINDOW_START = "2024-12-09"
WINDOW_END = "2025-03-12"
PERCENTILE_THRESHOLD = 0.80

OUT_FULL = Path("research/percentile_filter_results.csv")
OUT_WORST = Path("research/percentile_filter_worst_window.csv")
OUT_EXPOSURE = Path("research/percentile_filter_exposure_summary.csv")


def _build_cfg(percentile_filter_enabled: bool) -> WalkForwardConfig:
    return WalkForwardConfig(
        train_years=3,
        test_months=6,
        step_months=6,
        positions=12,
        universe_top_n=800,
        rebalance_weekday=0,
        rebalance_interval_weeks=3,  # rebalance_days ~ 15
        starting_cash=100_000.0,
        liq_lookback=60,
        mom_3m=63,
        mom_6m=126,
        mom_12m=252,
        w_3m=0.6,
        w_6m=0.3,
        w_12m=0.1,
        use_strength_filter=False,
        percentile_filter_enabled=percentile_filter_enabled,
        percentile_threshold=PERCENTILE_THRESHOLD,
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

    print("[INFO] Loading SPY OHLCV")
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


def _compute_metrics(equity: pd.Series) -> dict[str, float]:
    equity = equity.dropna()
    if len(equity) < 2:
        raise ValueError("Equity series has fewer than 2 rows; cannot compute metrics.")

    returns = equity.pct_change().dropna()
    if returns.empty:
        raise ValueError("Equity returns are empty; cannot compute metrics.")

    final_multiple = float(equity.iloc[-1] / equity.iloc[0])
    cagr = float(final_multiple ** (252 / len(returns)) - 1.0)
    volatility = float(returns.std() * sqrt(252))

    std = float(returns.std())
    sharpe = float((returns.mean() / std) * sqrt(252)) if std > 0 else float("nan")

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = float(drawdown.min())

    return {
        "final_multiple": final_multiple,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def _build_windows(
    symbol_dfs: dict[str, pd.DataFrame], market_df: pd.DataFrame, cfg: WalkForwardConfig
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[tuple[pd.Timestamp, pd.Timestamp]]]:
    market_df = _ensure_datetime_index(_normalize_cols(market_df))
    symbol_dfs = {s: _ensure_datetime_index(_normalize_cols(df)) for s, df in symbol_dfs.items()}

    min_need = max(cfg.liq_lookback, cfg.mom_12m + 1, cfg.market_sma_days + 1)
    start_candidates = [
        df.index[min_need] for df in symbol_dfs.values() if len(df.index) > min_need
    ]
    if not start_candidates:
        raise ValueError("Not enough history across symbols for configured lookbacks.")

    global_start = max(max(start_candidates), market_df.index[min_need])
    global_end = market_df.index.max()

    first_test_start = _year_delta(global_start, cfg.train_years)
    if first_test_start > global_end:
        raise ValueError("Not enough history for first test window.")

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = first_test_start
    while True:
        test_start = cursor
        test_end = _month_delta(test_start, cfg.test_months) - pd.Timedelta(days=1)
        if test_start >= global_end:
            break
        test_end = min(test_end, global_end)
        windows.append((test_start, test_end))
        cursor = _month_delta(cursor, cfg.step_months)
        if cursor > global_end:
            break

    return market_df, symbol_dfs, windows


def _run_variant(
    name: str,
    symbol_dfs: dict[str, pd.DataFrame],
    market_df: pd.DataFrame,
    cfg: WalkForwardConfig,
) -> dict[str, Any]:
    market_df, symbol_dfs, windows = _build_windows(symbol_dfs, market_df, cfg)

    cash = cfg.starting_cash
    holdings: dict[str, float] = {}

    oos_values: list[float] = []
    oos_dates: list[pd.Timestamp] = []
    rebalance_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []

    for ws, we in windows:
        cfg_window = WalkForwardConfig(**{**cfg.__dict__, "starting_cash": cash})
        snapshot_dates = set(market_df.index[(market_df.index >= ws) & (market_df.index <= we)])

        res = run_weekly_portfolio(
            symbol_dfs=symbol_dfs,
            market_df=market_df,
            start=ws,
            end=we,
            cfg=cfg_window,
            initial_cash=cash,
            initial_holdings=holdings,
            snapshot_dates=snapshot_dates,
        )

        eq = res.equity_curve
        if oos_dates:
            eq = eq[eq.index > oos_dates[-1]]

        oos_values.extend(eq.values.tolist())
        oos_dates.extend(eq.index.tolist())

        for rec in res.rebalance_records:
            symbols = [s for s in str(rec.get("selected_symbols", "")).split("|") if s]
            rebalance_rows.append(
                {
                    "rebalance_date": pd.Timestamp(rec["rebalance_date"]),
                    "eligible_count": int(rec.get("eligible_count", 0)),
                    "selected_count": len(symbols),
                    "turnover": float(rec.get("turnover", 0.0)),
                    "target_exposure": float(rec.get("target_exposure", 0.0)),
                    "selected_symbols": "|".join(symbols),
                }
            )

        for snap in res.state_snapshots:
            eq_s = float(snap.get("equity", np.nan))
            invested_s = float(snap.get("invested_value", np.nan))
            cash_frac = float("nan")
            if np.isfinite(eq_s) and eq_s > 0:
                cash_frac = max(0.0, min(1.0, 1.0 - invested_s / eq_s))

            snapshot_rows.append(
                {
                    "date": pd.Timestamp(snap["date"]),
                    "equity": eq_s,
                    "invested_value": invested_s,
                    "cash": float(snap.get("cash", np.nan)),
                    "cash_fraction": cash_frac,
                    "holdings_count": int(snap.get("holdings_count", 0)),
                }
            )

        cash = float(res.ending_cash)
        holdings = {s: float(sh) for s, sh in res.ending_holdings.items()}

    if not oos_dates:
        raise RuntimeError(f"No OOS output for variant={name}")

    equity_oos = pd.Series(oos_values, index=pd.DatetimeIndex(oos_dates), name=f"equity_{name}")
    if equity_oos.empty:
        raise RuntimeError(f"Empty OOS equity series for variant={name}")

    rebal_df = pd.DataFrame(rebalance_rows).sort_values("rebalance_date").reset_index(drop=True)
    snaps_df = (
        pd.DataFrame(snapshot_rows)
        .sort_values("date")
        .drop_duplicates(subset=["date"])
        .reset_index(drop=True)
    )

    full_metrics = _compute_metrics(equity_oos)
    if not np.isfinite(full_metrics["final_multiple"]):
        raise RuntimeError(f"Invalid full-period metrics for variant={name}")

    ww = equity_oos.loc[pd.Timestamp(WINDOW_START):pd.Timestamp(WINDOW_END)]
    worst_metrics = _compute_metrics(ww)

    avg_positions = float(rebal_df["selected_count"].mean()) if not rebal_df.empty else float("nan")
    avg_turnover = float(rebal_df["turnover"].mean()) if not rebal_df.empty else float("nan")

    pct_time_some_cash = float((snaps_df["cash_fraction"] > 0.0).mean()) if not snaps_df.empty else float("nan")
    avg_cash_fraction = float(snaps_df["cash_fraction"].mean()) if not snaps_df.empty else float("nan")

    binding_stats = {
        "rebalance_rows": int(len(rebal_df)),
        "rebalance_with_fewer_than_12_eligible": int((rebal_df["eligible_count"] < 12).sum()) if not rebal_df.empty else 0,
        "avg_eligible_count": float(rebal_df["eligible_count"].mean()) if not rebal_df.empty else float("nan"),
        "min_eligible_count": int(rebal_df["eligible_count"].min()) if not rebal_df.empty else 0,
    }

    return {
        "name": name,
        "equity_oos": equity_oos,
        "rebalances": rebal_df,
        "snapshots": snaps_df,
        "full_metrics": full_metrics,
        "worst_metrics": worst_metrics,
        "avg_positions": avg_positions,
        "avg_turnover": avg_turnover,
        "pct_time_some_cash": pct_time_some_cash,
        "avg_cash_fraction": avg_cash_fraction,
        "binding_stats": binding_stats,
    }


def _build_output_tables(
    baseline: dict[str, Any], filtered: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_df = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "final_multiple": baseline["full_metrics"]["final_multiple"],
                "cagr": baseline["full_metrics"]["cagr"],
                "volatility": baseline["full_metrics"]["volatility"],
                "sharpe": baseline["full_metrics"]["sharpe"],
                "max_drawdown": baseline["full_metrics"]["max_drawdown"],
                "avg_turnover": baseline["avg_turnover"],
            },
            {
                "variant": "percentile_filtered_top20",
                "final_multiple": filtered["full_metrics"]["final_multiple"],
                "cagr": filtered["full_metrics"]["cagr"],
                "volatility": filtered["full_metrics"]["volatility"],
                "sharpe": filtered["full_metrics"]["sharpe"],
                "max_drawdown": filtered["full_metrics"]["max_drawdown"],
                "avg_turnover": filtered["avg_turnover"],
            },
        ]
    )

    worst_df = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "worst_window_return": baseline["worst_metrics"]["final_multiple"] - 1.0,
                "worst_window_sharpe": baseline["worst_metrics"]["sharpe"],
                "worst_window_max_drawdown": baseline["worst_metrics"]["max_drawdown"],
            },
            {
                "variant": "percentile_filtered_top20",
                "worst_window_return": filtered["worst_metrics"]["final_multiple"] - 1.0,
                "worst_window_sharpe": filtered["worst_metrics"]["sharpe"],
                "worst_window_max_drawdown": filtered["worst_metrics"]["max_drawdown"],
            },
        ]
    )

    exposure_df = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "avg_positions": baseline["avg_positions"],
                "pct_time_some_cash": baseline["pct_time_some_cash"],
                "avg_cash_fraction": baseline["avg_cash_fraction"],
            },
            {
                "variant": "percentile_filtered_top20",
                "avg_positions": filtered["avg_positions"],
                "pct_time_some_cash": filtered["pct_time_some_cash"],
                "avg_cash_fraction": filtered["avg_cash_fraction"],
            },
        ]
    )

    binding_df = pd.DataFrame(
        [
            {
                "variant": "percentile_filtered_top20",
                "rebalance_rows": filtered["binding_stats"]["rebalance_rows"],
                "rebalance_with_fewer_than_12_eligible": filtered["binding_stats"]["rebalance_with_fewer_than_12_eligible"],
                "avg_eligible_count": filtered["binding_stats"]["avg_eligible_count"],
                "min_eligible_count": filtered["binding_stats"]["min_eligible_count"],
            }
        ]
    )

    return full_df, worst_df, exposure_df, binding_df


def _print_interpretation(full_df: pd.DataFrame, worst_df: pd.DataFrame, binding_df: pd.DataFrame) -> None:
    base_full = full_df.loc[full_df["variant"] == "baseline"].iloc[0]
    filt_full = full_df.loc[full_df["variant"] == "percentile_filtered_top20"].iloc[0]

    base_worst = worst_df.loc[worst_df["variant"] == "baseline"].iloc[0]
    filt_worst = worst_df.loc[worst_df["variant"] == "percentile_filtered_top20"].iloc[0]

    cagr_change = float(filt_full["cagr"] - base_full["cagr"])
    sharpe_change = float(filt_full["sharpe"] - base_full["sharpe"])
    ww_ret_change = float(filt_worst["worst_window_return"] - base_worst["worst_window_return"])

    dd_improved = float(filt_worst["worst_window_max_drawdown"]) > float(base_worst["worst_window_max_drawdown"])
    ww_ret_improved = float(filt_worst["worst_window_return"]) > float(base_worst["worst_window_return"])
    sharpe_improved = float(filt_full["sharpe"]) > float(base_full["sharpe"])

    binding_row = binding_df.iloc[0]
    bind_count = int(binding_row["rebalance_with_fewer_than_12_eligible"])
    bind_total = int(binding_row["rebalance_rows"])
    binding_text = "often" if bind_total > 0 and (bind_count / bind_total) >= 0.25 else "rarely"

    cagr_flag = "significant" if cagr_change < -0.02 else "not significant"

    print("\n=== INTERPRETATION ===")
    print(
        f"1) Worst-window drawdown improved: {'yes' if dd_improved else 'no'} "
        f"(baseline={base_worst['worst_window_max_drawdown']:.2%}, filtered={filt_worst['worst_window_max_drawdown']:.2%})."
    )
    print(
        f"2) Worst-window return improved: {'yes' if ww_ret_improved else 'no'} "
        f"(delta={ww_ret_change:+.2%})."
    )
    print(
        f"3) Full-period Sharpe improved: {'yes' if sharpe_improved else 'no'} "
        f"(delta={sharpe_change:+.4f})."
    )
    print(
        f"4) CAGR change vs baseline: {cagr_change:+.2%} ({cagr_flag} by the -2.00pp rule)."
    )
    print(
        f"5) Percentile filter binding frequency: {binding_text} "
        f"({bind_count}/{bind_total} rebalances had eligible_count < 12)."
    )


def main() -> None:
    symbol_dfs, market_df = _load_data()

    print("[INFO] Running baseline variant...")
    baseline = _run_variant(
        name="baseline",
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=_build_cfg(percentile_filter_enabled=False),
    )

    print("[INFO] Running percentile_filtered_top20 variant...")
    filtered = _run_variant(
        name="percentile_filtered_top20",
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=_build_cfg(percentile_filter_enabled=True),
    )

    full_df, worst_df, exposure_df, binding_df = _build_output_tables(baseline, filtered)

    print("\n=== BASELINE VS PERCENTILE FILTERED (FULL OOS) ===")
    print(full_df.to_string(index=False))

    print("\n=== WORST WINDOW COMPARISON ===")
    print(worst_df.to_string(index=False))

    print("\n=== EXPOSURE SUMMARY ===")
    print(exposure_df.to_string(index=False))

    print("\n=== FILTER BINDING DIAGNOSTICS ===")
    print(binding_df.to_string(index=False))

    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(OUT_FULL, index=False)
    worst_df.to_csv(OUT_WORST, index=False)
    exposure_df.to_csv(OUT_EXPOSURE, index=False)

    print(f"\n[INFO] Saved {OUT_FULL}")
    print(f"[INFO] Saved {OUT_WORST}")
    print(f"[INFO] Saved {OUT_EXPOSURE}")

    _print_interpretation(full_df, worst_df, binding_df)


if __name__ == "__main__":
    main()
