#!/usr/bin/env python3
"""
Compare baseline momentum strategy vs a momentum-effectiveness skip filter.

Filtered rule (momentum_effectiveness_skip):
- At each scheduled rebalance date, compute recent momentum effectiveness over a
  63-bar window:
    effectiveness = avg(return(top-12 by momentum at t-63 to t))
                    - avg(return(universe at t-63 to t))
- If effectiveness < 0, skip the rebalance entirely.
- Skip means: keep holdings/cash unchanged; no turnover/cost on that date.

Run:
    python3 -m research.run_momentum_effectiveness_filter_test
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

OUT_FULL = Path("research/momentum_effectiveness_results.csv")
OUT_WORST = Path("research/momentum_effectiveness_worst_window.csv")
OUT_DIAG = Path("research/momentum_effectiveness_diagnostics.csv")
OUT_VALUES = Path("research/momentum_effectiveness_values.csv")


def _build_cfg(market_filter_mode: str) -> WalkForwardConfig:
    return WalkForwardConfig(
        train_years=3,
        test_months=6,
        step_months=6,
        positions=12,
        universe_top_n=800,
        rebalance_weekday=0,
        rebalance_interval_weeks=3,
        starting_cash=100_000.0,
        liq_lookback=60,
        mom_3m=63,
        mom_6m=126,
        mom_12m=252,
        w_3m=0.6,
        w_6m=0.3,
        w_12m=0.1,
        use_strength_filter=False,
        percentile_filter_enabled=False,
        market_filter_mode=market_filter_mode,
        momentum_effectiveness_lookback=63,
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
        "volatility": vol,
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
            was_skipped = bool(rec.get("skipped", False))
            if bool(rec.get("choppy_override", False)):
                raise RuntimeError(
                    f"FAIL: variant={name} emitted choppy exposure overrides; this should be skip-only logic."
                )

            symbols = [s for s in str(rec.get("selected_symbols", "")).split("|") if s]
            rebalance_rows.append(
                {
                    "rebalance_date": pd.Timestamp(rec["rebalance_date"]),
                    "skipped": was_skipped,
                    "skip_reason": str(rec.get("skip_reason", "")),
                    "momentum_effectiveness": rec.get("momentum_effectiveness"),
                    "turnover": float(rec.get("turnover", 0.0)),
                    "estimated_cost": float(rec.get("estimated_cost", 0.0)),
                    "selected_count": len(symbols) if not was_skipped else None,
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

    full_metrics = _compute_metrics(equity_oos)
    ww = equity_oos.loc[pd.Timestamp(WINDOW_START):pd.Timestamp(WINDOW_END)]
    worst_metrics = _compute_metrics(ww)

    skipped_df = rebal_df[rebal_df["skipped"]].copy()
    executed_df = rebal_df[~rebal_df["skipped"]].copy()

    if not skipped_df.empty and (skipped_df["turnover"] > 1e-9).any():
        raise RuntimeError("FAIL: skipped rebalances contain non-zero turnover.")

    avg_turnover = float(rebal_df["turnover"].mean()) if not rebal_df.empty else float("nan")

    ww_start = pd.Timestamp(WINDOW_START)
    ww_end = pd.Timestamp(WINDOW_END)
    skipped_in_ww = int(
        ((skipped_df["rebalance_date"] >= ww_start) & (skipped_df["rebalance_date"] <= ww_end)).sum()
    ) if not skipped_df.empty else 0

    return {
        "name": name,
        "equity_oos": equity_oos,
        "rebal_df": rebal_df,
        "skipped_df": skipped_df,
        "executed_df": executed_df,
        "full_metrics": full_metrics,
        "worst_metrics": worst_metrics,
        "avg_turnover": avg_turnover,
        "total_rebalances": int(len(rebal_df)),
        "skipped_rebalances": int(len(skipped_df)),
        "executed_rebalances": int(len(executed_df)),
        "skipped_in_worst_window": skipped_in_ww,
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
                "sharpe": baseline["full_metrics"]["sharpe"],
                "max_drawdown": baseline["full_metrics"]["max_drawdown"],
                "avg_turnover": baseline["avg_turnover"],
            },
            {
                "variant": "momentum_effectiveness_skip",
                "final_multiple": filtered["full_metrics"]["final_multiple"],
                "cagr": filtered["full_metrics"]["cagr"],
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
                "variant": "momentum_effectiveness_skip",
                "worst_window_return": filtered["worst_metrics"]["final_multiple"] - 1.0,
                "worst_window_sharpe": filtered["worst_metrics"]["sharpe"],
                "worst_window_max_drawdown": filtered["worst_metrics"]["max_drawdown"],
            },
        ]
    )

    diag_df = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "total_rebalances": baseline["total_rebalances"],
                "skipped_rebalances": baseline["skipped_rebalances"],
                "executed_rebalances": baseline["executed_rebalances"],
                "pct_skipped": 0.0,
                "skipped_in_worst_window": baseline["skipped_in_worst_window"],
            },
            {
                "variant": "momentum_effectiveness_skip",
                "total_rebalances": filtered["total_rebalances"],
                "skipped_rebalances": filtered["skipped_rebalances"],
                "executed_rebalances": filtered["executed_rebalances"],
                "pct_skipped": (
                    filtered["skipped_rebalances"] / filtered["total_rebalances"]
                    if filtered["total_rebalances"] > 0 else 0.0
                ),
                "skipped_in_worst_window": filtered["skipped_in_worst_window"],
            },
        ]
    )

    values_df = filtered["rebal_df"][
        ["rebalance_date", "momentum_effectiveness", "skipped"]
    ].copy()
    values_df["decision"] = np.where(values_df["skipped"], "skip", "execute")
    values_df = values_df[["rebalance_date", "momentum_effectiveness", "decision"]]

    return full_df, worst_df, diag_df, values_df


def _print_interpretation(full_df: pd.DataFrame, worst_df: pd.DataFrame, diag_df: pd.DataFrame) -> None:
    base_full = full_df.loc[full_df["variant"] == "baseline"].iloc[0]
    filt_full = full_df.loc[full_df["variant"] == "momentum_effectiveness_skip"].iloc[0]
    base_worst = worst_df.loc[worst_df["variant"] == "baseline"].iloc[0]
    filt_worst = worst_df.loc[worst_df["variant"] == "momentum_effectiveness_skip"].iloc[0]
    filt_diag = diag_df.loc[diag_df["variant"] == "momentum_effectiveness_skip"].iloc[0]

    cagr_change = float(filt_full["cagr"] - base_full["cagr"])
    sharpe_change = float(filt_full["sharpe"] - base_full["sharpe"])
    ww_change = float(filt_worst["worst_window_return"] - base_worst["worst_window_return"])

    cagr_flag = "significant" if cagr_change < -0.02 else "not significant"

    print("\n=== INTERPRETATION ===")
    print(
        f"Did worst window improve? {'yes' if ww_change > 0 else 'no'} "
        f"(return delta={ww_change:+.2%})."
    )
    print(
        f"Did Sharpe improve? {'yes' if sharpe_change > 0 else 'no'} "
        f"(delta={sharpe_change:+.4f})."
    )
    print(
        f"Did CAGR drop significantly? {'yes' if cagr_change < -0.02 else 'no'} "
        f"(delta={cagr_change:+.2%}; {cagr_flag} by -2.00pp rule)."
    )
    print(
        f"How often was momentum not working? {int(filt_diag['skipped_rebalances'])}/"
        f"{int(filt_diag['total_rebalances'])} rebalances "
        f"({float(filt_diag['pct_skipped']):.1%} skipped)."
    )
    print(
        f"Did filter trigger during bad window? {'yes' if int(filt_diag['skipped_in_worst_window']) > 0 else 'no'} "
        f"({int(filt_diag['skipped_in_worst_window'])} skipped in {WINDOW_START} -> {WINDOW_END})."
    )


def main() -> None:
    symbol_dfs, market_df = _load_data()

    print("[INFO] Running baseline variant...")
    baseline = _run_variant(
        name="baseline",
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=_build_cfg(market_filter_mode="none"),
    )

    print("[INFO] Running momentum_effectiveness_skip variant...")
    filtered = _run_variant(
        name="momentum_effectiveness_skip",
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=_build_cfg(market_filter_mode="momentum_effectiveness_skip"),
    )

    if filtered["rebal_df"]["momentum_effectiveness"].dropna().empty:
        raise RuntimeError("FAIL: effectiveness calculation is empty.")

    if filtered["skipped_rebalances"] == 0:
        raise RuntimeError("FAIL: no rebalances were skipped.")

    if not np.isfinite(filtered["avg_turnover"]) or not np.isfinite(baseline["avg_turnover"]):
        raise RuntimeError("FAIL: turnover statistics are invalid.")

    if filtered["avg_turnover"] >= baseline["avg_turnover"]:
        raise RuntimeError(
            "FAIL: skip logic did not reduce turnover versus baseline."
        )

    full_df, worst_df, diag_df, values_df = _build_output_tables(baseline, filtered)

    print("\n=== FULL OOS COMPARISON ===")
    print(full_df.to_string(index=False))

    print("\n=== WORST WINDOW COMPARISON ===")
    print(worst_df.to_string(index=False))

    print("\n=== EFFECTIVENESS DIAGNOSTICS ===")
    print(diag_df.to_string(index=False))

    print("\n=== EFFECTIVENESS VALUES TABLE ===")
    print(values_df.to_string(index=False))

    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(OUT_FULL, index=False)
    worst_df.to_csv(OUT_WORST, index=False)
    diag_df.to_csv(OUT_DIAG, index=False)
    values_df.to_csv(OUT_VALUES, index=False)

    print(f"\n[INFO] Saved {OUT_FULL}")
    print(f"[INFO] Saved {OUT_WORST}")
    print(f"[INFO] Saved {OUT_DIAG}")
    print(f"[INFO] Saved {OUT_VALUES}")

    _print_interpretation(full_df, worst_df, diag_df)


if __name__ == "__main__":
    main()
