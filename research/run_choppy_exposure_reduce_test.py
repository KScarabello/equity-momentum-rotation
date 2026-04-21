#!/usr/bin/env python3
"""
Compare baseline momentum strategy vs a choppy-market exposure-reduction overlay.

Filtered rule (choppy_filter_reduce_exposure):
- On each scheduled rebalance date, evaluate three SPY-based conditions.
- If ALL three are true, the market is classified as "choppy" and portfolio exposure
  is reduced to CHOPPY_EXPOSURE (default 70% invested, 30% cash).
- Rebalance always executes — positions are still updated with fresh momentum rankings.
- Only the target portfolio fraction is scaled down; weights per position remain equal.

Run:
    python3 -m research.run_choppy_exposure_reduce_test
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
CHOPPY_EXPOSURE = 0.70         # fraction of portfolio invested during choppy regime
CHOPPY_VOL_LOOKBACK = 126

OUT_FULL     = Path("research/choppy_exposure_reduce_results.csv")
OUT_WORST    = Path("research/choppy_exposure_reduce_worst_window.csv")
OUT_EXPOSURE = Path("research/choppy_exposure_reduce_exposure_summary.csv")
OUT_DIAG     = Path("research/choppy_exposure_reduce_rebalance_diagnostics.csv")


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
        choppy_vol_lookback=CHOPPY_VOL_LOOKBACK,
        choppy_reduce_exposure=CHOPPY_EXPOSURE,
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
    max_drawdown = float(((cumulative - peak) / peak).min())

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
    market_df  = _ensure_datetime_index(_normalize_cols(market_df))
    symbol_dfs = {s: _ensure_datetime_index(_normalize_cols(df)) for s, df in symbol_dfs.items()}

    min_need = max(cfg.liq_lookback, cfg.mom_12m + 1, cfg.market_sma_days + 1)
    start_candidates = [
        df.index[min_need] for df in symbol_dfs.values() if len(df.index) > min_need
    ]
    if not start_candidates:
        raise ValueError("Not enough history across symbols for configured lookbacks.")

    global_start = max(max(start_candidates), market_df.index[min_need])
    global_end   = market_df.index.max()

    first_test_start = _year_delta(global_start, cfg.train_years)
    if first_test_start > global_end:
        raise ValueError("Not enough history for first test window.")

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = first_test_start
    while True:
        test_start = cursor
        test_end   = _month_delta(test_start, cfg.test_months) - pd.Timedelta(days=1)
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

    cash     = cfg.starting_cash
    holdings: dict[str, float] = {}

    oos_values: list[float] = []
    oos_dates:  list[pd.Timestamp] = []
    all_rebal_rows: list[dict[str, Any]] = []
    snapshot_rows:  list[dict[str, Any]] = []

    for ws, we in windows:
        cfg_window     = WalkForwardConfig(**{**cfg.__dict__, "starting_cash": cash})
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
            was_skipped       = bool(rec.get("skipped", False))
            was_choppy_scaled = bool(rec.get("choppy_override", False))
            symbols           = [s for s in str(rec.get("selected_symbols", "")).split("|") if s]
            all_rebal_rows.append(
                {
                    "rebalance_date":   pd.Timestamp(rec["rebalance_date"]),
                    "skipped":          was_skipped,
                    "choppy_override":  was_choppy_scaled,
                    "target_exposure":  float(rec.get("target_exposure", 1.0)),
                    "risk_on":          rec.get("risk_on"),
                    "selected_count":   len(symbols) if not was_skipped else None,
                    "turnover":         float(rec.get("turnover", 0.0)),
                    "estimated_cost":   float(rec.get("estimated_cost", 0.0)),
                    "selected_symbols": "|".join(symbols),
                }
            )

        for snap in res.state_snapshots:
            eq_s       = float(snap.get("equity", np.nan))
            invested_s = float(snap.get("invested_value", np.nan))
            cash_frac  = float("nan")
            if np.isfinite(eq_s) and eq_s > 0:
                cash_frac = max(0.0, min(1.0, 1.0 - invested_s / eq_s))
            snapshot_rows.append(
                {
                    "date":           pd.Timestamp(snap["date"]),
                    "equity":         eq_s,
                    "invested_value": invested_s,
                    "cash":           float(snap.get("cash", np.nan)),
                    "cash_fraction":  cash_frac,
                    "holdings_count": int(snap.get("holdings_count", 0)),
                }
            )

        cash     = float(res.ending_cash)
        holdings = {s: float(sh) for s, sh in res.ending_holdings.items()}

    if not oos_dates:
        raise RuntimeError(f"No OOS output for variant={name}")

    equity_oos = pd.Series(oos_values, index=pd.DatetimeIndex(oos_dates), name=f"equity_{name}")
    rebal_df   = (
        pd.DataFrame(all_rebal_rows)
        .sort_values("rebalance_date")
        .reset_index(drop=True)
    )
    snaps_df   = (
        pd.DataFrame(snapshot_rows)
        .sort_values("date")
        .drop_duplicates(subset=["date"])
        .reset_index(drop=True)
    )

    full_metrics  = _compute_metrics(equity_oos)
    ww = equity_oos.loc[pd.Timestamp(WINDOW_START):pd.Timestamp(WINDOW_END)]
    worst_metrics = _compute_metrics(ww)

    executed_df     = rebal_df[~rebal_df["skipped"]].copy()
    choppy_rebal_df = rebal_df[rebal_df["choppy_override"]].copy()

    # Validation: rebalance count must equal baseline (no skips in this variant)
    assert not rebal_df["skipped"].any(), (
        f"FAIL: {name} has unexpected skipped rebalances — check market_filter_mode."
    )

    ww_ts = (pd.Timestamp(WINDOW_START), pd.Timestamp(WINDOW_END))
    choppy_in_ww = int(
        (
            (choppy_rebal_df["rebalance_date"] >= ww_ts[0])
            & (choppy_rebal_df["rebalance_date"] <= ww_ts[1])
        ).sum()
    ) if not choppy_rebal_df.empty else 0

    avg_positions      = float(executed_df["selected_count"].mean()) if not executed_df.empty else float("nan")
    avg_turnover       = float(executed_df["turnover"].mean())       if not executed_df.empty else float("nan")
    pct_time_some_cash = float((snaps_df["cash_fraction"] > 0.0).mean()) if not snaps_df.empty else float("nan")
    avg_cash_fraction  = float(snaps_df["cash_fraction"].mean())          if not snaps_df.empty else float("nan")

    return {
        "name":              name,
        "equity_oos":        equity_oos,
        "rebal_df":          rebal_df,
        "choppy_rebal_df":   choppy_rebal_df,
        "snapshots":         snaps_df,
        "full_metrics":      full_metrics,
        "worst_metrics":     worst_metrics,
        "avg_positions":     avg_positions,
        "avg_turnover":      avg_turnover,
        "pct_time_some_cash": pct_time_some_cash,
        "avg_cash_fraction": avg_cash_fraction,
        "total_rebalances":  int(len(rebal_df)),
        "choppy_count":      int(len(choppy_rebal_df)),
        "choppy_in_ww":      choppy_in_ww,
    }


def _build_output_tables(
    baseline: dict[str, Any], filtered: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    full_df = pd.DataFrame(
        [
            {
                "variant":      "baseline",
                **baseline["full_metrics"],
                "avg_turnover": baseline["avg_turnover"],
                "total_rebalances": baseline["total_rebalances"],
            },
            {
                "variant":      "choppy_filter_reduce_exposure",
                **filtered["full_metrics"],
                "avg_turnover": filtered["avg_turnover"],
                "total_rebalances": filtered["total_rebalances"],
            },
        ]
    )

    worst_df = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "worst_window_return":       baseline["worst_metrics"]["final_multiple"] - 1.0,
                "worst_window_sharpe":       baseline["worst_metrics"]["sharpe"],
                "worst_window_max_drawdown": baseline["worst_metrics"]["max_drawdown"],
            },
            {
                "variant": "choppy_filter_reduce_exposure",
                "worst_window_return":       filtered["worst_metrics"]["final_multiple"] - 1.0,
                "worst_window_sharpe":       filtered["worst_metrics"]["sharpe"],
                "worst_window_max_drawdown": filtered["worst_metrics"]["max_drawdown"],
            },
        ]
    )

    diag_df = pd.DataFrame(
        [
            {
                "variant":                       "baseline",
                "total_rebalances":              baseline["total_rebalances"],
                "choppy_regime_rebalances":      0,
                "pct_rebalances_choppy":         0.0,
                "choppy_rebalances_in_ww":       0,
                "normal_rebalances_in_ww":
                    int(((baseline["rebal_df"]["rebalance_date"] >= pd.Timestamp(WINDOW_START))
                         & (baseline["rebal_df"]["rebalance_date"] <= pd.Timestamp(WINDOW_END))).sum()),
            },
            {
                "variant":                       "choppy_filter_reduce_exposure",
                "total_rebalances":              filtered["total_rebalances"],
                "choppy_regime_rebalances":      filtered["choppy_count"],
                "pct_rebalances_choppy":
                    filtered["choppy_count"] / max(filtered["total_rebalances"], 1),
                "choppy_rebalances_in_ww":       filtered["choppy_in_ww"],
                "normal_rebalances_in_ww":
                    int(((filtered["rebal_df"]["rebalance_date"] >= pd.Timestamp(WINDOW_START))
                         & (filtered["rebal_df"]["rebalance_date"] <= pd.Timestamp(WINDOW_END))).sum())
                    - filtered["choppy_in_ww"],
            },
        ]
    )

    exposure_df = pd.DataFrame(
        [
            {
                "variant":            "baseline",
                "avg_positions":      baseline["avg_positions"],
                "pct_time_some_cash": baseline["pct_time_some_cash"],
                "avg_cash_fraction":  baseline["avg_cash_fraction"],
            },
            {
                "variant":            "choppy_filter_reduce_exposure",
                "avg_positions":      filtered["avg_positions"],
                "pct_time_some_cash": filtered["pct_time_some_cash"],
                "avg_cash_fraction":  filtered["avg_cash_fraction"],
            },
        ]
    )

    return full_df, worst_df, diag_df, exposure_df


def _print_choppy_rebalance_details(
    choppy_rebal_df: pd.DataFrame, spy_close: pd.Series
) -> None:
    print(f"\n=== CHOPPY REBALANCE DETAILS (exposure reduced to {CHOPPY_EXPOSURE:.0%}) ===")
    if choppy_rebal_df.empty:
        print("No choppy-regime rebalances detected.")
        return

    rows: list[dict[str, Any]] = []
    for _, rec in choppy_rebal_df.iterrows():
        dt    = pd.Timestamp(rec["rebalance_date"])
        close = spy_close.loc[:dt].dropna()
        if len(close) < 50:
            rows.append({"rebalance_date": dt.date(), "note": "insufficient history"})
            continue

        sma20 = float(close.rolling(20, min_periods=20).mean().iloc[-1])
        sma50 = float(close.rolling(50, min_periods=50).mean().iloc[-1])
        c     = float(close.iloc[-1])
        rows.append(
            {
                "rebalance_date":      dt.date(),
                "target_exposure":     f"{rec['target_exposure']:.0%}",
                "selected_count":      int(rec["selected_count"]) if pd.notna(rec["selected_count"]) else "—",
                "turnover":            f"{rec['turnover']:.3f}",
                "spy_close":           round(c, 2),
                "sma20":               round(sma20, 2),
                "sma50":               round(sma50, 2),
                "close_vs_sma20_pct":  f"{(c / sma20 - 1) * 100:.2f}%",
                "sma20_vs_sma50_pct":  f"{(sma20 / sma50 - 1) * 100:.2f}%",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


def _print_interpretation(
    full_df: pd.DataFrame, worst_df: pd.DataFrame, diag_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
) -> None:
    base_full  = full_df.loc[full_df["variant"] == "baseline"].iloc[0]
    filt_full  = full_df.loc[full_df["variant"] == "choppy_filter_reduce_exposure"].iloc[0]
    base_worst = worst_df.loc[worst_df["variant"] == "baseline"].iloc[0]
    filt_worst = worst_df.loc[worst_df["variant"] == "choppy_filter_reduce_exposure"].iloc[0]
    filt_diag  = diag_df.loc[diag_df["variant"] == "choppy_filter_reduce_exposure"].iloc[0]

    cagr_change    = float(filt_full["cagr"]   - base_full["cagr"])
    sharpe_change  = float(filt_full["sharpe"] - base_full["sharpe"])
    ww_ret_change  = float(filt_worst["worst_window_return"] - base_worst["worst_window_return"])
    dd_delta       = float(filt_worst["worst_window_max_drawdown"] - base_worst["worst_window_max_drawdown"])

    dd_improved       = dd_delta > 0       # less negative = improved
    ww_ret_improved   = ww_ret_change > 0
    sharpe_improved   = sharpe_change > 0
    cagr_flag         = "significant" if cagr_change < -0.02 else "not significant"
    choppy_count      = int(filt_diag["choppy_regime_rebalances"])
    total_count       = int(filt_diag["total_rebalances"])
    choppy_in_ww      = int(filt_diag["choppy_rebalances_in_ww"])
    triggered_in_ww   = choppy_in_ww > 0

    print("\n=== INTERPRETATION ===")
    print(
        f"1) Worst-window max drawdown improved: {'yes' if dd_improved else 'no'} "
        f"(baseline={base_worst['worst_window_max_drawdown']:.2%}, "
        f"reduced_exposure={filt_worst['worst_window_max_drawdown']:.2%}, "
        f"delta={dd_delta:+.2%})."
    )
    print(
        f"2) Worst-window return improved: {'yes' if ww_ret_improved else 'no'} "
        f"(baseline={base_worst['worst_window_return']:.2%}, "
        f"reduced_exposure={filt_worst['worst_window_return']:.2%}, "
        f"delta={ww_ret_change:+.2%})."
    )
    print(
        f"3) Full-period Sharpe improved: {'yes' if sharpe_improved else 'no'} "
        f"(delta={sharpe_change:+.4f}, "
        f"baseline={base_full['sharpe']:.4f}, filtered={filt_full['sharpe']:.4f})."
    )
    print(
        f"4) CAGR change vs baseline: {cagr_change:+.2%} ({cagr_flag} by the -2.00pp rule)."
    )
    print(
        f"5) Choppy-regime trigger frequency: {choppy_count}/{total_count} rebalances "
        f"({choppy_count / max(total_count, 1):.1%} of all rebalances) triggered "
        f"the {CHOPPY_EXPOSURE:.0%} exposure cap."
    )
    print(
        f"6) Triggered inside worst window (2024-12-09 → 2025-03-12): "
        f"{'yes' if triggered_in_ww else 'no'} ({choppy_in_ww} choppy rebalance(s) "
        f"inside that window)."
    )
    print(
        f"7) Positions count unchanged: both variants average "
        f"baseline={exposure_df.loc[exposure_df['variant']=='baseline','avg_positions'].iloc[0]:.1f}, "
        f"filtered={exposure_df.loc[exposure_df['variant']=='choppy_filter_reduce_exposure','avg_positions'].iloc[0]:.1f} "
        f"(should be identical — only weights scale, not selection count)."
    )


def main() -> None:
    symbol_dfs, market_df = _load_data()
    market_df_norm = _ensure_datetime_index(_normalize_cols(market_df))
    spy_close = market_df_norm["close"].copy()

    print("[INFO] Running baseline variant...")
    baseline = _run_variant(
        name="baseline",
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=_build_cfg(market_filter_mode="none"),
    )

    print("[INFO] Running choppy_filter_reduce_exposure variant...")
    filtered = _run_variant(
        name="choppy_filter_reduce_exposure",
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=_build_cfg(market_filter_mode="choppy_filter_reduce_exposure"),
    )

    full_df, worst_df, diag_df, exposure_df = _build_output_tables(baseline, filtered)

    print("\n=== BASELINE VS CHOPPY REDUCE EXPOSURE (FULL OOS) ===")
    print(full_df.to_string(index=False))

    print("\n=== WORST WINDOW COMPARISON ===")
    print(worst_df.to_string(index=False))

    print("\n=== CHOPPY REGIME DIAGNOSTICS ===")
    print(diag_df.to_string(index=False))

    print("\n=== EXPOSURE SUMMARY ===")
    print(exposure_df.to_string(index=False))

    _print_choppy_rebalance_details(filtered["choppy_rebal_df"], spy_close)

    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(OUT_FULL,     index=False)
    worst_df.to_csv(OUT_WORST,   index=False)
    exposure_df.to_csv(OUT_EXPOSURE, index=False)

    # Full per-rebalance diagnostics (both variants) for deeper inspection
    diag_rows: list[dict[str, Any]] = []
    for variant_key, v in [("baseline", baseline), ("choppy_filter_reduce_exposure", filtered)]:
        for _, rec in v["rebal_df"].iterrows():
            diag_rows.append(
                {
                    "variant":         variant_key,
                    "rebalance_date":  rec["rebalance_date"],
                    "choppy_override": bool(rec.get("choppy_override", False)),
                    "target_exposure": rec.get("target_exposure"),
                    "risk_on":         rec.get("risk_on"),
                    "selected_count":  rec.get("selected_count"),
                    "turnover":        rec.get("turnover"),
                    "estimated_cost":  rec.get("estimated_cost"),
                }
            )
    pd.DataFrame(diag_rows).to_csv(OUT_DIAG, index=False)

    print(f"\n[INFO] Saved {OUT_FULL}")
    print(f"[INFO] Saved {OUT_WORST}")
    print(f"[INFO] Saved {OUT_EXPOSURE}")
    print(f"[INFO] Saved {OUT_DIAG}")

    _print_interpretation(full_df, worst_df, diag_df, exposure_df)


if __name__ == "__main__":
    main()
