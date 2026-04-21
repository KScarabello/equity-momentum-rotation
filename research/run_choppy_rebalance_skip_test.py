#!/usr/bin/env python3
"""
Compare baseline momentum strategy vs a true choppy-market rebalance skip overlay.

Filtered rule (choppy_rebalance_skip):
- On each scheduled rebalance date, evaluate the same SPY-based choppy classifier
  used by the existing choppy experiments.
- If ALL three conditions are true, the rebalance is skipped entirely.
- When skipped: holdings, cash, and weights are carried forward unchanged;
  no reranking, no resizing, no turnover, and no rebalance costs.

Run:
    python3 -m research.run_choppy_rebalance_skip_test
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
CHOPPY_VOL_LOOKBACK = 126

OUT_FULL = Path("research/choppy_rebalance_skip_results.csv")
OUT_WORST = Path("research/choppy_rebalance_skip_worst_window.csv")
OUT_EXPOSURE = Path("research/choppy_rebalance_skip_exposure_summary.csv")
OUT_DIAG = Path("research/choppy_rebalance_skip_diagnostics.csv")
OUT_HOLDINGS = Path("research/choppy_rebalance_skip_holdings_check.csv")


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


def _choppy_conditions_detail(
    spy_close: pd.Series, asof: pd.Timestamp, vol_lookback: int
) -> dict[str, Any]:
    close = spy_close.loc[:asof].dropna()
    if len(close) < 50 + vol_lookback + 20:
        return {"data_insufficient": True}

    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    if pd.isna(sma20.iloc[-1]) or pd.isna(sma50.iloc[-1]):
        return {"data_insufficient": True}

    c = float(close.iloc[-1])
    s20 = float(sma20.iloc[-1])
    s50 = float(sma50.iloc[-1])

    cond1 = abs(c / s20 - 1.0) < 0.02
    cond2 = abs(s20 / s50 - 1.0) < 0.015

    daily_rets = close.pct_change().dropna()
    rolling_vol = daily_rets.rolling(20, min_periods=20).std() * np.sqrt(252)
    rv_clean = rolling_vol.dropna()

    if len(rv_clean) < vol_lookback:
        current_vol = float(rv_clean.iloc[-1]) if not rv_clean.empty else float("nan")
        vol_median = float("nan")
        cond3 = None
    else:
        current_vol = float(rv_clean.iloc[-1])
        vol_median = float(rv_clean.iloc[-vol_lookback:].median())
        cond3 = current_vol > vol_median

    return {
        "spy_close": round(c, 4),
        "sma20": round(s20, 4),
        "sma50": round(s50, 4),
        "realized_vol_20d": round(current_vol, 4),
        "rolling_median_realized_vol_126d": round(vol_median, 4) if np.isfinite(vol_median) else float("nan"),
        "close_vs_sma20_pct": round((c / s20 - 1.0) * 100, 3),
        "sma20_vs_sma50_pct": round((s20 / s50 - 1.0) * 100, 3),
        "cond1_price_near_sma20": cond1,
        "cond2_sma20_near_sma50": cond2,
        "cond3_vol_above_median": cond3,
        "all_conditions_met": bool(cond1 and cond2 and cond3 is True),
    }


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
            was_skipped = bool(rec.get("skipped", False))
            if bool(rec.get("choppy_override", False)):
                raise RuntimeError(
                    f"FAIL: variant={name} emitted choppy exposure scaling records; expected true skip logic only."
                )

            symbols = [s for s in str(rec.get("selected_symbols", "")).split("|") if s]
            rebalance_rows.append(
                {
                    "rebalance_date": pd.Timestamp(rec["rebalance_date"]),
                    "skipped": was_skipped,
                    "risk_on": bool(rec.get("risk_on", False)),
                    "selected_count": len(symbols) if not was_skipped else None,
                    "turnover": float(rec.get("turnover", 0.0)),
                    "estimated_cost": float(rec.get("estimated_cost", 0.0)),
                    "estimated_slippage": float(rec.get("estimated_slippage", 0.0)),
                    "target_exposure": float(rec.get("target_exposure", 0.0)),
                    "selected_symbols": "|".join(symbols),
                    "holdings_count_before": rec.get("holdings_count_before"),
                    "holdings_count_after": rec.get("holdings_count_after"),
                    "holdings_signature_before": rec.get("holdings_signature_before", ""),
                    "holdings_signature_after": rec.get("holdings_signature_after", ""),
                    "cash_before": rec.get("cash_before"),
                    "cash_after": rec.get("cash_after"),
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
    ww = equity_oos.loc[pd.Timestamp(WINDOW_START):pd.Timestamp(WINDOW_END)]
    worst_metrics = _compute_metrics(ww)

    skipped_df = rebal_df[rebal_df["skipped"]].copy()
    executed_df = rebal_df[~rebal_df["skipped"]].copy()

    if not skipped_df.empty and (skipped_df["turnover"] > 1e-9).any():
        raise RuntimeError("FAIL: skipped rebalances contain non-zero turnover.")

    holdings_check_df = pd.DataFrame(
        {
            "skipped_rebalance_date": skipped_df["rebalance_date"],
            "prior_holdings_count": skipped_df["holdings_count_before"],
            "post_skip_holdings_count": skipped_df["holdings_count_after"],
            "holdings_changed": (
                (skipped_df["holdings_count_before"] != skipped_df["holdings_count_after"])
                | (skipped_df["holdings_signature_before"] != skipped_df["holdings_signature_after"])
                | ((skipped_df["cash_before"].fillna(0.0) - skipped_df["cash_after"].fillna(0.0)).abs() > 1e-9)
            ),
            "turnover_on_skipped_date": skipped_df["turnover"],
        }
    ) if not skipped_df.empty else pd.DataFrame(
        columns=[
            "skipped_rebalance_date",
            "prior_holdings_count",
            "post_skip_holdings_count",
            "holdings_changed",
            "turnover_on_skipped_date",
        ]
    )

    if not holdings_check_df.empty and holdings_check_df["holdings_changed"].any():
        raise RuntimeError("FAIL: skipped rebalances changed holdings or cash state.")

    ww_start = pd.Timestamp(WINDOW_START)
    ww_end = pd.Timestamp(WINDOW_END)
    skipped_in_ww = int(
        ((skipped_df["rebalance_date"] >= ww_start) & (skipped_df["rebalance_date"] <= ww_end)).sum()
    ) if not skipped_df.empty else 0
    executed_in_ww = int(
        ((executed_df["rebalance_date"] >= ww_start) & (executed_df["rebalance_date"] <= ww_end)).sum()
    ) if not executed_df.empty else 0

    avg_positions = float(executed_df["selected_count"].mean()) if not executed_df.empty else float("nan")
    avg_turnover = float(executed_df["turnover"].mean()) if not executed_df.empty else float("nan")
    pct_time_some_cash = float((snaps_df["cash_fraction"] > 0.0).mean()) if not snaps_df.empty else float("nan")
    avg_cash_fraction = float(snaps_df["cash_fraction"].mean()) if not snaps_df.empty else float("nan")

    return {
        "name": name,
        "equity_oos": equity_oos,
        "rebal_df": rebal_df,
        "skipped_df": skipped_df,
        "executed_df": executed_df,
        "holdings_check_df": holdings_check_df,
        "snapshots": snaps_df,
        "full_metrics": full_metrics,
        "worst_metrics": worst_metrics,
        "avg_positions": avg_positions,
        "avg_turnover": avg_turnover,
        "pct_time_some_cash": pct_time_some_cash,
        "avg_cash_fraction": avg_cash_fraction,
        "total_rebalances": int(len(rebal_df)),
        "executed_rebalances": int(len(executed_df)),
        "skipped_rebalances": int(len(skipped_df)),
        "skipped_in_ww": skipped_in_ww,
        "executed_in_ww": executed_in_ww,
    }


def _build_output_tables(
    baseline: dict[str, Any], filtered: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_df = pd.DataFrame(
        [
            {
                "variant": "baseline",
                **baseline["full_metrics"],
                "avg_turnover": baseline["avg_turnover"],
                "total_rebalances": baseline["total_rebalances"],
                "executed_rebalances": baseline["executed_rebalances"],
                "skipped_rebalances": baseline["skipped_rebalances"],
            },
            {
                "variant": "choppy_rebalance_skip",
                **filtered["full_metrics"],
                "avg_turnover": filtered["avg_turnover"],
                "total_rebalances": filtered["total_rebalances"],
                "executed_rebalances": filtered["executed_rebalances"],
                "skipped_rebalances": filtered["skipped_rebalances"],
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
                "variant": "choppy_rebalance_skip",
                "worst_window_return": filtered["worst_metrics"]["final_multiple"] - 1.0,
                "worst_window_sharpe": filtered["worst_metrics"]["sharpe"],
                "worst_window_max_drawdown": filtered["worst_metrics"]["max_drawdown"],
            },
        ]
    )

    baseline_dates = "|".join(
        d.strftime("%Y-%m-%d") for d in baseline["skipped_df"]["rebalance_date"]
    ) if not baseline["skipped_df"].empty else ""
    filtered_dates = "|".join(
        d.strftime("%Y-%m-%d") for d in filtered["skipped_df"]["rebalance_date"]
    ) if not filtered["skipped_df"].empty else ""

    diag_df = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "total_rebalances": baseline["total_rebalances"],
                "choppy_rebalance_dates": baseline_dates,
                "skipped_rebalances": baseline["skipped_rebalances"],
                "executed_rebalances": baseline["executed_rebalances"],
                "pct_rebalances_skipped": 0.0,
                "skipped_rebalances_inside_worst_window": baseline["skipped_in_ww"],
                "executed_rebalances_inside_worst_window": baseline["executed_in_ww"],
            },
            {
                "variant": "choppy_rebalance_skip",
                "total_rebalances": filtered["total_rebalances"],
                "choppy_rebalance_dates": filtered_dates,
                "skipped_rebalances": filtered["skipped_rebalances"],
                "executed_rebalances": filtered["executed_rebalances"],
                "pct_rebalances_skipped": (
                    filtered["skipped_rebalances"] / filtered["total_rebalances"]
                    if filtered["total_rebalances"] > 0 else 0.0
                ),
                "skipped_rebalances_inside_worst_window": filtered["skipped_in_ww"],
                "executed_rebalances_inside_worst_window": filtered["executed_in_ww"],
            },
        ]
    )

    holdings_df = filtered["holdings_check_df"].copy()

    exposure_df = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "avg_positions": baseline["avg_positions"],
                "pct_time_some_cash": baseline["pct_time_some_cash"],
                "avg_cash_fraction": baseline["avg_cash_fraction"],
            },
            {
                "variant": "choppy_rebalance_skip",
                "avg_positions": filtered["avg_positions"],
                "pct_time_some_cash": filtered["pct_time_some_cash"],
                "avg_cash_fraction": filtered["avg_cash_fraction"],
            },
        ]
    )

    return full_df, worst_df, diag_df, holdings_df, exposure_df


def _print_skipped_rebalance_details(
    skipped_df: pd.DataFrame, spy_close: pd.Series, vol_lookback: int
) -> None:
    print("\n=== SKIPPED REBALANCE DETAILS ===")
    if skipped_df.empty:
        print("No skipped rebalances.")
        return

    rows: list[dict[str, Any]] = []
    for _, rec in skipped_df.iterrows():
        dt = pd.Timestamp(rec["rebalance_date"])
        detail = _choppy_conditions_detail(spy_close, dt, vol_lookback)
        rows.append({"rebalance_date": dt.date(), **detail})

    detail_df = pd.DataFrame(rows)
    print(detail_df.to_string(index=False))


def _print_interpretation(
    full_df: pd.DataFrame,
    worst_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    holdings_df: pd.DataFrame,
) -> None:
    base_full = full_df.loc[full_df["variant"] == "baseline"].iloc[0]
    filt_full = full_df.loc[full_df["variant"] == "choppy_rebalance_skip"].iloc[0]
    base_worst = worst_df.loc[worst_df["variant"] == "baseline"].iloc[0]
    filt_worst = worst_df.loc[worst_df["variant"] == "choppy_rebalance_skip"].iloc[0]
    filt_diag = diag_df.loc[diag_df["variant"] == "choppy_rebalance_skip"].iloc[0]

    cagr_change = float(filt_full["cagr"] - base_full["cagr"])
    sharpe_change = float(filt_full["sharpe"] - base_full["sharpe"])
    ww_return_change = float(filt_worst["worst_window_return"] - base_worst["worst_window_return"])
    dd_delta = float(filt_worst["worst_window_max_drawdown"] - base_worst["worst_window_max_drawdown"])

    holdings_preserved = bool(
        holdings_df.empty
        or (
            (~holdings_df["holdings_changed"]).all()
            and (holdings_df["turnover_on_skipped_date"] <= 1e-9).all()
        )
    )

    cagr_flag = "significant" if cagr_change < -0.02 else "not significant"
    skip_count = int(filt_diag["skipped_rebalances"])
    total_count = int(filt_diag["total_rebalances"])
    skip_in_ww = int(filt_diag["skipped_rebalances_inside_worst_window"])

    print("\n=== INTERPRETATION ===")
    print(
        f"1) Did worst-window drawdown improve? {'yes' if dd_delta > 0 else 'no'} "
        f"(baseline={base_worst['worst_window_max_drawdown']:.2%}, "
        f"filtered={filt_worst['worst_window_max_drawdown']:.2%}, delta={dd_delta:+.2%})."
    )
    print(
        f"2) Did worst-window return improve? {'yes' if ww_return_change > 0 else 'no'} "
        f"(baseline={base_worst['worst_window_return']:.2%}, "
        f"filtered={filt_worst['worst_window_return']:.2%}, delta={ww_return_change:+.2%})."
    )
    print(
        f"3) Did full-period Sharpe improve? {'yes' if sharpe_change > 0 else 'no'} "
        f"(baseline={base_full['sharpe']:.4f}, filtered={filt_full['sharpe']:.4f}, delta={sharpe_change:+.4f})."
    )
    print(
        f"4) Did CAGR fall materially? {'yes' if cagr_change < -0.02 else 'no'} "
        f"(change={cagr_change:+.2%}; flagged {cagr_flag})."
    )
    print(
        f"5) How often did the filter trigger? {skip_count}/{total_count} rebalances "
        f"({skip_count / max(total_count, 1):.1%} skipped)."
    )
    print(
        f"6) Did it trigger during the known bad window? {'yes' if skip_in_ww > 0 else 'no'} "
        f"({skip_in_ww} skipped inside {WINDOW_START} -> {WINDOW_END})."
    )
    print(
        f"7) Did skipped rebalances truly preserve holdings and avoid turnover? "
        f"{'yes' if holdings_preserved else 'no'}"
        f"; holdings_changed_any={bool(not holdings_df.empty and holdings_df['holdings_changed'].any())}, "
        f"max_turnover_on_skipped_date={float(holdings_df['turnover_on_skipped_date'].max()) if not holdings_df.empty else 0.0:.6f}."
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

    print("[INFO] Running choppy_rebalance_skip variant...")
    filtered = _run_variant(
        name="choppy_rebalance_skip",
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=_build_cfg(market_filter_mode="skip_choppy_rebalance"),
    )

    full_df, worst_df, diag_df, holdings_df, exposure_df = _build_output_tables(baseline, filtered)

    if filtered["total_rebalances"] != baseline["total_rebalances"]:
        raise RuntimeError("FAIL: filtered variant changed the scheduled rebalance count.")

    print("\n=== BASELINE VS CHOPPY REBALANCE SKIP (FULL OOS) ===")
    print(full_df.to_string(index=False))

    print("\n=== WORST WINDOW COMPARISON ===")
    print(worst_df.to_string(index=False))

    print("\n=== CHOPPY SKIP DIAGNOSTICS ===")
    print(diag_df.to_string(index=False))

    print("\n=== HOLDINGS CONTINUITY CHECK ===")
    print(holdings_df.to_string(index=False) if not holdings_df.empty else "No skipped rebalances.")

    print("\n=== EXPOSURE SUMMARY ===")
    print(exposure_df.to_string(index=False))

    _print_skipped_rebalance_details(filtered["skipped_df"], spy_close, CHOPPY_VOL_LOOKBACK)

    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(OUT_FULL, index=False)
    worst_df.to_csv(OUT_WORST, index=False)
    exposure_df.to_csv(OUT_EXPOSURE, index=False)
    diag_df.to_csv(OUT_DIAG, index=False)
    holdings_df.to_csv(OUT_HOLDINGS, index=False)

    print(f"\n[INFO] Saved {OUT_FULL}")
    print(f"[INFO] Saved {OUT_WORST}")
    print(f"[INFO] Saved {OUT_EXPOSURE}")
    print(f"[INFO] Saved {OUT_DIAG}")
    print(f"[INFO] Saved {OUT_HOLDINGS}")

    _print_interpretation(full_df, worst_df, diag_df, holdings_df)


if __name__ == "__main__":
    main()
