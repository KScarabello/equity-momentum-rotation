#!/usr/bin/env python3
"""
Reconcile walk-forward and live-simulation engines at rebalance-event level.

Goal: identify the first rebalance where the engines diverge and quantify how.

Run:
    python -m research.run_engine_reconciliation
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from research.run_live_simulation import load_data, run_live_simulation
from research.walk_forward_momentum import (
    WalkForwardConfig,
    run_weekly_portfolio,
    _normalize_cols,
    _ensure_datetime_index,
)

# Reference configuration.
BASE_CFG = dict(
    positions=12,
    rebalance_interval_weeks=2,
    w_3m=0.60,
    w_6m=0.30,
    w_12m=0.10,
    cost_bps=5.0,
    slippage_bps=2.0,
    risk_off_exposure=0.25,
    min_exposure=0.25,
    max_exposure=1.0,
    exposure_slope=0.0,
    market_sma_days=200,
    require_positive_sma_slope=True,
    sma_slope_lookback=20,
    risk_on_buffer=0.0,
    min_rebalance_weight_change=0.0,
    stability_lookback_periods=1,
)

START_DATE = "2022-11-07"
END_DATE = "2025-12-24"


def _month_delta(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    return (ts + pd.DateOffset(months=months)).normalize()


def _year_delta(ts: pd.Timestamp, years: int) -> pd.Timestamp:
    return (ts + pd.DateOffset(years=years)).normalize()


def _build_walk_forward_windows(
    symbol_dfs: dict[str, pd.DataFrame],
    market_df: pd.DataFrame,
    cfg: WalkForwardConfig,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    market_df = _ensure_datetime_index(_normalize_cols(market_df))
    symbol_dfs = {s: _ensure_datetime_index(_normalize_cols(df)) for s, df in symbol_dfs.items()}

    market_dates = market_df.index
    min_need = max(cfg.liq_lookback, cfg.mom_12m + 1, cfg.market_sma_days + 1)
    start_candidates = [
        df.index[min_need] for df in symbol_dfs.values() if len(df.index) > min_need
    ]
    if not start_candidates:
        raise ValueError("Not enough history across symbols for configured lookbacks.")

    global_start = max(max(start_candidates), market_dates[min_need])
    global_end = market_dates.max()

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = _year_delta(global_start, cfg.train_years)
    if cursor > global_end:
        return windows

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

    return windows


def _run_walk_forward_rebalance_records(
    symbol_dfs: dict[str, pd.DataFrame],
    market_df: pd.DataFrame,
    cfg: WalkForwardConfig,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    windows = _build_walk_forward_windows(symbol_dfs, market_df, cfg)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    cash = float(cfg.starting_cash)
    holdings: dict[str, float] = {}
    rows: list[dict] = []

    for ws, we in windows:
        # Keep walk-forward mechanics unchanged: carry ending cash from prior window.
        cfg_local = WalkForwardConfig(**{**cfg.__dict__, "starting_cash": cash})
        res = run_weekly_portfolio(
            symbol_dfs=symbol_dfs,
            market_df=market_df,
            start=ws,
            end=we,
            cfg=cfg_local,
            initial_cash=cash,
            initial_holdings=holdings,
        )

        for rec in res.rebalance_records:
            dt = pd.Timestamp(rec["rebalance_date"])
            if start_ts <= dt <= end_ts:
                rows.append(
                    {
                        "rebalance_date": dt,
                        "engine_name": "walk_forward",
                        "equity_before_rebalance": float(rec["equity_before_rebalance"]),
                        "risk_on": bool(rec["risk_on"]),
                        "target_exposure": float(rec["target_exposure"]),
                        "selected_symbols": str(rec["selected_symbols"]),
                        "turnover_for_rebalance": float(rec["turnover"]),
                        "estimated_costs_for_rebalance": float(rec["estimated_cost"]),
                        "estimated_slippage_for_rebalance": float(rec["estimated_slippage"]),
                        "equity_after_rebalance": float(rec["equity_after_rebalance"]),
                    }
                )

        cash = float(res.ending_cash)
        holdings = {s: float(sh) for s, sh in res.ending_holdings.items()}

    return pd.DataFrame(rows).sort_values("rebalance_date").reset_index(drop=True)


def _run_live_sim_rebalance_records(
    symbol_dfs: dict[str, pd.DataFrame],
    market_df: pd.DataFrame,
    cfg: WalkForwardConfig,
    start_date: str,
    end_date: str,
    rebalance_reset_dates: set[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    equity_df, trades_df, diagnostics_df, _ = run_live_simulation(
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=cfg,
        sim_start=start_date,
        sim_end=end_date,
        rebalance_reset_dates=rebalance_reset_dates,
    )

    if diagnostics_df.empty:
        return pd.DataFrame(
            columns=[
                "rebalance_date",
                "engine_name",
                "equity_before_rebalance",
                "risk_on",
                "target_exposure",
                "selected_symbols",
                "turnover_for_rebalance",
                "estimated_costs_for_rebalance",
                "estimated_slippage_for_rebalance",
                "equity_after_rebalance",
            ]
        )

    trades = trades_df.copy()
    if trades.empty:
        cost_by_date = pd.DataFrame(
            {
                "rebalance_date": diagnostics_df["rebalance_date"].copy(),
                "estimated_costs_for_rebalance": 0.0,
                "estimated_slippage_for_rebalance": 0.0,
            }
        )
    else:
        trades["rebalance_date"] = pd.to_datetime(trades["rebalance_date"])
        cost_by_date = (
            trades.groupby("rebalance_date", as_index=False)[["estimated_cost", "estimated_slippage"]]
            .sum()
            .rename(
                columns={
                    "estimated_cost": "estimated_costs_for_rebalance",
                    "estimated_slippage": "estimated_slippage_for_rebalance",
                }
            )
        )

    eq_on_rebal = (
        equity_df[["date", "equity"]]
        .rename(columns={"date": "rebalance_date", "equity": "equity_after_rebalance"})
        .copy()
    )
    eq_on_rebal["rebalance_date"] = pd.to_datetime(eq_on_rebal["rebalance_date"])

    sim = diagnostics_df.copy()
    sim["rebalance_date"] = pd.to_datetime(sim["rebalance_date"])
    sim = sim.merge(cost_by_date, on="rebalance_date", how="left")
    sim = sim.merge(eq_on_rebal, on="rebalance_date", how="left")

    sim["estimated_costs_for_rebalance"] = sim["estimated_costs_for_rebalance"].fillna(0.0)
    sim["estimated_slippage_for_rebalance"] = sim["estimated_slippage_for_rebalance"].fillna(0.0)

    out = pd.DataFrame(
        {
            "rebalance_date": sim["rebalance_date"],
            "engine_name": "live_sim",
            "equity_before_rebalance": sim["equity_before"].astype(float),
            "risk_on": sim["risk_on"].astype(bool),
            "target_exposure": sim["target_exposure_pct"].astype(float) / 100.0,
            "selected_symbols": sim["chosen_symbols"].astype(str),
            "turnover_for_rebalance": sim["rebalance_turnover_pct"].astype(float) / 100.0,
            "estimated_costs_for_rebalance": sim["estimated_costs_for_rebalance"].astype(float),
            "estimated_slippage_for_rebalance": sim["estimated_slippage_for_rebalance"].astype(float),
            "equity_after_rebalance": sim["equity_after_rebalance"].astype(float),
        }
    )

    return out.sort_values("rebalance_date").reset_index(drop=True)


def _symbol_set(s: object) -> set[str]:
    if s is None:
        return set()
    text = str(s).strip()
    if text == "" or text.lower() == "nan":
        return set()
    return {x for x in text.split("|") if x}


def _reconciliation_table(wf_df: pd.DataFrame, sim_df: pd.DataFrame) -> pd.DataFrame:
    wf = wf_df.rename(
        columns={
            "risk_on": "wf_risk_on",
            "target_exposure": "wf_target_exposure",
            "selected_symbols": "wf_symbols",
            "turnover_for_rebalance": "wf_turnover",
            "estimated_costs_for_rebalance": "wf_costs",
            "estimated_slippage_for_rebalance": "wf_slippage",
            "equity_after_rebalance": "wf_equity",
            "equity_before_rebalance": "wf_equity_before",
        }
    )[[
        "rebalance_date",
        "wf_risk_on",
        "wf_target_exposure",
        "wf_symbols",
        "wf_turnover",
        "wf_costs",
        "wf_slippage",
        "wf_equity_before",
        "wf_equity",
    ]]

    sim = sim_df.rename(
        columns={
            "risk_on": "sim_risk_on",
            "target_exposure": "sim_target_exposure",
            "selected_symbols": "sim_symbols",
            "turnover_for_rebalance": "sim_turnover",
            "estimated_costs_for_rebalance": "sim_costs",
            "estimated_slippage_for_rebalance": "sim_slippage",
            "equity_after_rebalance": "sim_equity",
            "equity_before_rebalance": "sim_equity_before",
        }
    )[[
        "rebalance_date",
        "sim_risk_on",
        "sim_target_exposure",
        "sim_symbols",
        "sim_turnover",
        "sim_costs",
        "sim_slippage",
        "sim_equity_before",
        "sim_equity",
    ]]

    comp = wf.merge(sim, on="rebalance_date", how="inner").sort_values("rebalance_date").reset_index(drop=True)

    overlaps = []
    wf_only = []
    sim_only = []
    exact_match = []
    for _, r in comp.iterrows():
        w = _symbol_set(r["wf_symbols"])
        s = _symbol_set(r["sim_symbols"])
        overlaps.append(len(w & s))
        wf_only_set = sorted(w - s)
        sim_only_set = sorted(s - w)
        wf_only.append("|".join(wf_only_set))
        sim_only.append("|".join(sim_only_set))
        exact_match.append(w == s)

    comp["symbol_overlap_count"] = overlaps
    comp["symbols_exact_match"] = exact_match
    comp["wf_only_symbols"] = wf_only
    comp["sim_only_symbols"] = sim_only

    comp["regime_match"] = comp["wf_risk_on"] == comp["sim_risk_on"]
    comp["target_exposure_diff"] = (comp["wf_target_exposure"] - comp["sim_target_exposure"]).abs()
    comp["turnover_diff"] = (comp["wf_turnover"] - comp["sim_turnover"]).abs()
    comp["cost_diff"] = (comp["wf_costs"] - comp["sim_costs"]).abs()
    comp["slippage_diff"] = (comp["wf_slippage"] - comp["sim_slippage"]).abs()
    comp["equity_diff"] = comp["sim_equity"] - comp["wf_equity"]
    comp["equity_diff_abs"] = comp["equity_diff"].abs()

    return comp


def _write_summary(comp: pd.DataFrame, out_fp: Path) -> str:
    if comp.empty:
        text = "No overlapping rebalance dates between engines.\n"
        out_fp.write_text(text, encoding="utf-8")
        return text

    mismatch_mask = (
        (~comp["regime_match"]) |
        (~comp["symbols_exact_match"]) |
        (comp["target_exposure_diff"] > 1e-9)
    )
    first_div = comp.loc[mismatch_mask, "rebalance_date"].min() if mismatch_mask.any() else pd.NaT

    regime_mismatch_count = int((~comp["regime_match"]).sum())
    selection_mismatch_count = int((~comp["symbols_exact_match"]).sum())
    avg_turnover_diff = float(comp["turnover_diff"].mean())
    avg_cost_diff = float(comp["cost_diff"].mean())
    max_equity_diff = float(comp["equity_diff_abs"].max())

    lines = [
        "=== ENGINE RECONCILIATION SUMMARY ===",
        f"First divergence date: {first_div.date() if pd.notna(first_div) else 'None'}",
        f"Regime mismatches: {regime_mismatch_count}",
        f"Selection mismatches: {selection_mismatch_count}",
        f"Average turnover difference: {avg_turnover_diff:.6f}",
        f"Average cost difference: {avg_cost_diff:.6f}",
        f"Max equity difference: {max_equity_diff:.2f}",
    ]

    text = "\n".join(lines) + "\n"
    out_fp.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    cfg = WalkForwardConfig(**BASE_CFG)

    print("[INFO] Loading data")
    symbol_dfs, market_df = load_data(market_symbol=cfg.market_symbol)

    # Use walk-forward boundary starts as cadence reset anchors so live simulation
    # rebalancing schedule matches per-window anchoring behavior.
    windows = _build_walk_forward_windows(symbol_dfs, market_df, cfg)
    reset_dates = {pd.Timestamp(ws) for ws, _ in windows}

    print("[INFO] Collecting walk-forward rebalance records...")
    wf_df = _run_walk_forward_rebalance_records(
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=cfg,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    print("[INFO] Collecting live simulation rebalance records...")
    sim_df = _run_live_sim_rebalance_records(
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=cfg,
        start_date=START_DATE,
        end_date=END_DATE,
        rebalance_reset_dates=reset_dates,
    )

    print("[INFO] Building merged reconciliation table...")
    comp = _reconciliation_table(wf_df, sim_df)

    comp.to_csv("engine_reconciliation.csv", index=False)
    summary_text = _write_summary(comp, Path("engine_reconciliation_summary.txt"))

    print("\n" + summary_text.rstrip())
    print("[INFO] Wrote engine_reconciliation.csv")
    print("[INFO] Wrote engine_reconciliation_summary.txt")


if __name__ == "__main__":
    main()
