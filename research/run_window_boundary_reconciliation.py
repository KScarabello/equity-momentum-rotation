#!/usr/bin/env python3
"""
Window-boundary reconciliation between walk-forward and live simulation engines.

Purpose: isolate remaining accounting/state continuity mismatches specifically at
walk-forward test-window transitions.

Run:
    python -m research.run_window_boundary_reconciliation
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from research.run_live_simulation import load_data, run_live_simulation
from research.walk_forward_momentum import (
    WalkForwardConfig,
    run_weekly_portfolio,
    _normalize_cols,
    _ensure_datetime_index,
)

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


def _window_trade_bounds(market_df: pd.DataFrame, ws: pd.Timestamp, we: pd.Timestamp) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    cal = market_df.index[(market_df.index >= ws) & (market_df.index <= we)]
    if len(cal) == 0:
        return None, None
    return pd.Timestamp(cal.min()), pd.Timestamp(cal.max())


def _collect_boundaries(
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    market_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    rows: list[dict] = []
    for i in range(1, len(windows)):
        pws, pwe = windows[i - 1]
        nws, nwe = windows[i]

        before_first, before_last = _window_trade_bounds(market_df, pws, pwe)
        after_first, after_last = _window_trade_bounds(market_df, nws, nwe)
        if before_last is None or after_first is None:
            continue

        if after_first < start_ts or after_first > end_ts:
            continue

        rows.append(
            {
                "boundary_date": after_first,
                "prior_window_end": pwe,
                "next_window_start": nws,
                "before_date": before_last,
                "after_date": after_first,
            }
        )

    return pd.DataFrame(rows).sort_values("boundary_date").reset_index(drop=True)


def _run_walk_forward_snapshots(
    symbol_dfs: dict[str, pd.DataFrame],
    market_df: pd.DataFrame,
    cfg: WalkForwardConfig,
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    snapshot_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    cash = float(cfg.starting_cash)
    holdings: dict[str, float] = {}
    rows: list[dict] = []

    for ws, we in windows:
        cfg_local = WalkForwardConfig(**{**cfg.__dict__, "starting_cash": cash})
        res = run_weekly_portfolio(
            symbol_dfs=symbol_dfs,
            market_df=market_df,
            start=ws,
            end=we,
            cfg=cfg_local,
            initial_cash=cash,
            initial_holdings=holdings,
            snapshot_dates=snapshot_dates,
        )

        rows.extend(res.state_snapshots)
        cash = float(res.ending_cash)
        holdings = {s: float(sh) for s, sh in res.ending_holdings.items()}

    if not rows:
        return pd.DataFrame(columns=["date", "equity", "cash", "invested_value", "holdings_count", "top3_holdings"])

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return out


def _run_live_snapshots(
    symbol_dfs: dict[str, pd.DataFrame],
    market_df: pd.DataFrame,
    cfg: WalkForwardConfig,
    start_date: str,
    end_date: str,
    snapshot_dates: set[pd.Timestamp],
    rebalance_reset_dates: set[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    _, _, _, summary = run_live_simulation(
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=cfg,
        sim_start=start_date,
        sim_end=end_date,
        snapshot_dates=snapshot_dates,
        rebalance_reset_dates=rebalance_reset_dates,
    )

    rows = summary.get("state_snapshots", [])
    if not rows:
        return pd.DataFrame(columns=["date", "equity", "cash", "invested_value", "holdings_count", "top3_holdings"])

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    return out


def _state_row(df: pd.DataFrame, dt: pd.Timestamp, prefix: str) -> dict:
    m = df[df["date"] == dt]
    if m.empty:
        return {
            f"{prefix}_equity": float("nan"),
            f"{prefix}_cash": float("nan"),
            f"{prefix}_invested": float("nan"),
            f"{prefix}_holdings_count": float("nan"),
            f"{prefix}_top3": "",
        }
    r = m.iloc[-1]
    return {
        f"{prefix}_equity": float(r["equity"]),
        f"{prefix}_cash": float(r["cash"]),
        f"{prefix}_invested": float(r["invested_value"]),
        f"{prefix}_holdings_count": int(r["holdings_count"]),
        f"{prefix}_top3": str(r["top3_holdings"]),
    }


def _build_comparison(boundaries: pd.DataFrame, wf_state: pd.DataFrame, sim_state: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for _, b in boundaries.iterrows():
        before_dt = pd.Timestamp(b["before_date"])
        after_dt = pd.Timestamp(b["after_date"])

        wf_before = _state_row(wf_state, before_dt, "wf_before")
        sim_before = _state_row(sim_state, before_dt, "sim_before")
        wf_after = _state_row(wf_state, after_dt, "wf_after")
        sim_after = _state_row(sim_state, after_dt, "sim_after")

        row = {
            "boundary_date": pd.Timestamp(b["boundary_date"]),
            "prior_window_end": pd.Timestamp(b["prior_window_end"]),
            "next_window_start": pd.Timestamp(b["next_window_start"]),
            "before_date": before_dt,
            "after_date": after_dt,

            "wf_equity_before": wf_before["wf_before_equity"],
            "sim_equity_before": sim_before["sim_before_equity"],
            "wf_cash_before": wf_before["wf_before_cash"],
            "sim_cash_before": sim_before["sim_before_cash"],
            "wf_invested_before": wf_before["wf_before_invested"],
            "sim_invested_before": sim_before["sim_before_invested"],
            "wf_holdings_count_before": wf_before["wf_before_holdings_count"],
            "sim_holdings_count_before": sim_before["sim_before_holdings_count"],
            "wf_top3_before": wf_before["wf_before_top3"],
            "sim_top3_before": sim_before["sim_before_top3"],

            "wf_equity_after": wf_after["wf_after_equity"],
            "sim_equity_after": sim_after["sim_after_equity"],
            "wf_cash_after": wf_after["wf_after_cash"],
            "sim_cash_after": sim_after["sim_after_cash"],
            "wf_invested_after": wf_after["wf_after_invested"],
            "sim_invested_after": sim_after["sim_after_invested"],
            "wf_holdings_count_after": wf_after["wf_after_holdings_count"],
            "sim_holdings_count_after": sim_after["sim_after_holdings_count"],
            "wf_top3_after": wf_after["wf_after_top3"],
            "sim_top3_after": sim_after["sim_after_top3"],
        }

        row["equity_diff_before"] = row["sim_equity_before"] - row["wf_equity_before"]
        row["equity_diff_after"] = row["sim_equity_after"] - row["wf_equity_after"]
        row["cash_diff_before"] = row["sim_cash_before"] - row["wf_cash_before"]
        row["cash_diff_after"] = row["sim_cash_after"] - row["wf_cash_after"]
        row["invested_diff_before"] = row["sim_invested_before"] - row["wf_invested_before"]
        row["invested_diff_after"] = row["sim_invested_after"] - row["wf_invested_after"]

        rows.append(row)

    return pd.DataFrame(rows).sort_values("boundary_date").reset_index(drop=True)


def _write_summary(comp: pd.DataFrame, out_fp: Path, material_threshold: float = 1.0) -> str:
    if comp.empty:
        text = "=== WINDOW BOUNDARY RECONCILIATION SUMMARY ===\nBoundaries checked: 0\n"
        out_fp.write_text(text, encoding="utf-8")
        return text

    before_abs = comp["equity_diff_before"].abs()
    after_abs = comp["equity_diff_after"].abs()
    cash_abs = pd.concat([comp["cash_diff_before"].abs(), comp["cash_diff_after"].abs()], axis=0)
    invested_abs = pd.concat([comp["invested_diff_before"].abs(), comp["invested_diff_after"].abs()], axis=0)

    mismatch_mask = (
        (before_abs > material_threshold)
        | (after_abs > material_threshold)
        | (comp["cash_diff_before"].abs() > material_threshold)
        | (comp["cash_diff_after"].abs() > material_threshold)
        | (comp["invested_diff_before"].abs() > material_threshold)
        | (comp["invested_diff_after"].abs() > material_threshold)
    )

    first_mismatch = comp.loc[mismatch_mask, "boundary_date"].min() if mismatch_mask.any() else pd.NaT

    lines = [
        "=== WINDOW BOUNDARY RECONCILIATION SUMMARY ===",
        f"Boundaries checked: {len(comp)}",
        f"First materially mismatched boundary: {first_mismatch.date() if pd.notna(first_mismatch) else 'None'}",
        f"Max equity diff before boundary: {before_abs.max():.2f}",
        f"Max equity diff after boundary: {after_abs.max():.2f}",
        f"Max cash diff: {cash_abs.max():.2f}",
        f"Max invested-value diff: {invested_abs.max():.2f}",
    ]

    text = "\n".join(lines) + "\n"
    out_fp.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    cfg = WalkForwardConfig(**BASE_CFG)

    print("[INFO] Loading data")
    symbol_dfs, market_df = load_data(market_symbol=cfg.market_symbol)
    market_df = _ensure_datetime_index(_normalize_cols(market_df))

    print("[INFO] Building walk-forward windows and boundaries...")
    windows = _build_walk_forward_windows(symbol_dfs, market_df, cfg)
    boundaries = _collect_boundaries(windows, market_df, START_DATE, END_DATE)

    if boundaries.empty:
        print("[WARN] No boundaries found in requested range.")
        empty = pd.DataFrame()
        empty.to_csv("window_boundary_reconciliation.csv", index=False)
        text = _write_summary(empty, Path("window_boundary_reconciliation_summary.txt"))
        print("\n" + text.rstrip())
        return

    snapshot_dates = set(pd.to_datetime(boundaries["before_date"]).tolist())
    snapshot_dates.update(set(pd.to_datetime(boundaries["after_date"]).tolist()))

    print("[INFO] Collecting walk-forward state snapshots...")
    wf_state = _run_walk_forward_snapshots(
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=cfg,
        windows=windows,
        snapshot_dates=snapshot_dates,
    )

    print("[INFO] Collecting live simulation state snapshots...")
    reset_dates = set(pd.to_datetime(boundaries["boundary_date"]).tolist())
    sim_state = _run_live_snapshots(
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=cfg,
        start_date=START_DATE,
        end_date=END_DATE,
        snapshot_dates=snapshot_dates,
        rebalance_reset_dates=reset_dates,
    )

    print("[INFO] Building boundary reconciliation table...")
    comp = _build_comparison(boundaries, wf_state, sim_state)

    comp.to_csv("window_boundary_reconciliation.csv", index=False)
    text = _write_summary(comp, Path("window_boundary_reconciliation_summary.txt"))

    print("\n" + text.rstrip())
    print("[INFO] Wrote window_boundary_reconciliation.csv")
    print("[INFO] Wrote window_boundary_reconciliation_summary.txt")


if __name__ == "__main__":
    main()
