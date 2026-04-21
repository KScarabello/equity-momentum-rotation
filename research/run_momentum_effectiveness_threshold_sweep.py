#!/usr/bin/env python3
"""
Threshold sweep for momentum-effectiveness rebalance-skip filter.

Variants:
- baseline: never skip on momentum effectiveness
- me_skip_lt_0: skip if momentum_effectiveness < 0.00
- me_skip_lt_neg_0p02: skip if momentum_effectiveness < -0.02

Run:
    python3 -m research.run_momentum_effectiveness_threshold_sweep
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Any, Optional

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

OUT_FULL = Path("research/momentum_effectiveness_threshold_sweep_results.csv")
OUT_WORST = Path("research/momentum_effectiveness_threshold_sweep_worst_window.csv")
OUT_DIAG = Path("research/momentum_effectiveness_threshold_sweep_diagnostics.csv")
OUT_VALUES = Path("research/momentum_effectiveness_threshold_sweep_values.csv")
OUT_HOLDINGS = Path("research/momentum_effectiveness_threshold_sweep_holdings_check.csv")


VARIANTS: list[tuple[str, Optional[float]]] = [
    ("baseline", None),
    ("me_skip_lt_0", 0.00),
    ("me_skip_lt_neg_0p02", -0.02),
]


def _build_cfg(skip_threshold: Optional[float]) -> WalkForwardConfig:
    # market_filter_mode kept on momentum_effectiveness_skip for all variants so the
    # same effectiveness series is computed on rebalance dates.
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
        market_filter_mode="momentum_effectiveness_skip",
        momentum_effectiveness_lookback=63,
        momentum_effectiveness_skip_threshold=skip_threshold,
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
    std = float(returns.std())
    sharpe = float((returns.mean() / std) * sqrt(252)) if std > 0 else float("nan")

    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = float(drawdown.min())

    return {
        "final_multiple": final_multiple,
        "cagr": cagr,
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
    rebal_rows: list[dict[str, Any]] = []

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
            skipped = bool(rec.get("skipped", False))
            if bool(rec.get("choppy_override", False)):
                raise RuntimeError(
                    f"FAIL: variant={name} emitted choppy exposure override; expected momentum-effectiveness path only."
                )

            symbols = [s for s in str(rec.get("selected_symbols", "")).split("|") if s]
            rebal_rows.append(
                {
                    "rebalance_date": pd.Timestamp(rec["rebalance_date"]),
                    "skipped": skipped,
                    "momentum_effectiveness": rec.get("momentum_effectiveness"),
                    "turnover": float(rec.get("turnover", 0.0)),
                    "estimated_cost": float(rec.get("estimated_cost", 0.0)),
                    "selected_count": len(symbols) if not skipped else None,
                    "holdings_count_before": rec.get("holdings_count_before"),
                    "holdings_count_after": rec.get("holdings_count_after"),
                    "holdings_signature_before": rec.get("holdings_signature_before", ""),
                    "holdings_signature_after": rec.get("holdings_signature_after", ""),
                }
            )

        cash = float(res.ending_cash)
        holdings = {s: float(sh) for s, sh in res.ending_holdings.items()}

    if not oos_dates:
        raise RuntimeError(f"No OOS output for variant={name}")

    equity_oos = pd.Series(oos_values, index=pd.DatetimeIndex(oos_dates), name=f"equity_{name}")
    if equity_oos.empty:
        raise RuntimeError(f"Empty OOS equity series for variant={name}")

    rebal_df = pd.DataFrame(rebal_rows).sort_values("rebalance_date").reset_index(drop=True)
    if rebal_df.empty:
        raise RuntimeError(f"No rebalance records for variant={name}")

    full_metrics = _compute_metrics(equity_oos)
    ww = equity_oos.loc[pd.Timestamp(WINDOW_START):pd.Timestamp(WINDOW_END)]
    worst_metrics = _compute_metrics(ww)

    skipped_df = rebal_df[rebal_df["skipped"]].copy()
    executed_df = rebal_df[~rebal_df["skipped"]].copy()

    if not skipped_df.empty and (skipped_df["turnover"] > 1e-9).any():
        raise RuntimeError(f"FAIL: skipped rebalances have non-zero turnover for {name}.")

    holdings_check_df = pd.DataFrame(
        {
            "variant": name,
            "skipped_rebalance_date": skipped_df["rebalance_date"],
            "prior_holdings_count": skipped_df["holdings_count_before"],
            "post_skip_holdings_count": skipped_df["holdings_count_after"],
            "holdings_changed": (
                (skipped_df["holdings_count_before"] != skipped_df["holdings_count_after"])
                | (skipped_df["holdings_signature_before"] != skipped_df["holdings_signature_after"])
            ),
            "turnover_on_skipped_date": skipped_df["turnover"],
        }
    ) if not skipped_df.empty else pd.DataFrame(
        columns=[
            "variant",
            "skipped_rebalance_date",
            "prior_holdings_count",
            "post_skip_holdings_count",
            "holdings_changed",
            "turnover_on_skipped_date",
        ]
    )

    if not holdings_check_df.empty and holdings_check_df["holdings_changed"].any():
        raise RuntimeError(f"FAIL: holdings changed on skipped dates for {name}.")

    avg_turnover = float(rebal_df["turnover"].mean())

    ww_start = pd.Timestamp(WINDOW_START)
    ww_end = pd.Timestamp(WINDOW_END)
    skipped_in_ww = int(
        ((skipped_df["rebalance_date"] >= ww_start) & (skipped_df["rebalance_date"] <= ww_end)).sum()
    ) if not skipped_df.empty else 0

    return {
        "name": name,
        "equity_oos": equity_oos,
        "rebal_df": rebal_df,
        "full_metrics": full_metrics,
        "worst_metrics": worst_metrics,
        "avg_turnover": avg_turnover,
        "total_rebalances": int(len(rebal_df)),
        "skipped_rebalances": int(len(skipped_df)),
        "executed_rebalances": int(len(executed_df)),
        "pct_skipped": float(len(skipped_df) / max(len(rebal_df), 1)),
        "skipped_in_worst_window": skipped_in_ww,
        "holdings_check_df": holdings_check_df,
    }


def _build_tables(results: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    order = ["baseline", "me_skip_lt_0", "me_skip_lt_neg_0p02"]

    full_rows = []
    worst_rows = []
    diag_rows = []
    holdings_rows = []

    for name in order:
        r = results[name]
        full_rows.append(
            {
                "variant": name,
                "final_multiple": r["full_metrics"]["final_multiple"],
                "cagr": r["full_metrics"]["cagr"],
                "sharpe": r["full_metrics"]["sharpe"],
                "max_drawdown": r["full_metrics"]["max_drawdown"],
                "avg_turnover": r["avg_turnover"],
            }
        )
        worst_rows.append(
            {
                "variant": name,
                "worst_window_return": r["worst_metrics"]["final_multiple"] - 1.0,
                "worst_window_sharpe": r["worst_metrics"]["sharpe"],
                "worst_window_max_drawdown": r["worst_metrics"]["max_drawdown"],
            }
        )
        diag_rows.append(
            {
                "variant": name,
                "total_rebalances": r["total_rebalances"],
                "skipped_rebalances": r["skipped_rebalances"],
                "executed_rebalances": r["executed_rebalances"],
                "pct_skipped": r["pct_skipped"],
                "skipped_in_worst_window": r["skipped_in_worst_window"],
            }
        )
        if not r["holdings_check_df"].empty:
            holdings_rows.append(r["holdings_check_df"])

    full_df = pd.DataFrame(full_rows)
    worst_df = pd.DataFrame(worst_rows)
    diag_df = pd.DataFrame(diag_rows)
    holdings_df = (
        pd.concat(holdings_rows, ignore_index=True) if holdings_rows else pd.DataFrame(
            columns=[
                "variant",
                "skipped_rebalance_date",
                "prior_holdings_count",
                "post_skip_holdings_count",
                "holdings_changed",
                "turnover_on_skipped_date",
            ]
        )
    )

    base_df = results["baseline"]["rebal_df"][["rebalance_date", "momentum_effectiveness", "skipped"]].copy()
    lt0_df = results["me_skip_lt_0"]["rebal_df"][["rebalance_date", "skipped"]].copy()
    lt002_df = results["me_skip_lt_neg_0p02"]["rebal_df"][["rebalance_date", "skipped"]].copy()

    base_df["baseline_decision"] = np.where(base_df["skipped"], "skip", "execute")
    lt0_df["me_skip_lt_0_decision"] = np.where(lt0_df["skipped"], "skip", "execute")
    lt002_df["me_skip_lt_neg_0p02_decision"] = np.where(lt002_df["skipped"], "skip", "execute")

    values_df = (
        base_df[["rebalance_date", "momentum_effectiveness", "baseline_decision"]]
        .merge(lt0_df[["rebalance_date", "me_skip_lt_0_decision"]], on="rebalance_date", how="inner")
        .merge(lt002_df[["rebalance_date", "me_skip_lt_neg_0p02_decision"]], on="rebalance_date", how="inner")
        .sort_values("rebalance_date")
        .reset_index(drop=True)
    )

    return full_df, worst_df, diag_df, values_df, holdings_df


def _print_interpretation(full_df: pd.DataFrame, worst_df: pd.DataFrame, diag_df: pd.DataFrame) -> None:
    b = full_df.loc[full_df["variant"] == "baseline"].iloc[0]
    v0 = full_df.loc[full_df["variant"] == "me_skip_lt_0"].iloc[0]
    v2 = full_df.loc[full_df["variant"] == "me_skip_lt_neg_0p02"].iloc[0]

    wb = worst_df.loc[worst_df["variant"] == "baseline"].iloc[0]
    w0 = worst_df.loc[worst_df["variant"] == "me_skip_lt_0"].iloc[0]
    w2 = worst_df.loc[worst_df["variant"] == "me_skip_lt_neg_0p02"].iloc[0]

    d0 = diag_df.loc[diag_df["variant"] == "me_skip_lt_0"].iloc[0]
    d2 = diag_df.loc[diag_df["variant"] == "me_skip_lt_neg_0p02"].iloc[0]

    cagr_delta0 = float(v0["cagr"] - b["cagr"])
    cagr_delta2 = float(v2["cagr"] - b["cagr"])
    sharpe_delta0 = float(v0["sharpe"] - b["sharpe"])
    sharpe_delta2 = float(v2["sharpe"] - b["sharpe"])
    ww_delta0 = float(w0["worst_window_return"] - wb["worst_window_return"])
    ww_delta2 = float(w2["worst_window_return"] - wb["worst_window_return"])

    fewer_skips = int(d2["skipped_rebalances"]) < int(d0["skipped_rebalances"])
    preserve_worst_window = ww_delta2 > 0
    recover_sharpe = float(v2["sharpe"]) > float(v0["sharpe"])
    improve_dd_vs_lt0 = float(v2["max_drawdown"]) > float(v0["max_drawdown"])

    # Simple tradeoff score (higher is better): sharpe + worst-window return - drawdown magnitude.
    score = {
        "baseline": float(b["sharpe"] + wb["worst_window_return"] + b["max_drawdown"]),
        "me_skip_lt_0": float(v0["sharpe"] + w0["worst_window_return"] + v0["max_drawdown"]),
        "me_skip_lt_neg_0p02": float(v2["sharpe"] + w2["worst_window_return"] + v2["max_drawdown"]),
    }
    best_variant = max(score, key=score.get)

    def _cagr_flag(x: float) -> str:
        return "significant" if x < -0.02 else "not significant"

    print("\n=== INTERPRETATION ===")
    print(f"1) Did the -0.02 threshold skip fewer rebalances than the <0 version? {'yes' if fewer_skips else 'no'}.")
    print(
        f"2) Did the -0.02 threshold preserve some of the worst-window improvement? "
        f"{'yes' if preserve_worst_window else 'no'} (delta vs baseline={ww_delta2:+.2%})."
    )
    print(
        f"3) Did the -0.02 threshold recover full-period Sharpe relative to the <0 version? "
        f"{'yes' if recover_sharpe else 'no'} (delta vs <0={float(v2['sharpe'] - v0['sharpe']):+.4f})."
    )
    print(
        f"4) Did full-period max drawdown improve relative to the <0 version? "
        f"{'yes' if improve_dd_vs_lt0 else 'no'} "
        f"(<0={v0['max_drawdown']:.2%}, -0.02={v2['max_drawdown']:.2%})."
    )
    print(f"5) Which variant looks like the best tradeoff? {best_variant}.")

    print("\nDelta vs baseline:")
    print(
        f"- me_skip_lt_0: cagr_delta_vs_baseline={cagr_delta0:+.2%} "
        f"({_cagr_flag(cagr_delta0)}), sharpe_delta_vs_baseline={sharpe_delta0:+.4f}, "
        f"worst_window_return_delta_vs_baseline={ww_delta0:+.2%}."
    )
    print(
        f"- me_skip_lt_neg_0p02: cagr_delta_vs_baseline={cagr_delta2:+.2%} "
        f"({_cagr_flag(cagr_delta2)}), sharpe_delta_vs_baseline={sharpe_delta2:+.4f}, "
        f"worst_window_return_delta_vs_baseline={ww_delta2:+.2%}."
    )


def main() -> None:
    symbol_dfs, market_df = _load_data()

    results: dict[str, dict[str, Any]] = {}
    for i, (name, threshold) in enumerate(VARIANTS, start=1):
        print(f"[INFO] Running variant {i}/{len(VARIANTS)}: {name} (threshold={threshold})")
        results[name] = _run_variant(
            name=name,
            symbol_dfs=symbol_dfs,
            market_df=market_df,
            cfg=_build_cfg(skip_threshold=threshold),
        )

    # Failure conditions
    for name in ["me_skip_lt_0", "me_skip_lt_neg_0p02"]:
        r = results[name]
        if r["rebal_df"]["momentum_effectiveness"].dropna().empty:
            raise RuntimeError(f"FAIL: effectiveness calculation is empty for {name}.")

    v0_dec = np.where(results["me_skip_lt_0"]["rebal_df"]["skipped"], "skip", "execute")
    v2_dec = np.where(results["me_skip_lt_neg_0p02"]["rebal_df"]["skipped"], "skip", "execute")
    if np.array_equal(v0_dec, v2_dec):
        raise RuntimeError("FAIL: threshold variant decisions are identical to the <0 version.")

    full_df, worst_df, diag_df, values_df, holdings_df = _build_tables(results)

    if not holdings_df.empty:
        if holdings_df["holdings_changed"].any():
            raise RuntimeError("FAIL: holdings changed on skipped rebalances.")
        if (holdings_df["turnover_on_skipped_date"] > 1e-9).any():
            raise RuntimeError("FAIL: skipped rebalances still show non-zero turnover.")

    print("\n=== FULL OOS COMPARISON ===")
    print(full_df.to_string(index=False))

    print("\n=== WORST WINDOW COMPARISON ===")
    print(worst_df.to_string(index=False))

    print("\n=== SKIP DIAGNOSTICS ===")
    print(diag_df.to_string(index=False))

    print("\n=== EFFECTIVENESS VALUES TABLE ===")
    print(values_df.to_string(index=False))

    print("\n=== HOLDINGS CONTINUITY CHECK ===")
    if holdings_df.empty:
        print("No skipped rebalances to validate.")
    else:
        # Keep console readable while still printing representative rows if long.
        if len(holdings_df) > 40:
            print(holdings_df.head(40).to_string(index=False))
            print(f"... ({len(holdings_df)} rows total)")
        else:
            print(holdings_df.to_string(index=False))

    OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(OUT_FULL, index=False)
    worst_df.to_csv(OUT_WORST, index=False)
    diag_df.to_csv(OUT_DIAG, index=False)
    values_df.to_csv(OUT_VALUES, index=False)
    holdings_df.to_csv(OUT_HOLDINGS, index=False)

    print(f"\n[INFO] Saved {OUT_FULL}")
    print(f"[INFO] Saved {OUT_WORST}")
    print(f"[INFO] Saved {OUT_DIAG}")
    print(f"[INFO] Saved {OUT_VALUES}")
    print(f"[INFO] Saved {OUT_HOLDINGS}")

    _print_interpretation(full_df, worst_df, diag_df)


if __name__ == "__main__":
    main()
