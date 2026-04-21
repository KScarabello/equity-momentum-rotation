#!/usr/bin/env python3
"""
Compare baseline momentum strategy vs a simple momentum-strength filter.

New rule:
- After ranking by momentum score, only keep names with score > 0.
- If fewer than N remain, hold fewer names and keep residual cash.

Run:
    python3 -m research.run_strength_filter_test
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
OUT_CSV = Path("research/strength_filter_results.csv")


def _build_cfg(use_strength_filter: bool) -> WalkForwardConfig:
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
        use_strength_filter=use_strength_filter,
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
        return {
            "final_multiple": float("nan"),
            "cagr": float("nan"),
            "volatility": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }

    rets = equity.pct_change().dropna()
    if rets.empty:
        return {
            "final_multiple": float("nan"),
            "cagr": float("nan"),
            "volatility": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }

    final_multiple = float(equity.iloc[-1] / equity.iloc[0])
    cagr = float(final_multiple ** (252 / len(rets)) - 1.0)
    vol = float(rets.std() * sqrt(252))

    std = float(rets.std())
    sharpe = float((rets.mean() / std) * sqrt(252)) if std > 0 else float("nan")

    cum = (1 + rets).cumprod()
    max_drawdown = float((cum / cum.cummax() - 1.0).min())

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
    rebal_df = pd.DataFrame(rebalance_rows).sort_values("rebalance_date").reset_index(drop=True)
    snaps_df = pd.DataFrame(snapshot_rows).sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    full_metrics = _compute_metrics(equity_oos)

    ww = equity_oos.loc[pd.Timestamp(WINDOW_START):pd.Timestamp(WINDOW_END)]
    worst_metrics = _compute_metrics(ww)

    avg_positions = float(rebal_df["selected_count"].mean()) if not rebal_df.empty else float("nan")
    avg_turnover = float(rebal_df["turnover"].mean()) if not rebal_df.empty else float("nan")
    max_turnover = float(rebal_df["turnover"].max()) if not rebal_df.empty else float("nan")

    pct_time_some_cash = float((snaps_df["cash_fraction"] > 0.0).mean()) if not snaps_df.empty else float("nan")
    avg_cash_fraction = float(snaps_df["cash_fraction"].mean()) if not snaps_df.empty else float("nan")

    return {
        "name": name,
        "equity_oos": equity_oos,
        "rebalances": rebal_df,
        "snapshots": snaps_df,
        "full_metrics": full_metrics,
        "worst_metrics": worst_metrics,
        "avg_positions": avg_positions,
        "avg_turnover": avg_turnover,
        "max_turnover": max_turnover,
        "pct_time_some_cash": pct_time_some_cash,
        "avg_cash_fraction": avg_cash_fraction,
    }


def _comparison_rows(base: dict[str, Any], filt: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variant": "baseline",
                "use_strength_filter": False,
                **base["full_metrics"],
                "avg_positions": base["avg_positions"],
                "avg_turnover": base["avg_turnover"],
                "max_turnover": base["max_turnover"],
                "pct_time_some_cash": base["pct_time_some_cash"],
                "avg_cash_fraction": base["avg_cash_fraction"],
                "worst_window_return": base["worst_metrics"]["final_multiple"] - 1.0,
                "worst_window_sharpe": base["worst_metrics"]["sharpe"],
                "worst_window_max_drawdown": base["worst_metrics"]["max_drawdown"],
            },
            {
                "variant": "filtered",
                "use_strength_filter": True,
                **filt["full_metrics"],
                "avg_positions": filt["avg_positions"],
                "avg_turnover": filt["avg_turnover"],
                "max_turnover": filt["max_turnover"],
                "pct_time_some_cash": filt["pct_time_some_cash"],
                "avg_cash_fraction": filt["avg_cash_fraction"],
                "worst_window_return": filt["worst_metrics"]["final_multiple"] - 1.0,
                "worst_window_sharpe": filt["worst_metrics"]["sharpe"],
                "worst_window_max_drawdown": filt["worst_metrics"]["max_drawdown"],
            },
        ]
    )


def _print_interpretation(df: pd.DataFrame) -> None:
    base = df.loc[df["variant"] == "baseline"].iloc[0]
    filt = df.loc[df["variant"] == "filtered"].iloc[0]

    dd_improved = float(filt["worst_window_max_drawdown"]) > float(base["worst_window_max_drawdown"])
    sharpe_improved = float(filt["sharpe"]) > float(base["sharpe"])
    cagr_delta = float(filt["cagr"] - base["cagr"])

    print("\n=== INTERPRETATION ===")
    print(
        "Worst-window drawdown improved: "
        f"{'yes' if dd_improved else 'no'} "
        f"(baseline={base['worst_window_max_drawdown']:.2%}, filtered={filt['worst_window_max_drawdown']:.2%})"
    )
    print(
        "Sharpe improved: "
        f"{'yes' if sharpe_improved else 'no'} "
        f"(baseline={base['sharpe']:.3f}, filtered={filt['sharpe']:.3f})"
    )
    print(
        "CAGR change: "
        f"{cagr_delta:+.2%} "
        "(significant drop if less than -2.00%)"
    )


def main() -> None:
    symbol_dfs, market_df = _load_data()

    print("[INFO] Running baseline variant (no strength filter)...")
    baseline = _run_variant(
        name="baseline",
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=_build_cfg(use_strength_filter=False),
    )

    print("[INFO] Running filtered variant (momentum_score > 0)...")
    filtered = _run_variant(
        name="filtered",
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=_build_cfg(use_strength_filter=True),
    )

    results_df = _comparison_rows(baseline, filtered)

    print("\n=== BASELINE VS FILTERED (FULL OOS) ===")
    full_cols = [
        "variant",
        "final_multiple",
        "cagr",
        "volatility",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
    ]
    print(results_df[full_cols].to_string(index=False))

    print("\n=== WORST WINDOW COMPARISON ===")
    worst_cols = [
        "variant",
        "worst_window_return",
        "worst_window_sharpe",
        "worst_window_max_drawdown",
    ]
    print(results_df[worst_cols].to_string(index=False))

    print("\n=== EXPOSURE SUMMARY ===")
    exposure_cols = [
        "variant",
        "avg_positions",
        "pct_time_some_cash",
        "avg_cash_fraction",
    ]
    print(results_df[exposure_cols].to_string(index=False))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUT_CSV, index=False)
    print(f"\n[INFO] Saved {OUT_CSV}")

    _print_interpretation(results_df)


if __name__ == "__main__":
    main()
