#!/usr/bin/env python3
"""
Run regime breakdown analysis for the walk-forward momentum strategy.

This script keeps baseline strategy settings fixed and analyzes OOS performance
across calendar segments, rolling windows, and SPY-defined market regimes.

Run:
    python3 -m research.run_regime_breakdown
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from research.run_walk_forward import STOOQ_DIR, build_universe_from_stooq, fetch_ohlcv
from research.walk_forward_momentum import WalkForwardConfig, walk_forward_validate


OUT_CALENDAR = Path("research/regime_breakdown_calendar_year.csv")
OUT_REGIMES = Path("research/regime_breakdown_market_regimes.csv")
OUT_ROLLING = Path("research/regime_breakdown_rolling_summary.csv")


def compute_metrics(series: pd.Series) -> dict:
    series = series.dropna()
    if len(series) < 2:
        return {
            "final_multiple": float("nan"),
            "cagr": float("nan"),
            "vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "rows": int(len(series)),
        }

    returns = series.pct_change().dropna()
    if returns.empty:
        return {
            "final_multiple": float("nan"),
            "cagr": float("nan"),
            "vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "rows": int(len(series)),
        }

    final_multiple = float(series.iloc[-1] / series.iloc[0])
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
        "vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "rows": int(len(series)),
    }


def _build_baseline_cfg() -> WalkForwardConfig:
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


def _load_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.Series]:
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

    spy_close = market_df["close"].copy()
    print(f"[INFO] Loaded OHLCV for {len(symbol_dfs)} symbols")
    return symbol_dfs, market_df, spy_close


def get_baseline_oos_equity(symbol_dfs: dict[str, pd.DataFrame], market_df: pd.DataFrame) -> pd.Series:
    cfg = _build_baseline_cfg()
    _, equity_oos, _ = walk_forward_validate(
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=cfg,
        return_debug=True,
    )
    if equity_oos.empty:
        raise RuntimeError("Baseline OOS equity is empty.")
    return equity_oos


def _align_strategy_spy(strategy_equity: pd.Series, spy_close: pd.Series) -> pd.DataFrame:
    combined = pd.concat([strategy_equity.rename("strategy"), spy_close.rename("spy")], axis=1, sort=False).dropna()
    if combined.empty:
        raise RuntimeError("Aligned strategy/SPY OOS range is empty.")

    # Normalize SPY to strategy initial equity to make level-based metrics comparable.
    combined["spy_equity"] = (combined["spy"] / combined["spy"].iloc[0]) * float(combined["strategy"].iloc[0])
    return combined


def build_calendar_breakdown(aligned: pd.DataFrame) -> pd.DataFrame:
    periods = [
        ("2022_partial", "2022-11-07", "2022-12-31"),
        ("2023", "2023-01-01", "2023-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025_partial", "2025-01-01", "2025-12-24"),
    ]

    rows: list[dict] = []
    for label, start, end in periods:
        seg = aligned.loc[start:end]
        if seg.empty:
            continue

        s = compute_metrics(seg["strategy"])
        b = compute_metrics(seg["spy_equity"])

        rows.append(
            {
                "period_label": label,
                "start_date": str(seg.index.min().date()),
                "end_date": str(seg.index.max().date()),
                "rows": int(len(seg)),
                "strategy_final_multiple": s["final_multiple"],
                "strategy_cagr": s["cagr"],
                "strategy_vol": s["vol"],
                "strategy_sharpe": s["sharpe"],
                "strategy_max_drawdown": s["max_drawdown"],
                "spy_final_multiple": b["final_multiple"],
                "spy_cagr": b["cagr"],
                "spy_vol": b["vol"],
                "spy_sharpe": b["sharpe"],
                "spy_max_drawdown": b["max_drawdown"],
            }
        )

    return pd.DataFrame(rows)


def _rolling_max_drawdown(window_returns: pd.Series) -> float:
    cum = (1 + window_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def build_rolling_summary(aligned: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    s_rets = aligned["strategy"].pct_change().dropna()
    b_rets = aligned["spy_equity"].pct_change().dropna()

    common_idx = s_rets.index.intersection(b_rets.index)
    s_rets = s_rets.loc[common_idx]
    b_rets = b_rets.loc[common_idx]

    if len(common_idx) < window:
        raise RuntimeError(f"Not enough aligned return rows for rolling window={window}.")

    rows = []
    for i in range(window - 1, len(common_idx)):
        idx_slice = common_idx[i - window + 1 : i + 1]
        w_start = idx_slice[0]
        w_end = idx_slice[-1]

        sr = s_rets.loc[idx_slice]
        br = b_rets.loc[idx_slice]

        s_ret = float((1 + sr).prod() - 1)
        b_ret = float((1 + br).prod() - 1)

        s_std = float(sr.std())
        b_std = float(br.std())
        s_sharpe = float((sr.mean() / s_std) * sqrt(252)) if s_std > 0 else float("nan")
        b_sharpe = float((br.mean() / b_std) * sqrt(252)) if b_std > 0 else float("nan")

        s_mdd = _rolling_max_drawdown(sr)
        b_mdd = _rolling_max_drawdown(br)

        rows.append(
            {
                "window_start": w_start,
                "window_end": w_end,
                "strategy_rolling_return": s_ret,
                "spy_rolling_return": b_ret,
                "strategy_rolling_sharpe": s_sharpe,
                "spy_rolling_sharpe": b_sharpe,
                "strategy_rolling_max_drawdown": s_mdd,
                "spy_rolling_max_drawdown": b_mdd,
            }
        )

    roll = pd.DataFrame(rows)

    best_ret_idx = roll["strategy_rolling_return"].idxmax()
    worst_ret_idx = roll["strategy_rolling_return"].idxmin()
    best_sharpe_idx = roll["strategy_rolling_sharpe"].idxmax()
    worst_sharpe_idx = roll["strategy_rolling_sharpe"].idxmin()

    def pick(metric_label: str, idx: int, s_col: str, b_col: str) -> dict:
        r = roll.loc[idx]
        return {
            "metric_label": metric_label,
            "window_start": str(pd.Timestamp(r["window_start"]).date()),
            "window_end": str(pd.Timestamp(r["window_end"]).date()),
            "strategy_value": float(r[s_col]),
            "spy_value": float(r[b_col]),
        }

    summary = pd.DataFrame(
        [
            pick("best_63d_return", best_ret_idx, "strategy_rolling_return", "spy_rolling_return"),
            pick("worst_63d_return", worst_ret_idx, "strategy_rolling_return", "spy_rolling_return"),
            pick("best_63d_sharpe", best_sharpe_idx, "strategy_rolling_sharpe", "spy_rolling_sharpe"),
            pick("worst_63d_sharpe", worst_sharpe_idx, "strategy_rolling_sharpe", "spy_rolling_sharpe"),
        ]
    )

    return summary


def build_market_regime_breakdown(aligned: pd.DataFrame) -> pd.DataFrame:
    spy = aligned["spy"]
    sma200 = spy.rolling(200, min_periods=200).mean()
    slope = sma200 - sma200.shift(20)

    regime = pd.Series(index=aligned.index, dtype="object")
    regime[(spy > sma200) & (slope > 0)] = "Uptrend"
    regime[(spy < sma200) & (slope < 0)] = "Downtrend"
    regime = regime.fillna("Neutral")

    rows: list[dict] = []
    for reg in ["Uptrend", "Neutral", "Downtrend"]:
        seg = aligned.loc[regime == reg]
        if seg.empty:
            continue

        s = compute_metrics(seg["strategy"])
        b = compute_metrics(seg["spy_equity"])

        rows.append(
            {
                "regime": reg,
                "days": int(len(seg)),
                "strategy_cagr": s["cagr"],
                "strategy_sharpe": s["sharpe"],
                "strategy_max_drawdown": s["max_drawdown"],
                "spy_cagr": b["cagr"],
                "spy_sharpe": b["sharpe"],
                "spy_max_drawdown": b["max_drawdown"],
            }
        )

    return pd.DataFrame(rows)


def _print_human_summary(calendar_df: pd.DataFrame, regime_df: pd.DataFrame) -> None:
    if calendar_df.empty:
        print("[WARN] Calendar breakdown is empty; skipping human summary.")
        return

    best_year = calendar_df.loc[calendar_df["strategy_cagr"].idxmax()]
    worst_year = calendar_df.loc[calendar_df["strategy_cagr"].idxmin()]

    print("\n=== SUMMARY INSIGHTS ===")
    print(
        f"Best calendar segment for strategy: {best_year['period_label']} "
        f"(CAGR={best_year['strategy_cagr']:.4f}, Sharpe={best_year['strategy_sharpe']:.3f})"
    )
    print(
        f"Worst calendar segment for strategy: {worst_year['period_label']} "
        f"(CAGR={worst_year['strategy_cagr']:.4f}, Sharpe={worst_year['strategy_sharpe']:.3f})"
    )

    print("Strategy vs SPY by calendar segment:")
    for _, r in calendar_df.iterrows():
        beat = "BEAT" if float(r["strategy_cagr"]) > float(r["spy_cagr"]) else "LAGGED"
        print(
            f"- {r['period_label']}: strategy_cagr={r['strategy_cagr']:.4f}, "
            f"spy_cagr={r['spy_cagr']:.4f} -> {beat}"
        )

    if regime_df.empty:
        print("No market regime rows available.")
        return

    favored = regime_df.loc[regime_df["strategy_cagr"].idxmax()]
    hurt = regime_df.loc[regime_df["strategy_cagr"].idxmin()]
    print(
        f"Regime favoring strategy most: {favored['regime']} "
        f"(strategy_cagr={favored['strategy_cagr']:.4f}, spy_cagr={favored['spy_cagr']:.4f})"
    )
    print(
        f"Regime hurting strategy most: {hurt['regime']} "
        f"(strategy_cagr={hurt['strategy_cagr']:.4f}, spy_cagr={hurt['spy_cagr']:.4f})"
    )


def main() -> None:
    symbol_dfs, market_df, spy_close = _load_data()

    print("[INFO] Running baseline walk-forward to get OOS equity...")
    strategy_oos = get_baseline_oos_equity(symbol_dfs, market_df)

    aligned = _align_strategy_spy(strategy_oos, spy_close)

    calendar_df = build_calendar_breakdown(aligned)
    regime_df = build_market_regime_breakdown(aligned)
    rolling_summary_df = build_rolling_summary(aligned, window=63)

    print("\n=== CALENDAR YEAR BREAKDOWN ===")
    print(calendar_df.to_string(index=False))

    print("\n=== MARKET REGIME BREAKDOWN ===")
    print(regime_df.to_string(index=False))

    print("\n=== ROLLING 63D SUMMARY ===")
    print(rolling_summary_df.to_string(index=False))

    _print_human_summary(calendar_df, regime_df)

    OUT_CALENDAR.parent.mkdir(parents=True, exist_ok=True)
    calendar_df.to_csv(OUT_CALENDAR, index=False)
    regime_df.to_csv(OUT_REGIMES, index=False)
    rolling_summary_df.to_csv(OUT_ROLLING, index=False)

    print(f"\n[INFO] Saved {OUT_CALENDAR}")
    print(f"[INFO] Saved {OUT_REGIMES}")
    print(f"[INFO] Saved {OUT_ROLLING}")


if __name__ == "__main__":
    main()
