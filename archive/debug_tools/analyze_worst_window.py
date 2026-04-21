#!/usr/bin/env python3
"""
Analyze the worst 63-day strategy window in detail.

Target worst window (from prior regime analysis):
    2024-12-09 -> 2025-03-12

Run:
    python3 -m research.analyze_worst_window
"""

from __future__ import annotations

from collections import Counter
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

OUT_SYMBOL_FREQUENCY = Path("research/worst_window_symbol_frequency.csv")
OUT_REBALANCE_SUMMARY = Path("research/worst_window_rebalance_summary.csv")


def _build_baseline_cfg() -> WalkForwardConfig:
    # Baseline settings are intentionally fixed for diagnosis-only analysis.
    return WalkForwardConfig(
        train_years=3,
        test_months=6,
        step_months=6,
        positions=12,
        universe_top_n=800,
        rebalance_weekday=0,
        rebalance_interval_weeks=3,  # rebalance_days ~ 15 trading days
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
            "window_return": float("nan"),
            "cagr": float("nan"),
            "vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }

    rets = equity.pct_change().dropna()
    if rets.empty:
        return {
            "window_return": float("nan"),
            "cagr": float("nan"),
            "vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }

    window_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (252 / len(rets)) - 1.0)
    vol = float(rets.std() * sqrt(252))

    std = float(rets.std())
    sharpe = float((rets.mean() / std) * sqrt(252)) if std > 0 else float("nan")

    cum = (1 + rets).cumprod()
    dd = (cum / cum.cummax() - 1.0).min()

    return {
        "window_return": window_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd),
    }


def _build_walk_forward_windows(
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


def _run_full_oos_with_details(
    symbol_dfs: dict[str, pd.DataFrame], market_df: pd.DataFrame, cfg: WalkForwardConfig
) -> tuple[pd.Series, pd.DataFrame]:
    market_df, symbol_dfs, windows = _build_walk_forward_windows(symbol_dfs, market_df, cfg)

    cash = cfg.starting_cash
    holdings: dict[str, float] = {}

    all_equity_values: list[float] = []
    all_equity_dates: list[pd.Timestamp] = []
    rebalance_rows: list[dict[str, Any]] = []

    for window_start, window_end in windows:
        cfg_window = WalkForwardConfig(**{**cfg.__dict__, "starting_cash": cash})
        result = run_weekly_portfolio(
            symbol_dfs=symbol_dfs,
            market_df=market_df,
            start=window_start,
            end=window_end,
            cfg=cfg_window,
            initial_cash=cash,
            initial_holdings=holdings,
        )

        eq = result.equity_curve
        if all_equity_dates:
            eq = eq[eq.index > all_equity_dates[-1]]

        all_equity_values.extend(eq.values.tolist())
        all_equity_dates.extend(eq.index.tolist())

        for rec in result.rebalance_records:
            symbols = [s for s in str(rec.get("selected_symbols", "")).split("|") if s]
            selected_count = len(symbols)
            target_exposure = float(rec.get("target_exposure", 0.0))
            implied_weight = target_exposure / selected_count if selected_count > 0 else 0.0

            rebalance_rows.append(
                {
                    "rebalance_date": pd.Timestamp(rec["rebalance_date"]),
                    "window_start": window_start,
                    "window_end": window_end,
                    "risk_on": bool(rec.get("risk_on", False)),
                    "target_exposure": target_exposure,
                    "selected_count": selected_count,
                    "selected_symbols": "|".join(symbols),
                    "implied_equal_symbol_weight": implied_weight,
                    "turnover": float(rec.get("turnover", 0.0)),
                    "estimated_cost": float(rec.get("estimated_cost", 0.0)),
                    "estimated_slippage": float(rec.get("estimated_slippage", 0.0)),
                    "equity_before_rebalance": float(rec.get("equity_before_rebalance", np.nan)),
                    "equity_after_rebalance": float(rec.get("equity_after_rebalance", np.nan)),
                }
            )

        cash = float(result.ending_cash)
        holdings = {s: float(sh) for s, sh in result.ending_holdings.items()}

    if not all_equity_dates:
        raise RuntimeError("No OOS equity generated.")

    combined_equity = pd.Series(all_equity_values, index=pd.DatetimeIndex(all_equity_dates), name="equity_oos")
    rebalance_df = pd.DataFrame(rebalance_rows).sort_values("rebalance_date").reset_index(drop=True)

    return combined_equity, rebalance_df


def _safe_return(symbol_dfs: dict[str, pd.DataFrame], symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    df = symbol_dfs.get(symbol)
    if df is None or df.empty:
        return None

    close = _ensure_datetime_index(_normalize_cols(df))["close"]
    s_hist = close.loc[:start]
    e_hist = close.loc[:end]
    if s_hist.empty or e_hist.empty:
        return None

    s_px = float(s_hist.iloc[-1])
    e_px = float(e_hist.iloc[-1])
    if s_px <= 0:
        return None

    return e_px / s_px - 1.0


def _build_symbol_frequency(window_rebals: pd.DataFrame) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    for sym_str in window_rebals["selected_symbols"].fillna(""):
        symbols = [s for s in str(sym_str).split("|") if s]
        counter.update(symbols)

    if not counter:
        return pd.DataFrame(columns=["symbol", "times_selected"])

    freq_df = pd.DataFrame(
        [{"symbol": sym, "times_selected": cnt} for sym, cnt in counter.items()]
    ).sort_values(["times_selected", "symbol"], ascending=[False, True])

    return freq_df.reset_index(drop=True)


def _build_rebalance_summary(
    window_rebals: pd.DataFrame,
    all_rebals: pd.DataFrame,
    symbol_dfs: dict[str, pd.DataFrame],
    spy_close: pd.Series,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    if window_rebals.empty:
        return pd.DataFrame()

    all_dates = all_rebals["rebalance_date"].tolist()
    date_to_pos = {d: i for i, d in enumerate(all_dates)}

    rows: list[dict[str, Any]] = []
    for _, row in window_rebals.iterrows():
        reb_date = pd.Timestamp(row["rebalance_date"])
        symbols = [s for s in str(row["selected_symbols"]).split("|") if s]

        pos = date_to_pos[reb_date]
        if pos + 1 < len(all_dates):
            next_reb = pd.Timestamp(all_dates[pos + 1])
        else:
            next_reb = window_end

        if next_reb < reb_date:
            next_reb = reb_date

        rets: dict[str, float] = {}
        for sym in symbols:
            r = _safe_return(symbol_dfs, sym, reb_date, next_reb)
            if r is not None and np.isfinite(r):
                rets[sym] = float(r)

        avg_pick_ret = float(np.mean(list(rets.values()))) if rets else float("nan")

        best_symbol = ""
        best_symbol_return = float("nan")
        worst_symbol = ""
        worst_symbol_return = float("nan")

        if rets:
            best_symbol = max(rets, key=rets.get)
            best_symbol_return = rets[best_symbol]
            worst_symbol = min(rets, key=rets.get)
            worst_symbol_return = rets[worst_symbol]

        spy_ret = _safe_return({"SPY": pd.DataFrame({"close": spy_close})}, "SPY", reb_date, next_reb)

        rows.append(
            {
                "rebalance_date": reb_date,
                "next_rebalance_date": next_reb,
                "holding_days": int((next_reb - reb_date).days),
                "risk_on": bool(row["risk_on"]),
                "target_exposure": float(row["target_exposure"]),
                "selected_count": int(row["selected_count"]),
                "implied_equal_symbol_weight": float(row["implied_equal_symbol_weight"]),
                "turnover": float(row["turnover"]),
                "selected_symbols": "|".join(symbols),
                "avg_selected_return_to_next_rebalance": avg_pick_ret,
                "best_symbol": best_symbol,
                "best_symbol_return": best_symbol_return,
                "worst_symbol": worst_symbol,
                "worst_symbol_return": worst_symbol_return,
                "spy_return_same_period": float(spy_ret) if spy_ret is not None else float("nan"),
            }
        )

    return pd.DataFrame(rows)


def _market_context(spy_window: pd.Series) -> dict[str, Any]:
    spy_window = spy_window.dropna()
    if len(spy_window) < 5:
        return {
            "market_return": float("nan"),
            "market_vol": float("nan"),
            "market_drawdown": float("nan"),
            "crosses_sma20": 0,
            "trend_r2": float("nan"),
            "market_state": "unknown",
        }

    rets = spy_window.pct_change().dropna()
    market_return = float(spy_window.iloc[-1] / spy_window.iloc[0] - 1.0)
    market_vol = float(rets.std() * sqrt(252)) if not rets.empty else float("nan")

    cum = (1 + rets).cumprod() if not rets.empty else pd.Series(dtype="float64")
    market_drawdown = float((cum / cum.cummax() - 1.0).min()) if not cum.empty else float("nan")

    sma20 = spy_window.rolling(20, min_periods=20).mean()
    above = (spy_window > sma20).astype(float)
    crosses = int((above.diff().abs() > 0).sum()) if len(above) > 1 else 0

    y = np.log(spy_window / spy_window.iloc[0]).values
    x = np.arange(len(y), dtype=float)
    corr = np.corrcoef(x, y)[0, 1] if len(y) > 2 else 0.0
    trend_r2 = float(corr**2) if np.isfinite(corr) else 0.0

    if trend_r2 >= 0.35 and crosses <= 4:
        market_state = "trending"
    else:
        market_state = "choppy"

    direction = "up" if market_return >= 0 else "down"

    return {
        "market_return": market_return,
        "market_vol": market_vol,
        "market_drawdown": market_drawdown,
        "crosses_sma20": crosses,
        "trend_r2": trend_r2,
        "market_state": f"{market_state}-{direction}",
    }


def _print_performance_section(strategy_m: dict[str, float], spy_m: dict[str, float]) -> None:
    print("\n=== WORST WINDOW PERFORMANCE ===")
    table = pd.DataFrame(
        [
            {
                "asset": "Strategy",
                "window_return": strategy_m["window_return"],
                "sharpe": strategy_m["sharpe"],
                "max_drawdown": strategy_m["max_drawdown"],
                "vol": strategy_m["vol"],
            },
            {
                "asset": "SPY",
                "window_return": spy_m["window_return"],
                "sharpe": spy_m["sharpe"],
                "max_drawdown": spy_m["max_drawdown"],
                "vol": spy_m["vol"],
            },
        ]
    )
    print(table.to_string(index=False))


def _print_summary(
    symbol_freq: pd.DataFrame,
    rebalance_summary: pd.DataFrame,
    market_ctx: dict[str, Any],
) -> None:
    total_slots = int(rebalance_summary["selected_count"].sum()) if not rebalance_summary.empty else 0
    top3_share = 0.0
    if total_slots > 0 and not symbol_freq.empty:
        top3_share = float(symbol_freq.head(3)["times_selected"].sum() / total_slots)

    negative_period_frac = float(
        (rebalance_summary["avg_selected_return_to_next_rebalance"] < 0).mean()
    ) if not rebalance_summary.empty else float("nan")

    avg_turnover = float(rebalance_summary["turnover"].mean()) if not rebalance_summary.empty else float("nan")

    if np.isfinite(avg_turnover) and avg_turnover >= 0.50:
        turnover_text = "high"
    elif np.isfinite(avg_turnover) and avg_turnover >= 0.25:
        turnover_text = "moderate"
    else:
        turnover_text = "low"

    concentration_text = "clustered" if top3_share >= 0.40 else "diversified"

    reversal_text = "yes" if np.isfinite(negative_period_frac) and negative_period_frac > 0.50 else "mixed"

    print("\n=== FINAL SUMMARY ===")
    print(f"Name concentration: portfolio was {concentration_text} (top-3 name share={top3_share:.1%}).")
    print(f"Name reversal evidence: {reversal_text} (negative pick-period ratio={negative_period_frac:.1%}).")
    print(f"Turnover profile: {turnover_text} (avg turnover={avg_turnover:.2f}).")
    print(
        "Market context: "
        f"{market_ctx['market_state']} (SPY return={market_ctx['market_return']:.2%}, "
        f"vol={market_ctx['market_vol']:.2%}, drawdown={market_ctx['market_drawdown']:.2%}, "
        f"SMA20 crosses={market_ctx['crosses_sma20']})."
    )

    likely_cause = (
        "Likely underperformance driver: momentum picks faced frequent reversals during a "
        f"{market_ctx['market_state']} tape while turnover stayed {turnover_text}, "
        "adding trading friction on top of weak pick follow-through."
    )
    print(likely_cause)


def main() -> None:
    cfg = _build_baseline_cfg()
    window_start = pd.Timestamp(WINDOW_START)
    window_end = pd.Timestamp(WINDOW_END)

    symbol_dfs, market_df = _load_data()
    market_df = _ensure_datetime_index(_normalize_cols(market_df))
    spy_close = market_df["close"].copy()

    print("[INFO] Running walk-forward replay with rebalance details...")
    equity_oos, all_rebals = _run_full_oos_with_details(symbol_dfs, market_df, cfg)

    aligned = pd.concat(
        [equity_oos.rename("strategy"), spy_close.rename("spy")],
        axis=1,
        sort=False,
    ).dropna()

    window_df = aligned.loc[window_start:window_end].copy()
    if window_df.empty:
        raise RuntimeError("Target worst window has no aligned strategy/SPY rows.")

    strategy_metrics = _compute_metrics(window_df["strategy"])
    spy_norm = (window_df["spy"] / window_df["spy"].iloc[0]) * float(window_df["strategy"].iloc[0])
    spy_metrics = _compute_metrics(spy_norm)

    in_window_rebals = all_rebals.loc[
        (all_rebals["rebalance_date"] >= window_start)
        & (all_rebals["rebalance_date"] <= window_end)
    ].copy()

    symbol_freq = _build_symbol_frequency(in_window_rebals)
    rebalance_summary = _build_rebalance_summary(
        window_rebals=in_window_rebals,
        all_rebals=all_rebals,
        symbol_dfs=symbol_dfs,
        spy_close=spy_close,
        window_end=window_end,
    )

    market_ctx = _market_context(window_df["spy"])

    OUT_SYMBOL_FREQUENCY.parent.mkdir(parents=True, exist_ok=True)
    symbol_freq.to_csv(OUT_SYMBOL_FREQUENCY, index=False)
    rebalance_summary.to_csv(OUT_REBALANCE_SUMMARY, index=False)

    _print_performance_section(strategy_metrics, spy_metrics)

    print("\n=== SYMBOL FREQUENCY ===")
    if symbol_freq.empty:
        print("No selected symbols found in target window.")
    else:
        print(symbol_freq.head(15).to_string(index=False))

    print("\n=== TURNOVER SUMMARY ===")
    if rebalance_summary.empty:
        print("No rebalances found in target window.")
    else:
        print(
            pd.DataFrame(
                [
                    {
                        "avg_turnover": float(rebalance_summary["turnover"].mean()),
                        "max_turnover": float(rebalance_summary["turnover"].max()),
                        "num_rebalances": int(len(rebalance_summary)),
                    }
                ]
            ).to_string(index=False)
        )

    print("\n=== REBALANCE SNAPSHOTS ===")
    if rebalance_summary.empty:
        print("No snapshots available.")
    else:
        show_cols = [
            "rebalance_date",
            "next_rebalance_date",
            "selected_count",
            "turnover",
            "avg_selected_return_to_next_rebalance",
            "best_symbol",
            "best_symbol_return",
            "worst_symbol",
            "worst_symbol_return",
        ]
        print(rebalance_summary[show_cols].head(8).to_string(index=False))

    print("\n=== MARKET CONTEXT ===")
    print(
        pd.DataFrame(
            [
                {
                    "spy_return": market_ctx["market_return"],
                    "spy_vol": market_ctx["market_vol"],
                    "spy_drawdown": market_ctx["market_drawdown"],
                    "sma20_crosses": market_ctx["crosses_sma20"],
                    "trend_r2": market_ctx["trend_r2"],
                    "market_state": market_ctx["market_state"],
                }
            ]
        ).to_string(index=False)
    )

    print("\n=== OPTIONAL SECTOR CHECK ===")
    print("Sector metadata is not available in the current Stooq cache; sector concentration skipped.")

    _print_summary(symbol_freq=symbol_freq, rebalance_summary=rebalance_summary, market_ctx=market_ctx)

    print(f"\n[INFO] Saved {OUT_SYMBOL_FREQUENCY}")
    print(f"[INFO] Saved {OUT_REBALANCE_SUMMARY}")


if __name__ == "__main__":
    main()
