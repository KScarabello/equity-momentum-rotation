#!/usr/bin/env python3
"""
Live-style day-by-day execution simulation for the equity momentum strategy.

This runner intentionally reuses existing strategy logic (signals, market filter,
rebalance cadence, and execution friction assumptions) while simulating portfolio
state forward one trading day at a time.

Run:
    python -m research.run_live_simulation
"""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np

from research.data_stooq import list_stooq_parquets, _infer_symbol_from_filename
from research.walk_forward_momentum import (
    WalkForwardConfig,
    _normalize_cols,
    _ensure_datetime_index,
    _get_close,
    _clip_to_index,
    _week_rebalance_dates,
    dollar_volume_rank,
    momentum_score,
    market_risk_on,
    compute_market_exposure,
)

STOOQ_DIR = Path("data_cache/stooq")

# Example configuration defaults.
# Include both legacy risk_off_exposure and the current exposure controls so
# behavior is explicit and aligned with the example setup.
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

# Optional reconciliation window for apples-to-apples comparison with walk-forward OOS.
# Set to None to preserve full-period behavior.
SIM_START = "2022-11-05"
SIM_END = "2025-12-24"


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    for a in ["adj close", "adjclose", "adj_close"]:
        if a in df.columns and "close" not in df.columns:
            df["close"] = df[a]
    required = ["close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Have: {list(df.columns)}")
    return df.copy()


def fetch_ohlcv(symbol: str, stooq_dir: Path = STOOQ_DIR) -> pd.DataFrame:
    fp = stooq_dir / f"{symbol}.US.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing parquet for {symbol}: {fp}")
    return _normalize_ohlcv(pd.read_parquet(fp))


def load_data(market_symbol: str = "SPY") -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    print("[INFO] Loading universe from parquet cache")
    files = list_stooq_parquets(STOOQ_DIR)
    symbols: list[str] = []
    for fp in files:
        sym = _infer_symbol_from_filename(fp)
        if sym and sym not in symbols:
            symbols.append(sym)
    print(f"[INFO] Universe loaded from cache: {len(symbols)}")

    print("[INFO] Loading market proxy OHLCV")
    market_df = fetch_ohlcv(market_symbol)

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


def _compute_start_end(symbol_dfs: dict[str, pd.DataFrame], market_df: pd.DataFrame, cfg: WalkForwardConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    # Match the same minimum-history discipline used in walk_forward_validate.
    min_need = max(cfg.liq_lookback, cfg.mom_12m + 1, cfg.market_sma_days + 1)
    start_candidates = [
        df.index[min_need] for df in symbol_dfs.values() if len(df.index) > min_need
    ]
    if not start_candidates:
        raise ValueError("Not enough symbol history for configured lookbacks.")
    if len(market_df.index) <= min_need:
        raise ValueError("Not enough market history for configured lookbacks.")

    start = max(max(start_candidates), market_df.index[min_need])
    end = market_df.index.max()
    return start, end


def _build_rebalance_dates(
    cal: pd.DatetimeIndex,
    weekday: int,
    interval_weeks: int,
    rebalance_reset_dates: set[pd.Timestamp] | None = None,
) -> pd.DatetimeIndex:
    """
    Build rebalance dates with optional cadence anchor resets.

    Default behavior (no reset dates): global weekday schedule sliced by interval.
    With reset dates: cadence restarts from each reset segment's first eligible date,
    matching walk-forward per-window anchoring.
    """
    if len(cal) == 0:
        return pd.DatetimeIndex([])

    if not rebalance_reset_dates:
        rebals = _week_rebalance_dates(cal, weekday)
        if interval_weeks > 1:
            rebals = rebals[::interval_weeks]
        if len(rebals) == 0:
            rebals = pd.DatetimeIndex([cal[0]])
        return rebals

    # Convert reset anchors to first trading day on/after each reset date.
    anchors = [pd.Timestamp(cal[0])]
    sorted_resets = sorted({pd.Timestamp(d) for d in rebalance_reset_dates})
    for d in sorted_resets:
        seg = cal[cal >= d]
        if len(seg) == 0:
            continue
        anchors.append(pd.Timestamp(seg[0]))

    anchors = sorted(set(anchors))

    all_rebals: list[pd.Timestamp] = []
    for i, seg_start in enumerate(anchors):
        seg_end = anchors[i + 1] - pd.Timedelta(days=1) if i + 1 < len(anchors) else cal[-1]
        seg_cal = cal[(cal >= seg_start) & (cal <= seg_end)]
        if len(seg_cal) == 0:
            continue
        seg_rebals = _week_rebalance_dates(seg_cal, weekday)
        if interval_weeks > 1:
            seg_rebals = seg_rebals[::interval_weeks]
        if len(seg_rebals) == 0:
            seg_rebals = pd.DatetimeIndex([seg_cal[0]])
        all_rebals.extend(list(seg_rebals))

    return pd.DatetimeIndex(sorted(set(all_rebals)))


def run_live_simulation(
    symbol_dfs: dict[str, pd.DataFrame],
    market_df: pd.DataFrame,
    cfg: WalkForwardConfig,
    sim_start: str | None = None,
    sim_end: str | None = None,
    snapshot_dates: set[pd.Timestamp] | None = None,
    rebalance_reset_dates: set[pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    symbol_dfs = {s: _ensure_datetime_index(_normalize_cols(df)) for s, df in symbol_dfs.items()}
    market_df = _ensure_datetime_index(_normalize_cols(market_df))

    start, end = _compute_start_end(symbol_dfs, market_df, cfg)

    if sim_start is not None:
        start = max(start, pd.Timestamp(sim_start))
    if sim_end is not None:
        end = min(end, pd.Timestamp(sim_end))
    if start > end:
        raise ValueError(
            f"Invalid simulation range after clipping: start={start.date()} > end={end.date()}"
        )

    cal = market_df.index[(market_df.index >= start) & (market_df.index <= end)]
    cal = pd.DatetimeIndex(cal).sort_values()
    if len(cal) == 0:
        raise ValueError("No trading dates for simulation period.")

    rebals = _build_rebalance_dates(
        cal=cal,
        weekday=cfg.rebalance_weekday,
        interval_weeks=cfg.rebalance_interval_weeks,
        rebalance_reset_dates=rebalance_reset_dates,
    )

    closes = {s: _get_close(df) for s, df in symbol_dfs.items()}

    cash = float(cfg.starting_cash)
    holdings: dict[str, float] = {}

    equity_rows: list[dict] = []
    trade_rows: list[dict] = []
    diag_rows: list[dict] = []

    total_trade_count = 0
    number_of_rebalances = 0
    total_costs = 0.0
    turnover_per_rebalance: list[float] = []

    ranking_history: list[set[str]] = []
    state_snapshots: list[dict] = []

    def get_px(sym: str, dt: pd.Timestamp) -> float:
        cs = closes.get(sym)
        if cs is None:
            return float("nan")
        h = cs.loc[:dt]
        if len(h) == 0:
            return float("nan")
        return float(h.iloc[-1])

    def mark_to_market(dt: pd.Timestamp) -> tuple[float, float]:
        invested = 0.0
        for sym, sh in holdings.items():
            px = get_px(sym, dt)
            if np.isfinite(px):
                invested += sh * px
        equity = float(cash + invested)
        return equity, float(invested)

    def liquid_universe(asof: pd.Timestamp) -> list[str]:
        dv = dollar_volume_rank(symbol_dfs, asof, cfg)
        return list(dv.head(cfg.universe_top_n).index)

    def top3_holdings_str(dt: pd.Timestamp) -> str:
        vals = []
        for sym, sh in holdings.items():
            px = get_px(sym, dt)
            if np.isfinite(px):
                vals.append((sym, abs(float(sh) * px), float(sh)))
        vals.sort(key=lambda x: x[1], reverse=True)
        return "|".join([f"{sym}:{sh:.6f}" for sym, _, sh in vals[:3]])

    for dt in cal:
        if dt in rebals:
            number_of_rebalances += 1
            pv_before, invested_before = mark_to_market(dt)

            risk_on = market_risk_on(market_df, dt, cfg)
            target_exposure = compute_market_exposure(market_df, dt, cfg, risk_on)

            m_hist = _clip_to_index(market_df, market_df.index.min(), dt)
            m_close = float(_get_close(m_hist).iloc[-1])
            m_sma = float(_get_close(m_hist).rolling(cfg.market_sma_days, min_periods=cfg.market_sma_days).mean().iloc[-1])
            sma_slope_ok = True
            if cfg.require_positive_sma_slope:
                sma_series = _get_close(m_hist).rolling(cfg.market_sma_days, min_periods=cfg.market_sma_days).mean().dropna()
                if len(sma_series) <= cfg.sma_slope_lookback:
                    sma_slope_ok = False
                else:
                    sma_slope_ok = float(sma_series.iloc[-1]) > float(sma_series.iloc[-1 - cfg.sma_slope_lookback])

            uni = liquid_universe(dt)
            scored: list[tuple[str, float]] = []
            for sym in uni:
                sc = momentum_score(symbol_dfs[sym], dt, cfg)
                if sc is not None and np.isfinite(sc):
                    scored.append((sym, float(sc)))
            scored.sort(key=lambda x: x[1], reverse=True)

            top_pool_size = cfg.positions * 2
            top_pool = {s for s, _ in scored[:top_pool_size]}
            ranking_history.append(top_pool)
            if len(ranking_history) > cfg.stability_lookback_periods:
                ranking_history.pop(0)

            if cfg.stability_lookback_periods > 1 and len(ranking_history) >= cfg.stability_lookback_periods:
                stable_scored = [(s, sc) for s, sc in scored if all(s in hist for hist in ranking_history)]
            else:
                stable_scored = scored

            picks = [s for s, _ in stable_scored[: cfg.positions]]
            if len(picks) < cfg.positions:
                picks_set = set(picks)
                for s, _ in scored:
                    if len(picks) >= cfg.positions:
                        break
                    if s not in picks_set:
                        picks.append(s)
                        picks_set.add(s)

            target_weights = {s: (1.0 / len(picks)) for s in picks} if picks else {}
            target_values = {s: pv_before * target_exposure * w for s, w in target_weights.items()}

            current_values: dict[str, float] = {}
            for sym, sh in holdings.items():
                px = get_px(sym, dt)
                if np.isfinite(px):
                    current_values[sym] = sh * px

            rebalance_notional = 0.0

            # Full exits first.
            for sym in list(holdings.keys()):
                if sym in target_values:
                    continue
                cur_val = current_values.get(sym, 0.0)
                if abs(cur_val) <= 0.0:
                    del holdings[sym]
                    continue

                px = get_px(sym, dt)
                if not np.isfinite(px) or px <= 0:
                    continue

                old_sh = holdings.get(sym, 0.0)
                dsh = -old_sh
                traded_notional = abs(cur_val)

                del holdings[sym]

                trade_rows.append(
                    {
                        "rebalance_date": dt,
                        "symbol": sym,
                        "action": "SELL",
                        "shares_change": float(dsh),
                        "price_used": float(px),
                        "traded_notional": float(traded_notional),
                        "estimated_cost": float(traded_notional * (cfg.cost_bps / 10_000.0)),
                        "estimated_slippage": float(traded_notional * (cfg.slippage_bps / 10_000.0)),
                        "target_weight": 0.0,
                    }
                )

                rebalance_notional += traded_notional
                total_trade_count += 1

            # Resizes and adds.
            for sym, tval in target_values.items():
                px = get_px(sym, dt)
                if not np.isfinite(px) or px <= 0:
                    continue

                cur_val = current_values.get(sym, 0.0)
                diff = float(tval - cur_val)

                is_resize_trade = sym in current_values
                if is_resize_trade:
                    current_weight = (cur_val / pv_before) if pv_before > 0 else 0.0
                    target_weight = target_weights.get(sym, 0.0)
                    weight_diff = abs(target_weight - current_weight)
                    if weight_diff < cfg.min_rebalance_weight_change:
                        continue

                if abs(diff) / max(pv_before, 1e-9) < 1e-4:
                    continue

                dsh = float(diff / px)
                old_sh = holdings.get(sym, 0.0)
                new_sh = old_sh + dsh
                holdings[sym] = new_sh

                action = "BUY" if old_sh == 0.0 and dsh > 0 else "RESIZE"
                traded_notional = abs(diff)

                trade_rows.append(
                    {
                        "rebalance_date": dt,
                        "symbol": sym,
                        "action": action,
                        "shares_change": float(dsh),
                        "price_used": float(px),
                        "traded_notional": float(traded_notional),
                        "estimated_cost": float(traded_notional * (cfg.cost_bps / 10_000.0)),
                        "estimated_slippage": float(traded_notional * (cfg.slippage_bps / 10_000.0)),
                        "target_weight": float(target_weights.get(sym, 0.0) * target_exposure),
                    }
                )

                rebalance_notional += traded_notional
                total_trade_count += 1

            # Apply friction and recompute cash from current holdings at rebalance close.
            est_cost = rebalance_notional * (cfg.cost_bps / 10_000.0)
            est_slippage = rebalance_notional * (cfg.slippage_bps / 10_000.0)
            total_costs += est_cost + est_slippage

            holdings_value_after = 0.0
            for sym, sh in holdings.items():
                px = get_px(sym, dt)
                if np.isfinite(px):
                    holdings_value_after += sh * px

            cash = pv_before - holdings_value_after - est_cost - est_slippage

            pv_after, _ = mark_to_market(dt)

            # Fill resulting weight post-trade for this rebalance date.
            if pv_after > 0 and trade_rows:
                for row in trade_rows:
                    if row["rebalance_date"] != dt:
                        continue
                    sym = row["symbol"]
                    sh = holdings.get(sym, 0.0)
                    px = get_px(sym, dt)
                    if np.isfinite(px):
                        row["resulting_weight"] = float((sh * px) / pv_after)
                    else:
                        row["resulting_weight"] = float("nan")

            turnover = rebalance_notional / pv_before if pv_before > 0 else 0.0
            turnover_per_rebalance.append(float(turnover))

            diag_rows.append(
                {
                    "rebalance_date": dt,
                    "risk_on": bool(risk_on),
                    "target_exposure_pct": float(target_exposure * 100.0),
                    "market_close": float(m_close),
                    "market_sma": float(m_sma) if np.isfinite(m_sma) else float("nan"),
                    "sma_slope_ok": bool(sma_slope_ok),
                    "pick_count": int(len(picks)),
                    "chosen_symbols": "|".join(picks),
                    "equity_before": float(pv_before),
                    "invested_before": float(invested_before),
                    "rebalance_turnover_pct": float(turnover * 100.0),
                    "rebalance_trade_count": int(sum(1 for r in trade_rows if r["rebalance_date"] == dt)),
                }
            )

        equity, invested = mark_to_market(dt)
        exposure = (invested / equity) if equity > 0 else 0.0
        equity_rows.append(
            {
                "date": dt,
                "equity": float(equity),
                "cash": float(cash),
                "invested_value": float(invested),
                "exposure_pct": float(exposure * 100.0),
            }
        )

        if snapshot_dates is not None and dt in snapshot_dates:
            state_snapshots.append(
                {
                    "date": dt,
                    "equity": float(equity),
                    "cash": float(cash),
                    "invested_value": float(invested),
                    "holdings_count": int(len(holdings)),
                    "top3_holdings": top3_holdings_str(dt),
                }
            )

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trade_rows)
    diagnostics_df = pd.DataFrame(diag_rows)

    eq_series = pd.Series(equity_df["equity"].values, index=pd.DatetimeIndex(equity_df["date"]))
    rets = eq_series.pct_change().dropna()
    years = len(eq_series) / 252.0
    cagr = (eq_series.iloc[-1] / eq_series.iloc[0]) ** (1.0 / max(years, 1e-9)) - 1.0
    sharpe = (rets.mean() / rets.std() * np.sqrt(252.0)) if rets.std() > 0 else 0.0
    max_dd = (eq_series / eq_series.cummax() - 1.0).min()

    avg_turnover = float(np.mean(turnover_per_rebalance)) if turnover_per_rebalance else 0.0

    summary = {
        "final_equity": float(eq_series.iloc[-1]),
        "cagr_pct": float(cagr * 100.0),
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(max_dd * 100.0),
        "avg_turnover_pct": float(avg_turnover * 100.0),
        "total_costs": float(total_costs),
        "number_of_rebalances": int(number_of_rebalances),
        "total_trade_count": int(total_trade_count),
        "start_date": str(cal.min().date()),
        "end_date": str(cal.max().date()),
        "state_snapshots": state_snapshots,
    }

    return equity_df, trades_df, diagnostics_df, summary


def save_outputs(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    summary: dict,
    out_dir: Path,
    file_suffix: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    equity_fp = out_dir / f"live_sim_equity{file_suffix}.csv"
    trades_fp = out_dir / f"live_sim_trades{file_suffix}.csv"
    diag_fp = out_dir / f"live_sim_rebalance_diagnostics{file_suffix}.csv"
    summary_fp = out_dir / f"live_sim_summary{file_suffix}.json"

    equity_df.to_csv(equity_fp, index=False)
    trades_df.to_csv(trades_fp, index=False)
    diagnostics_df.to_csv(diag_fp, index=False)
    with summary_fp.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Wrote {equity_fp}")
    print(f"[INFO] Wrote {trades_fp}")
    print(f"[INFO] Wrote {diag_fp}")
    print(f"[INFO] Wrote {summary_fp}")


def print_summary(summary: dict) -> None:
    print("\n=== LIVE SIMULATION SUMMARY ===")
    print(f"final_equity:         {summary['final_equity']:.2f}")
    print(f"cagr_pct:             {summary['cagr_pct']:.2f}")
    print(f"sharpe:               {summary['sharpe']:.3f}")
    print(f"max_drawdown_pct:     {summary['max_drawdown_pct']:.2f}")
    print(f"avg_turnover_pct:     {summary['avg_turnover_pct']:.2f}")
    print(f"total_costs:          {summary['total_costs']:.2f}")
    print(f"number_of_rebalances: {summary['number_of_rebalances']}")
    print(f"total_trade_count:    {summary['total_trade_count']}")
    print(f"start_date:           {summary['start_date']}")
    print(f"end_date:             {summary['end_date']}")


def print_reconciliation_notes(summary: dict) -> None:
    print("\n=== RECONCILIATION NOTES ===")
    print("This live simulation was run over:")
    print(f"start_date: {summary['start_date']}")
    print(f"end_date:   {summary['end_date']}")
    print()
    print("Compare against walk-forward OOS results over the same range.")
    print("Key metrics to compare:")
    print("- final_equity")
    print("- cagr_pct")
    print("- sharpe")
    print("- max_drawdown_pct")
    print("- avg_turnover_pct")
    print("- total_costs")


def main() -> None:
    cfg = WalkForwardConfig(**BASE_CFG)

    print("[INFO] Loading data")
    symbol_dfs, market_df = load_data(market_symbol=cfg.market_symbol)

    print("[INFO] Running day-by-day live simulation...")
    equity_df, trades_df, diagnostics_df, summary = run_live_simulation(
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=cfg,
        sim_start=SIM_START,
        sim_end=SIM_END,
    )

    file_suffix = "_oos" if (SIM_START is not None or SIM_END is not None) else ""

    save_outputs(
        equity_df=equity_df,
        trades_df=trades_df,
        diagnostics_df=diagnostics_df,
        summary=summary,
        out_dir=Path("."),
        file_suffix=file_suffix,
    )

    print_summary(summary)
    print_reconciliation_notes(summary)


if __name__ == "__main__":
    main()
