#!/usr/bin/env python3
"""
preview_current_holdings.py — Read-only preview of what the strategy would hold TODAY.

Uses existing strategy logic (signals, market filter, config) without placing trades
or requiring Alpaca.

Usage:
    python preview_current_holdings.py            # default top-N from config
    python preview_current_holdings.py --top-n 15
    python preview_current_holdings.py --stooq-dir path/to/stooq
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config.live_trading_config import _build_strategy_defaults
from research.data_stooq import list_stooq_parquets, _infer_symbol_from_filename
from research.walk_forward_momentum import (
    WalkForwardConfig,
    dollar_volume_rank,
    momentum_score,
    market_risk_on,
    compute_market_exposure,
)

STOOQ_DIR = Path("data_cache/stooq")
_WFCFG_FIELDS = {f.name for f in dataclasses.fields(WalkForwardConfig)}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw parquet frame to a lowercase-column, date-indexed DataFrame."""
    if "date" in df.columns:
        df = df.set_index(pd.to_datetime(df["date"])).drop(columns=["date"])
    elif "Date" in df.columns:
        df = df.set_index(pd.to_datetime(df["Date"])).drop(columns=["Date"])
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    # Map adj close variants to 'close' when a dedicated close column is absent
    for alias in ("adj close", "adjclose", "adj_close"):
        if alias in df.columns and "close" not in df.columns:
            df = df.rename(columns={alias: "close"})
    return df


def load_symbol_dfs(
    stooq_dir: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Load all per-symbol OHLCV parquets; also return the SPY frame separately."""
    files = list_stooq_parquets(stooq_dir)
    symbol_dfs: dict[str, pd.DataFrame] = {}
    market_df: pd.DataFrame | None = None

    for fp in files:
        sym = _infer_symbol_from_filename(fp)
        if not sym:
            continue
        try:
            df = _normalize_ohlcv(pd.read_parquet(fp))
        except Exception:
            continue
        if "close" not in df.columns:
            continue
        symbol_dfs[sym] = df
        if sym == "SPY":
            market_df = df

    return symbol_dfs, market_df


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def build_config(top_n: int | None = None) -> WalkForwardConfig:
    """Merge public defaults with private overrides, then create a WalkForwardConfig."""
    defaults = _build_strategy_defaults()
    # Keep only keys WalkForwardConfig accepts
    filtered = {k: v for k, v in defaults.items() if k in _WFCFG_FIELDS}
    if top_n is not None:
        filtered["positions"] = top_n
    return WalkForwardConfig(**filtered)


# ---------------------------------------------------------------------------
# Main preview
# ---------------------------------------------------------------------------


def preview(top_n: int | None = None, stooq_dir: Path = STOOQ_DIR) -> None:
    print(f"{'=' * 60}")
    print(f"  Equity Momentum Strategy — Holdings Preview")
    print(f"  Run date : {date.today()}")
    print(f"{'=' * 60}\n")

    cfg = build_config(top_n)
    n = cfg.positions
    print(f"Config  : top_n={n}  weights=({cfg.w_3m:.0%}/{cfg.w_6m:.0%}/{cfg.w_12m:.0%})"
          f"  mom=({cfg.mom_3m}/{cfg.mom_6m}/{cfg.mom_12m} days)")
    print(f"Universe: top {cfg.universe_top_n} by $ volume  |  "
          f"market_sma={cfg.market_sma_days}d\n")

    # ---- Load data --------------------------------------------------------
    print(f"[INFO] Loading price data from '{stooq_dir}' ...")
    symbol_dfs, market_df = load_symbol_dfs(stooq_dir)
    print(f"[INFO] Loaded {len(symbol_dfs)} symbols\n")

    if not symbol_dfs:
        print("[ERROR] No symbol data found. Check --stooq-dir path.")
        return

    # Determine signal date (latest bar available in the dataset)
    latest_dates = [df.index[-1] for df in symbol_dfs.values() if len(df) > 0]
    asof: pd.Timestamp = max(latest_dates)
    print(f"[INFO] Signal date (latest available close): {asof.date()}\n")

    # ---- Market regime (SPY MA200) ----------------------------------------
    if market_df is not None:
        spy_close_series = market_df.loc[:asof, "close"]
        spy_close = float(spy_close_series.iloc[-1])
        spy_sma_series = market_df["close"].rolling(cfg.market_sma_days, min_periods=cfg.market_sma_days).mean()
        spy_sma_valid = spy_sma_series.dropna()
        spy_sma = float(spy_sma_valid.iloc[-1]) if len(spy_sma_valid) > 0 else float("nan")

        risk_on = market_risk_on(market_df, asof, cfg)
        exposure = compute_market_exposure(market_df, asof, cfg, risk_on)

        regime_label = "RISK-ON  ✓" if risk_on else "RISK-OFF ✗"
        print(f"Market regime (SPY vs SMA{cfg.market_sma_days}):")
        print(f"  SPY close  : ${spy_close:,.2f}")
        print(f"  SMA{cfg.market_sma_days:<3}      : ${spy_sma:,.2f}")
        print(f"  Status     : {regime_label}")
        print(f"  Exposure   : {exposure:.0%}\n")
    else:
        print("[WARN] SPY not found in data cache; market filter skipped.\n")
        risk_on = True
        exposure = 1.0

    # ---- Liquidity filter: top-N universe by $ volume ----------------------
    dv = dollar_volume_rank(symbol_dfs, asof, cfg)
    universe = list(dv.head(cfg.universe_top_n).index)
    print(f"[INFO] Universe (top {cfg.universe_top_n} by trailing $ volume): {len(universe)} symbols\n")

    # ---- Score all universe symbols ----------------------------------------
    scored: list[tuple[str, float]] = []
    skipped = 0
    for sym in universe:
        sc = momentum_score(symbol_dfs[sym], asof, cfg)
        if sc is None or not np.isfinite(sc):
            skipped += 1
            continue
        scored.append((sym, float(sc)))

    scored.sort(key=lambda x: x[1], reverse=True)
    print(f"[INFO] Symbols scored: {len(scored)}  |  skipped (insufficient history): {skipped}\n")

    if not scored:
        print("[ERROR] No symbols could be scored. The data cache may need refreshing.")
        return

    # ---- Optional strategy-level filters (mirrors compute_rebalance_target) -
    if cfg.use_strength_filter:
        pre = len(scored)
        scored = [(s, sc) for s, sc in scored if sc > 0.0]
        print(f"[INFO] Strength filter (score > 0): {pre} → {len(scored)} symbols")

    if cfg.percentile_filter_enabled and scored:
        threshold = float(np.quantile([sc for _, sc in scored], cfg.percentile_threshold))
        pre = len(scored)
        scored = [(s, sc) for s, sc in scored if sc >= threshold]
        print(f"[INFO] Percentile filter (≥{cfg.percentile_threshold:.0%}): {pre} → {len(scored)} symbols")

    eligible_count = len(scored)
    print(f"[INFO] Eligible after filters: {eligible_count}\n")

    # ---- Select top N -------------------------------------------------------
    selected_set = {s for s, _ in scored[:n]}

    # ---- Display table (top 2×N or 30, whichever is larger) ----------------
    display_rows = max(n * 2, 30)
    scores_dict = dict(scored)

    col_rank   = 4
    col_sym    = 8
    col_score  = 15
    col_price  = 10
    col_status = 12

    header = (
        f"{'Rank':>{col_rank}}  "
        f"{'Symbol':<{col_sym}}  "
        f"{'Momentum Score':>{col_score}}  "
        f"{'Price':>{col_price}}  "
        f"{'Status':<{col_status}}"
    )
    sep = "-" * len(header)

    print(header)
    print(sep)

    for rank, (sym, sc) in enumerate(scored[:display_rows], start=1):
        latest_close: float = float("nan")
        df = symbol_dfs.get(sym)
        if df is not None:
            hist = df.loc[:asof, "close"]
            if len(hist) > 0:
                latest_close = float(hist.iloc[-1])

        price_str = f"${latest_close:,.2f}" if np.isfinite(latest_close) else "N/A"
        status = "◀ SELECTED" if sym in selected_set else ""

        print(
            f"{rank:>{col_rank}}  "
            f"{sym:<{col_sym}}  "
            f"{sc:>{col_score}.4f}  "
            f"{price_str:>{col_price}}  "
            f"{status:<{col_status}}"
        )

    if len(scored) > display_rows:
        print(f"       ... ({len(scored) - display_rows} more symbols not shown)")

    # ---- Summary ------------------------------------------------------------
    print()
    print(f"{'=' * 60}")
    print(f"  TOP {n} HOLDINGS — Strategy would hold these today")
    print(f"{'=' * 60}")

    selected_symbols = [s for s, _ in scored[:n]]
    for i, sym in enumerate(selected_symbols, 1):
        sc = scores_dict[sym]
        df = symbol_dfs.get(sym)
        latest_close = float("nan")
        if df is not None:
            hist = df.loc[:asof, "close"]
            if len(hist) > 0:
                latest_close = float(hist.iloc[-1])
        price_str = f"${latest_close:,.2f}" if np.isfinite(latest_close) else "N/A"
        print(f"  {i:>2}. {sym:<8}  score={sc:+.4f}  price={price_str}")

    print()
    if not risk_on:
        print(
            f"[RISK-OFF] Market is below its SMA{cfg.market_sma_days}. "
            f"Strategy targets {exposure:.0%} equity exposure."
        )
        print("           These holdings would be sized at reduced weight.\n")

    print(f"Signal date : {asof.date()}  |  Eligible symbols : {eligible_count}  |  Selected : {len(selected_symbols)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview what the momentum strategy would hold today (read-only)."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        metavar="N",
        help="Number of holdings to select (default: read from config).",
    )
    parser.add_argument(
        "--stooq-dir",
        type=Path,
        default=STOOQ_DIR,
        metavar="DIR",
        help=f"Path to stooq parquet cache (default: {STOOQ_DIR}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preview(top_n=args.top_n, stooq_dir=args.stooq_dir)
