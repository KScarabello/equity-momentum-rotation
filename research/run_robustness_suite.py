#!/usr/bin/env python3
"""
Robustness suite for the equity momentum strategy.

Tests small perturbations around the reference configuration to determine whether
the strategy is stable, moderately sensitive, or fragile.

Groups:
  1. Position count perturbation
  2. Rebalance frequency perturbation
  3. Momentum weight perturbation
  4. Risk-off exposure perturbation
  5. SMA length perturbation (slope filter on)

Run:
    python -m research.run_robustness_suite
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from research.walk_forward_momentum import WalkForwardConfig, walk_forward_validate
from research.data_stooq import list_stooq_parquets, _infer_symbol_from_filename

STOOQ_DIR = Path("data_cache/stooq")

# ---------------------------------------------------------------------------
# Validated baseline (center of all perturbations)
# ---------------------------------------------------------------------------
BASELINE = dict(
    positions=12,
    rebalance_interval_weeks=2,
    cost_bps=5.0,
    slippage_bps=2.0,
    min_rebalance_weight_change=0.0,
    w_3m=0.60,
    w_6m=0.30,
    w_12m=0.10,
    market_sma_days=200,
    risk_on_buffer=0.0,
    min_exposure=0.25,
    max_exposure=1.0,
    exposure_slope=0.0,
    stability_lookback_periods=1,
    require_positive_sma_slope=True,
    sma_slope_lookback=20,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _run(
    symbol_dfs: dict,
    market_df: pd.DataFrame,
    label: str,
    overrides: dict,
) -> dict:
    cfg = WalkForwardConfig(**{**BASELINE, **overrides})
    results_df, equity_oos = walk_forward_validate(
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        cfg=cfg,
    )
    rets = equity_oos.pct_change().dropna()
    years = len(equity_oos) / 252.0
    cagr = float((equity_oos.iloc[-1] / equity_oos.iloc[0]) ** (1.0 / max(years, 1e-9))) - 1.0
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252.0)) if rets.std() > 0 else 0.0
    max_dd = float((equity_oos / equity_oos.cummax() - 1.0).min())
    avg_turnover = float(results_df["avg_turnover"].mean()) if "avg_turnover" in results_df.columns else float("nan")
    total_costs = float(results_df["total_costs"].sum()) if "total_costs" in results_df.columns else float("nan")
    return {
        "label": label,
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "avg_turnover_pct": round(avg_turnover * 100, 2),
        "total_costs": round(total_costs, 2),
        "final_equity": round(float(equity_oos.iloc[-1]), 2),
    }


def _print_group(title: str, rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    print(f"\n=== ROBUSTNESS: {title} ===")
    print(df.to_string(index=False))
    return df


def _stability_verdict(df: pd.DataFrame, baseline_label: str) -> str:
    """
    Simple heuristic: compare Sharpe of non-baseline configs to baseline.
    - All within 0.05 → stable
    - Any deviation > 0.15 → fragile
    - Otherwise → moderately sensitive
    """
    baseline_sharpe = df.loc[df["label"] == baseline_label, "sharpe"].values
    if len(baseline_sharpe) == 0:
        baseline_sharpe = df["sharpe"].max()
    else:
        baseline_sharpe = baseline_sharpe[0]

    others = df.loc[df["label"] != baseline_label, "sharpe"]
    if len(others) == 0:
        return "stable (single config)"
    max_deviation = float((others - baseline_sharpe).abs().max())
    if max_deviation <= 0.05:
        return f"STABLE (max Sharpe deviation: {max_deviation:.3f})"
    elif max_deviation <= 0.15:
        return f"MODERATELY SENSITIVE (max Sharpe deviation: {max_deviation:.3f})"
    else:
        return f"FRAGILE (max Sharpe deviation: {max_deviation:.3f})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("[INFO] Loading data (shared across all groups)...")
    symbol_dfs, market_df = load_data()

    verdicts: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Group 1: Position count
    # ------------------------------------------------------------------
    print("\n[INFO] Group 1: Position count perturbation...")
    rows = []
    for pos in [10, 12, 14]:
        label = f"positions_{pos}"
        is_baseline = pos == 12
        print(f"  Running {label}{'  <- baseline' if is_baseline else ''}...")
        rows.append(_run(symbol_dfs, market_df, label, {"positions": pos}))
    df1 = _print_group("POSITION COUNT", rows)
    verdicts.append(("Position count", _stability_verdict(df1, "positions_12")))

    # ------------------------------------------------------------------
    # Group 2: Rebalance frequency
    # ------------------------------------------------------------------
    print("\n[INFO] Group 2: Rebalance frequency perturbation...")
    rows = []
    for weeks in [1, 2, 3]:
        label = f"rebal_{weeks}w"
        is_baseline = weeks == 2
        print(f"  Running {label}{'  <- baseline' if is_baseline else ''}...")
        rows.append(_run(symbol_dfs, market_df, label, {"rebalance_interval_weeks": weeks}))
    df2 = _print_group("REBALANCE FREQUENCY", rows)
    verdicts.append(("Rebalance frequency", _stability_verdict(df2, "rebal_2w")))

    # ------------------------------------------------------------------
    # Group 3: Momentum weights
    # ------------------------------------------------------------------
    print("\n[INFO] Group 3: Momentum weight perturbation...")
    weight_configs = [
        {"label": "baseline",  "w_3m": 0.60, "w_6m": 0.30, "w_12m": 0.10},
        {"label": "nearby_a",  "w_3m": 0.55, "w_6m": 0.30, "w_12m": 0.15},
        {"label": "nearby_b",  "w_3m": 0.65, "w_6m": 0.25, "w_12m": 0.10},
    ]
    rows = []
    for wc in weight_configs:
        is_baseline = wc["label"] == "baseline"
        print(f"  Running weights_{wc['label']}{'  <- baseline' if is_baseline else ''}...")
        rows.append(_run(
            symbol_dfs, market_df,
            f"weights_{wc['label']}",
            {"w_3m": wc["w_3m"], "w_6m": wc["w_6m"], "w_12m": wc["w_12m"]},
        ))
    df3 = _print_group("MOMENTUM WEIGHTS", rows)
    verdicts.append(("Momentum weights", _stability_verdict(df3, "weights_baseline")))

    # ------------------------------------------------------------------
    # Group 4: Risk-off exposure
    # ------------------------------------------------------------------
    print("\n[INFO] Group 4: Risk-off exposure perturbation...")
    rows = []
    for exposure in [0.20, 0.25, 0.30]:
        label = f"exposure_{int(exposure * 100)}pct"
        is_baseline = exposure == 0.25
        print(f"  Running {label}{'  <- baseline' if is_baseline else ''}...")
        rows.append(_run(symbol_dfs, market_df, label, {"min_exposure": exposure}))
    df4 = _print_group("RISK-OFF EXPOSURE", rows)
    verdicts.append(("Risk-off exposure", _stability_verdict(df4, "exposure_25pct")))

    # ------------------------------------------------------------------
    # Group 5: SMA length (slope filter on)
    # ------------------------------------------------------------------
    print("\n[INFO] Group 5: SMA length perturbation (slope filter on)...")
    rows = []
    for sma_days in [175, 200, 225]:
        label = f"sma_{sma_days}"
        is_baseline = sma_days == 200
        print(f"  Running {label}{'  <- baseline' if is_baseline else ''}...")
        rows.append(_run(symbol_dfs, market_df, label, {"market_sma_days": sma_days}))
    df5 = _print_group("SMA LENGTH", rows)
    verdicts.append(("SMA length", _stability_verdict(df5, "sma_200")))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY")
    print("=" * 60)
    print(f"{'Group':<25}  Verdict")
    print("-" * 60)
    for group_name, verdict in verdicts:
        print(f"  {group_name:<23}  {verdict}")

    all_stable = all("STABLE" in v and "MODERATELY" not in v for _, v in verdicts)
    any_fragile = any("FRAGILE" in v for _, v in verdicts)
    any_moderate = any("MODERATELY" in v for _, v in verdicts)

    print()
    if all_stable:
        print("OVERALL: Strategy appears STABLE across all tested perturbation groups.")
    elif any_fragile:
        print("OVERALL: Strategy shows FRAGILE behavior in one or more groups — review before live deployment.")
    else:
        print("OVERALL: Strategy is MODERATELY SENSITIVE — performs well near baseline but degrades with some perturbations.")


if __name__ == "__main__":
    main()
