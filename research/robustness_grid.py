"""
Robustness grid runner (no optimization).

Purpose:
- Quickly detect fragility: does performance collapse if you nudge parameters?
- Compare Strategy vs SPY over the same date range and cost assumptions.

Assumptions:
- You already have:
  - research.backtest_v0.backtest_rotation_v0(prices, config_path=..., cost_bps=...)
  - research.data_stooq.load_stooq_price_matrix(...)
- Your config file is config/alpha_v1.yaml (reference configuration).

Design:
- We *do not* mutate your YAML on disk.
- We load YAML to a dict, override a few keys in-memory, and pass that dict
  into a "config override" run by writing a temporary config file.

If you'd rather not write temp files, we can patch backtest_rotation_v0 to accept
a cfg dict directly — but this keeps your existing code unchanged.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

from research.backtest_v0 import backtest_rotation_v0
from research.data_stooq import load_stooq_price_matrix


# -----------------------------
# Helpers
# -----------------------------


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _write_temp_yaml(cfg: Dict[str, Any]) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp.flush()
    tmp.close()
    return tmp.name


def _annualized_cagr(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0])
    years = len(equity_curve) / 252.0
    if years <= 0:
        return 0.0
    return total_return ** (1.0 / years) - 1.0


def _annualized_sharpe(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(returns.std())
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float(returns.mean() / std * np.sqrt(252.0))


def _max_drawdown(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max
    return float(dd.min())


def _spy_buy_hold(
    prices: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    """
    prices: wide price matrix with a "SPY" column.
    Returns: (equity_curve, returns, metrics)
    """
    if "SPY" not in prices.columns:
        raise ValueError(
            "SPY column not found in prices matrix. Make sure you have SPY stooq parquet."
        )

    spy = prices["SPY"].dropna().sort_index()
    rets = spy.pct_change().fillna(0.0)
    equity = (1.0 + rets).cumprod()

    metrics = {
        "cagr": _annualized_cagr(equity),
        "sharpe": _annualized_sharpe(rets),
        "max_drawdown": _max_drawdown(equity),
        "final_equity": float(equity.iloc[-1]),
    }
    return equity, rets, metrics


def _align_to_common_dates(
    strategy_equity: pd.Series, spy_equity: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    idx = strategy_equity.index.intersection(spy_equity.index)
    return strategy_equity.loc[idx], spy_equity.loc[idx]


def _fmt_pct(x: float) -> str:
    return f"{x*100:6.2f}%"


# -----------------------------
# Grid Runner
# -----------------------------


def run_robustness_grid(
    stooq_dir: str = "data_cache/stooq",
    base_config_path: str = "config/alpha_v1.yaml",
    cost_bps: float = 10.0,
    ffill_limit: int = 3,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame of results for a small robustness grid.

    Grid (small, high-signal):
      - top_n: 5, 10, 15
      - rebalance_days_interval: 5, 10, 15
      - momentum (lookback, skip): (12,1), (9,1), (12,0)
    """
    base_cfg = _load_yaml(base_config_path)

    # Load price matrix ONCE
    prices = load_stooq_price_matrix(
        stooq_dir=stooq_dir,
        start=start,
        end=end,
    ).sort_index()

    # Optional gap handling
    prices = prices.ffill(limit=ffill_limit)

    # Benchmark (SPY) from same matrix
    spy_equity, spy_rets, spy_m = _spy_buy_hold(prices)

    # Define grid
    top_ns = [5, 10, 15]
    rebals = [5, 10, 15]
    mom_pairs = [(12, 1), (9, 1), (12, 0)]

    rows: List[Dict[str, Any]] = []

    for lookback_m, skip_m in mom_pairs:
        for top_n in top_ns:
            for rebal_days in rebals:
                cfg = copy.deepcopy(base_cfg)

                # Override in-memory (no disk mutation)
                cfg["signal"]["momentum"]["lookback_months"] = int(lookback_m)
                cfg["signal"]["momentum"]["skip_recent_months"] = int(skip_m)
                cfg["portfolio"]["top_n"] = int(top_n)
                cfg["rebalance"]["trading_days_interval"] = int(rebal_days)

                tmp_cfg_path = _write_temp_yaml(cfg)

                out = backtest_rotation_v0(
                    prices=prices.drop(
                        columns=["SPY"], errors="ignore"
                    ),  # strategy universe
                    config_path=tmp_cfg_path,
                    cost_bps=cost_bps,
                )

                strat_eq = out["equity_curve"]
                strat_rets = out["returns"]
                m = out["metrics"]

                # Align with SPY for fair comparison window
                strat_eq_aligned, spy_eq_aligned = _align_to_common_dates(
                    strat_eq, spy_equity
                )

                # Recompute strategy metrics on aligned window (so delisting/NaNs don't accidentally change window)
                strat_rets_aligned = strat_eq_aligned.pct_change().fillna(0.0)

                strat_metrics = {
                    "cagr": _annualized_cagr(strat_eq_aligned),
                    "sharpe": _annualized_sharpe(strat_rets_aligned),
                    "max_drawdown": _max_drawdown(strat_eq_aligned),
                    "final_equity": float(strat_eq_aligned.iloc[-1]),
                }

                rows.append(
                    {
                        "lookback_m": lookback_m,
                        "skip_m": skip_m,
                        "top_n": top_n,
                        "rebalance_days": rebal_days,
                        "cost_bps": cost_bps,
                        "strat_cagr": strat_metrics["cagr"],
                        "strat_sharpe": strat_metrics["sharpe"],
                        "strat_maxdd": strat_metrics["max_drawdown"],
                        "strat_final": strat_metrics["final_equity"],
                        "spy_cagr": spy_m["cagr"],
                        "spy_sharpe": spy_m["sharpe"],
                        "spy_maxdd": spy_m["max_drawdown"],
                        "spy_final": spy_m["final_equity"],
                        "cagr_minus_spy": strat_metrics["cagr"] - spy_m["cagr"],
                        "sharpe_minus_spy": strat_metrics["sharpe"] - spy_m["sharpe"],
                    }
                )

    df = pd.DataFrame(rows)

    # Helpful ordering: best excess CAGR first
    df = df.sort_values(
        ["cagr_minus_spy", "strat_sharpe"], ascending=[False, False]
    ).reset_index(drop=True)
    return df


def print_robustness_report(df: pd.DataFrame, top_k: int = 12) -> None:
    """
    Prints a compact view + a fragility hint.
    """
    if df.empty:
        print("No results.")
        return

    print("=" * 72)
    print("ROBUSTNESS GRID (Strategy vs SPY) — top configs by (CAGR - SPY)")
    print("=" * 72)

    view = df.head(top_k).copy()
    view["strat_cagr"] = view["strat_cagr"].map(_fmt_pct)
    view["spy_cagr"] = view["spy_cagr"].map(_fmt_pct)
    view["cagr_minus_spy"] = view["cagr_minus_spy"].map(_fmt_pct)
    view["strat_maxdd"] = view["strat_maxdd"].map(_fmt_pct)
    view["spy_maxdd"] = view["spy_maxdd"].map(_fmt_pct)

    cols = [
        "lookback_m",
        "skip_m",
        "top_n",
        "rebalance_days",
        "strat_cagr",
        "spy_cagr",
        "cagr_minus_spy",
        "strat_sharpe",
        "spy_sharpe",
        "strat_maxdd",
        "spy_maxdd",
    ]
    print(view[cols].to_string(index=False))

    # Fragility quick check: how many configs beat SPY by > 0?
    beat = (df["cagr_minus_spy"] > 0).mean()
    strong = (df["cagr_minus_spy"] > 0.03).mean()  # >3% CAGR excess
    print("\n" + "-" * 72)
    print(f"Configs beating SPY (excess CAGR > 0):   {beat*100:.1f}%")
    print(f"Configs beating SPY by > 3% CAGR:        {strong*100:.1f}%")
    print("-" * 72)
