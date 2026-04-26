"""
Minimal equity momentum rotation backtest.

Research sanity-check, not production code.
- Equal weight across selected holdings
- Rebalance every N trading days (from config)
- Momentum = multi-period style (from research.momentum helpers)
- Optional absolute momentum filter (from config)
- Optional simple transaction cost model applied on rebalance days
- Rebalance happens at CLOSE; new holdings apply NEXT trading day (more realistic)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from research.momentum import (
    compute_12_1_momentum,
    pick_top_n,
    apply_absolute_momentum_filter,
    load_config,
)


# ============================================================
# Helper Functions
# ============================================================


def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return (equity_curve - running_max) / running_max


def annualized_cagr(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2:
        return 0.0
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0])
    years = len(equity_curve) / 252.0
    if years <= 0:
        return 0.0
    return total_return ** (1.0 / years) - 1.0


def annualized_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - (risk_free_rate / 252.0)
    std = float(excess.std())
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(252.0))


def turnover_symmetric(prev_holdings: List[str], curr_holdings: List[str]) -> float:
    """
    Turnover = |prev Δ curr| / max(len(prev), len(curr))
    (counts BOTH buys and sells in a swap)
    """
    prev = set(prev_holdings)
    curr = set(curr_holdings)

    if len(prev) == 0 and len(curr) == 0:
        return 0.0
    if len(prev) == 0 or len(curr) == 0:
        return 1.0

    changes = len(prev.symmetric_difference(curr))
    max_positions = max(len(prev), len(curr))
    return changes / max_positions if max_positions > 0 else 0.0


def _safe_equal_weight_return(
    prev_row: pd.Series, curr_row: pd.Series, holdings: List[str]
) -> float:
    if not holdings:
        return 0.0

    prev_prices = prev_row[holdings]
    curr_prices = curr_row[holdings]

    valid = (~prev_prices.isna()) & (~curr_prices.isna())
    if valid.sum() == 0:
        return 0.0

    prev_prices = prev_prices[valid]
    curr_prices = curr_prices[valid]

    weight = 1.0 / len(prev_prices)
    constituent_returns = (curr_prices / prev_prices) - 1.0
    return float((constituent_returns * weight).sum())


# ============================================================
# Main Backtest Function
# ============================================================


def backtest_rotation_v0(
    prices: pd.DataFrame,
    config_path: str = "config/alpha_v1.yaml",
    cost_bps: float = 0.0,
) -> Dict[str, Any]:
    """
    Rebalance at close; new holdings apply next trading day.
    Transaction costs applied at rebalance close.

    cost_bps: basis points per rebalance, multiplied by turnover_symmetric(...)
              Example: cost_bps=10 => 0.10% * turnover
    """
    cfg = load_config(config_path)

    rebalance_interval = int(cfg["rebalance"]["trading_days_interval"])
    top_n = int(cfg["portfolio"]["top_n"])
    min_price = float(cfg["universe"]["filters"]["min_price"])

    abs_filter = cfg["signal"]["momentum"].get("absolute_momentum_filter", {})
    abs_enabled = bool(abs_filter.get("enabled", False))
    min_momentum = float(abs_filter.get("min_momentum", 0.0))

    lookback_months = int(cfg["signal"]["momentum"]["lookback_months"])
    skip_recent_months = int(cfg["signal"]["momentum"]["skip_recent_months"])

    lookback_days = lookback_months * 21
    skip_days = skip_recent_months * 21
    min_history = lookback_days + skip_days

    prices = prices.sort_index()
    dates = prices.index
    n_days = len(dates)

    cost_rate = float(cost_bps) / 10000.0

    holdings: List[str] = []
    equity = 1.0

    equity_curve: List[float] = []
    returns_series: List[float] = []

    holdings_history: List[Tuple[pd.Timestamp, List[str]]] = []
    turnover_events: List[float] = []

    # NOTE: We decide trades at close on rebalance day, so schedule based on index i (close).
    def is_rebalance_idx(i: int) -> bool:
        return (i >= min_history) and ((i - min_history) % rebalance_interval == 0)

    # Day 0: no return, and we do NOT rebalance at day 0 (no history anyway)
    equity_curve.append(equity)
    returns_series.append(0.0)

    for i in range(1, n_days):
        # 1) Realize return for day i based on holdings held from prior close
        daily_return = _safe_equal_weight_return(
            prices.iloc[i - 1], prices.iloc[i], holdings
        )
        equity *= 1.0 + daily_return

        # 2) Rebalance at CLOSE of day i (so changes apply from i+1)
        if is_rebalance_idx(i):
            historical_prices = prices.iloc[: i + 1]

            try:
                momentum = compute_12_1_momentum(
                    historical_prices,
                    lookback_months=lookback_months,
                    skip_recent_months=skip_recent_months,
                    min_price=min_price,
                )
            except ValueError:
                momentum = pd.Series(dtype=float)

            if not momentum.empty:
                top_picks = pick_top_n(momentum, top_n)
                top_picks = apply_absolute_momentum_filter(
                    momentum, top_picks, abs_enabled, min_momentum
                )
            else:
                top_picks = []

            t = turnover_symmetric(holdings, top_picks)
            turnover_events.append(t)

            if cost_rate > 0.0 and t > 0.0:
                equity *= 1.0 - cost_rate * t

            holdings = top_picks
            holdings_history.append((dates[i], holdings.copy()))

        equity_curve.append(equity)
        returns_series.append(daily_return)

    equity_curve_s = pd.Series(equity_curve, index=dates)
    returns_s = pd.Series(returns_series, index=dates)

    dd = compute_drawdown(equity_curve_s)

    metrics = {
        "cagr": annualized_cagr(equity_curve_s),
        "sharpe": annualized_sharpe(returns_s),
        "max_drawdown": float(dd.min()) if len(dd) else 0.0,
        "avg_turnover_per_rebalance": (
            float(np.mean(turnover_events)) if turnover_events else 0.0
        ),
    }

    return {
        "equity_curve": equity_curve_s,
        "returns": returns_s,
        "metrics": metrics,
        "holdings_history": holdings_history,
        "turnover_events": turnover_events,
    }


if __name__ == "__main__":
    # Self-test with synthetic data
    dates = pd.bdate_range("2022-01-01", "2025-01-01")
    tickers = [
        "AAPL",
        "MSFT",
        "META",
        "AMZN",
        "NVDA",
        "GOOGL",
        "JPM",
        "AXP",
        "COST",
        "LLY",
        "XOM",
    ]

    rng = np.random.default_rng(42)
    rets = rng.normal(0.0004, 0.02, size=(len(dates), len(tickers)))
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=tickers
    )

    out = backtest_rotation_v0(prices, config_path="config/alpha_v1.yaml", cost_bps=10)
    m = out["metrics"]

    print("=" * 60)
    print("BACKTEST RESULTS (SYNTHETIC)")
    print("=" * 60)
    print(f"CAGR:                         {m['cagr']*100:.2f}%")
    print(f"Sharpe Ratio:                 {m['sharpe']:.2f}")
    print(f"Max Drawdown:                 {m['max_drawdown']*100:.2f}%")
    print(f"Avg Turnover per Rebalance:   {m['avg_turnover_per_rebalance']*100:.2f}%")
    print("=" * 60)
    print(f"Final Equity:                 ${out['equity_curve'].iloc[-1]:.2f}")
    print(f"Number of Rebalances:         {len(out['holdings_history'])}")
    print("=" * 60)
