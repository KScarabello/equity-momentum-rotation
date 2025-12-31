import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Union


def load_config(path: Union[str, Path]) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def trading_days_from_months(months: int) -> int:
    # Approximate: 21 trading days per month
    return months * 21


def compute_12_1_momentum(
    prices: pd.DataFrame,
    lookback_months: int = 12,
    skip_recent_months: int = 1,
    min_price: float = 5.0,
) -> pd.Series:
    """
    prices: DataFrame indexed by date, columns = tickers, values = prices
    returns: Series indexed by ticker, momentum score
    """
    prices = prices.sort_index()

    lookback_days = trading_days_from_months(lookback_months)
    skip_days = trading_days_from_months(skip_recent_months)

    if len(prices) < lookback_days + skip_days:
        raise ValueError("Not enough data to compute momentum")

    end_prices = prices.iloc[-1 - skip_days]
    start_prices = prices.iloc[-1 - lookback_days]

    # Liquidity / sanity filter
    tradable = prices.iloc[-1] >= min_price
    end_prices = end_prices[tradable]
    start_prices = start_prices[tradable]

    momentum = (end_prices / start_prices) - 1
    momentum = momentum.replace([np.inf, -np.inf], np.nan).dropna()

    return momentum.sort_values(ascending=False)


def pick_top_n(momentum: pd.Series, n: int) -> list[str]:
    return list(momentum.head(n).index)


def apply_absolute_momentum_filter(
    scores: pd.Series,
    picks: list[str],
    enabled: bool,
    min_momentum: float,
) -> list[str]:
    """
    If enabled and the best momentum score is below min_momentum,
    return an empty list (meaning: hold CASH).
    """
    if not enabled:
        return picks
    if scores.empty:
        return []
    best = float(scores.iloc[0])
    if best < min_momentum:
        return []
    return picks


if __name__ == "__main__":
    # ---- Self-test with fake data ----
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

    rng = np.random.default_rng(0)
    returns = rng.normal(0.0004, 0.02, size=(len(dates), len(tickers)))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)),
        index=dates,
        columns=tickers,
    )

    cfg = load_config("config/alpha_v1.yaml")
    abs_filter = cfg["signal"]["momentum"].get("absolute_momentum_filter", {})
    abs_enabled = bool(abs_filter.get("enabled", False))
    min_mom = float(abs_filter.get("min_momentum", 0.0))
    n = cfg["portfolio"]["top_n"]
    min_price = cfg["universe"]["filters"]["min_price"]

scores = compute_12_1_momentum(prices, min_price=min_price)

top_n = int(cfg["portfolio"]["top_n"])
top = pick_top_n(scores, top_n)
top = apply_absolute_momentum_filter(scores, top, abs_enabled, min_mom)


if len(top) == 0:
    print("RISK-OFF TRIGGERED: best momentum < min_momentum")
    print("Holding CASH (no equity positions)")
else:
    print("Top picks:", top)
    print("Scores:")
    print(scores.loc[top])
