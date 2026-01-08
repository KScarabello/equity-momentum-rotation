import numpy as np
import pandas as pd

from research.momentum import (
    compute_12_1_momentum,
    apply_absolute_momentum_filter,
    pick_top_n,
)


def _make_prices_for_momentum(
    tickers=("AAA", "BBB", "CCC"),
    lookback_months=12,
    skip_recent_months=1,
):
    """
    Creates a business-day price DataFrame long enough for:
      start = t[-1 - lookback_days]
      end   = t[-1 - skip_days]
    using 21 trading days/month approximation, matching code.
    """
    lookback_days = lookback_months * 21
    skip_days = skip_recent_months * 21
    n = lookback_days + skip_days + 5  # extra cushion

    idx = pd.bdate_range("2020-01-01", periods=n)

    prices = pd.DataFrame(index=idx, columns=list(tickers), dtype=float)

    # Default: flat 100
    prices.loc[:, :] = 100.0

    # Choose the two anchor rows the algorithm uses
    start_i = -1 - lookback_days
    end_i = -1 - skip_days
    start_date = idx[start_i]
    end_date = idx[end_i]

    # Define simple momentum:
    # AAA: +50% over lookback window
    # BBB: +10%
    # CCC: -10%
    prices.loc[start_date, "AAA"] = 100.0
    prices.loc[end_date, "AAA"] = 150.0

    prices.loc[start_date, "BBB"] = 100.0
    prices.loc[end_date, "BBB"] = 110.0

    prices.loc[start_date, "CCC"] = 100.0
    prices.loc[end_date, "CCC"] = 90.0

    return prices


def test_compute_12_1_momentum_exact_math_and_order():
    prices = _make_prices_for_momentum()
    scores = compute_12_1_momentum(
        prices,
        lookback_months=12,
        skip_recent_months=1,
        min_price=0.0,
    )

    # Exact expected momentums
    assert np.isclose(scores.loc["AAA"], 0.50)
    assert np.isclose(scores.loc["BBB"], 0.10)
    assert np.isclose(scores.loc["CCC"], -0.10)

    # Must be sorted descending
    assert list(scores.index[:3]) == ["AAA", "BBB", "CCC"]


def test_min_price_filter_excludes_low_price_names():
    prices = _make_prices_for_momentum(tickers=("AAA", "PENNY"))
    # Make PENNY untradable at the final day
    prices.iloc[-1, prices.columns.get_loc("PENNY")] = 1.0  # below min_price
    # Ensure its start/end are otherwise fine
    scores = compute_12_1_momentum(
        prices,
        lookback_months=12,
        skip_recent_months=1,
        min_price=5.0,
    )
    assert "PENNY" not in scores.index
    assert "AAA" in scores.index


def test_absolute_momentum_filter_blocks_when_best_below_threshold():
    scores = pd.Series({"AAA": 0.05, "BBB": 0.01}).sort_values(ascending=False)
    picks = pick_top_n(scores, 1)
    out = apply_absolute_momentum_filter(scores, picks, enabled=True, min_momentum=0.10)
    assert out == []  # risk-off


def test_absolute_momentum_filter_allows_when_best_above_threshold():
    scores = pd.Series({"AAA": 0.20, "BBB": 0.05}).sort_values(ascending=False)
    picks = pick_top_n(scores, 1)
    out = apply_absolute_momentum_filter(scores, picks, enabled=True, min_momentum=0.10)
    assert out == ["AAA"]
