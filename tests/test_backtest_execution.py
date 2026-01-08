import numpy as np
import pandas as pd

from research.backtest_v0 import backtest_rotation_v0


def _write_test_config(tmp_path, trading_days_interval=5, top_n=1, min_momentum=0.0):
    """
    Writes a minimal alpha_v1-like YAML config for tests.
    """
    cfg = f"""
universe:
  filters:
    min_price: 0.0

signal:
  momentum:
    lookback_months: 1
    skip_recent_months: 0
    absolute_momentum_filter:
      enabled: true
      min_momentum: {min_momentum}

portfolio:
  top_n: {top_n}

rebalance:
  trading_days_interval: {trading_days_interval}
"""
    p = tmp_path / "alpha_test.yaml"
    p.write_text(cfg.strip() + "\n")
    return str(p)


def _make_simple_prices():
    """
    Two symbols:
      WIN: rises steadily
      LOSE: flat
    Enough history for lookback_months=1 and skip=0 using 21 trading days/month.
    """
    n = 60
    idx = pd.bdate_range("2021-01-01", periods=n)

    win = np.linspace(100, 160, n)
    lose = np.full(n, 100.0)

    return pd.DataFrame({"WIN": win, "LOSE": lose}, index=idx)


def test_backtest_returns_keys_and_shapes(tmp_path):
    prices = _make_simple_prices()
    cfg_path = _write_test_config(
        tmp_path, trading_days_interval=5, top_n=1, min_momentum=0.0
    )

    out = backtest_rotation_v0(prices, config_path=cfg_path, cost_bps=0.0)

    assert "equity_curve" in out
    assert "returns" in out
    assert "metrics" in out
    assert "holdings_history" in out
    assert len(out["equity_curve"]) == len(prices)
    assert len(out["returns"]) == len(prices)


def test_costs_reduce_equity_when_turnover_nonzero(tmp_path):
    prices = _make_simple_prices()

    # Force turnover by making LOSE spike in momentum halfway through:
    # We'll do this by making LOSE jump late so it becomes top pick at a rebalance.
    prices2 = prices.copy()
    prices2.iloc[-10:, prices2.columns.get_loc("LOSE")] = np.linspace(100, 200, 10)

    cfg_path = _write_test_config(
        tmp_path, trading_days_interval=5, top_n=1, min_momentum=0.0
    )

    out_no_cost = backtest_rotation_v0(prices2, config_path=cfg_path, cost_bps=0.0)
    out_cost = backtest_rotation_v0(
        prices2, config_path=cfg_path, cost_bps=50.0
    )  # 50 bps

    # With costs, final equity must be lower or equal
    assert out_cost["equity_curve"].iloc[-1] <= out_no_cost["equity_curve"].iloc[-1]


def test_absolute_momentum_filter_can_hold_cash(tmp_path):
    prices = _make_simple_prices()

    # Set min_momentum very high so we should go risk-off (CASH) always.
    cfg_path = _write_test_config(
        tmp_path, trading_days_interval=5, top_n=1, min_momentum=10.0
    )

    out = backtest_rotation_v0(prices, config_path=cfg_path, cost_bps=0.0)

    # If always cash after min_history, returns should be ~0 most of the time.
    # We don't assert all zeros because before min_history it's also effectively cash,
    # but this test ensures we don't accidentally take positions.
    holdings = [h for _, h in out["holdings_history"]]
    assert all(len(h) == 0 for h in holdings)
