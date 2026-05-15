import math

import pytest
import pandas as pd

from research.walk_forward_momentum import WalkForwardConfig, run_weekly_portfolio


def _make_cfg(execution_price: str) -> WalkForwardConfig:
    return WalkForwardConfig(
        positions=1,
        universe_top_n=2,
        rebalance_weekday=3,
        rebalance_interval_weeks=1,
        execution_price=execution_price,
        starting_cash=100.0,
        liq_lookback=1,
        mom_3m=1,
        mom_6m=2,
        mom_12m=3,
        w_3m=1.0,
        w_6m=0.0,
        w_12m=0.0,
        veto_if_12m_return_below=-1.0,
        market_symbol="SPY",
        market_sma_days=2,
        cost_bps=0.0,
        slippage_bps=0.0,
        min_exposure=0.0,
        max_exposure=1.0,
        exposure_slope=0.0,
    )


def _make_market(index: pd.DatetimeIndex, closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": closes,
            "volume": [1_000_000] * len(index),
        },
        index=index,
    )


def _make_symbol(index: pd.DatetimeIndex, closes: list[float], opens: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": opens,
            "close": closes,
            "volume": [100_000] * len(index),
        },
        index=index,
    )


def _gap_record(res, symbol: str) -> dict:
    for row in res.gap_records:
        if row["symbol"] == symbol:
            return row
    raise AssertionError(f"Missing gap record for {symbol}")


def test_next_open_uses_close_t_signal_and_open_t_plus_1_execution():
    idx = pd.bdate_range("2021-01-04", periods=5)
    market = _make_market(idx, [100, 101, 102, 103, 104])
    symbol_dfs = {
        "AAA": _make_symbol(idx, [8, 8, 9, 10, 12], [8, 8, 9, 10, 12]),
        "BBB": _make_symbol(idx, [8, 8, 9, 9, 100], [8, 8, 9, 9, 9]),
    }

    res = run_weekly_portfolio(
        symbol_dfs=symbol_dfs,
        market_df=market,
        start=idx[0],
        end=idx[-1],
        cfg=_make_cfg("next_open"),
    )

    assert res.rebalance_records[0]["selected_symbols"] == "AAA"
    assert res.rebalance_records[0]["decision_date"] == idx[3]
    assert res.rebalance_records[0]["execution_date"] == idx[4]
    assert res.ending_holdings["AAA"] == pytest.approx(100.0 / 12.0)

    gap = _gap_record(res, "AAA")
    assert gap["close_t"] == pytest.approx(10.0)
    assert gap["execution_price"] == pytest.approx(12.0)
    assert gap["overnight_gap_return"] == pytest.approx(0.2)
    assert gap["action"] == "bought"
    assert gap["pre_trade_weight"] == pytest.approx(0.0)
    assert gap["post_trade_weight"] == pytest.approx(1.0)
    assert gap["trade_weight_change"] == pytest.approx(1.0)
    assert gap["traded_notional_pct"] == pytest.approx(1.0)


def test_next_open_does_not_trade_at_close_t():
    idx = pd.bdate_range("2021-01-04", periods=5)
    market = _make_market(idx, [100, 101, 102, 103, 104])
    symbol_dfs = {
        "AAA": _make_symbol(idx, [8, 8, 9, 10, 12], [8, 8, 9, 10, 12]),
        "BBB": _make_symbol(idx, [8, 8, 9, 9, 100], [8, 8, 9, 9, 9]),
    }

    res = run_weekly_portfolio(
        symbol_dfs=symbol_dfs,
        market_df=market,
        start=idx[0],
        end=idx[-1],
        cfg=_make_cfg("next_open"),
        snapshot_dates={idx[3]},
    )

    assert len(res.state_snapshots) == 1
    snapshot = res.state_snapshots[0]
    assert snapshot["date"] == idx[3]
    assert snapshot["cash"] == pytest.approx(100.0)
    assert snapshot["invested_value"] == pytest.approx(0.0)
    assert snapshot["holdings_count"] == 0


def test_next_open_does_not_use_close_t_plus_1_for_signal():
    idx = pd.bdate_range("2021-01-04", periods=5)
    market = _make_market(idx, [100, 101, 102, 103, 104])
    symbol_dfs = {
        "AAA": _make_symbol(idx, [8, 8, 9, 10, 11], [8, 8, 9, 10, 11]),
        "BBB": _make_symbol(idx, [8, 8, 9, 9, 200], [8, 8, 9, 9, 9]),
    }

    res = run_weekly_portfolio(
        symbol_dfs=symbol_dfs,
        market_df=market,
        start=idx[0],
        end=idx[-1],
        cfg=_make_cfg("next_open"),
    )

    assert res.rebalance_records[0]["selected_symbols"] == "AAA"


def test_next_open_buys_gap_up_at_higher_open():
    idx = pd.bdate_range("2021-01-04", periods=5)
    market = _make_market(idx, [100, 101, 102, 103, 104])
    symbol_dfs = {
        "AAA": _make_symbol(idx, [8, 8, 9, 10, 12], [8, 8, 9, 10, 12]),
        "BBB": _make_symbol(idx, [8, 8, 9, 9, 9], [8, 8, 9, 9, 9]),
    }

    res = run_weekly_portfolio(
        symbol_dfs=symbol_dfs,
        market_df=market,
        start=idx[0],
        end=idx[-1],
        cfg=_make_cfg("next_open"),
    )

    assert res.ending_holdings["AAA"] == pytest.approx(100.0 / 12.0)
    assert res.equity_curve.loc[idx[-1]] == pytest.approx(100.0)


def test_next_open_buys_gap_down_at_lower_open():
    idx = pd.bdate_range("2021-01-04", periods=5)
    market = _make_market(idx, [100, 101, 102, 103, 104])
    symbol_dfs = {
        "AAA": _make_symbol(idx, [8, 8, 9, 10, 8], [8, 8, 9, 10, 8]),
        "BBB": _make_symbol(idx, [8, 8, 9, 9, 9], [8, 8, 9, 9, 9]),
    }

    res = run_weekly_portfolio(
        symbol_dfs=symbol_dfs,
        market_df=market,
        start=idx[0],
        end=idx[-1],
        cfg=_make_cfg("next_open"),
    )

    assert res.ending_holdings["AAA"] == pytest.approx(100.0 / 8.0)
    assert res.equity_curve.loc[idx[-1]] == pytest.approx(100.0)


def test_missing_next_day_open_skips_fill_without_fake_price():
    idx = pd.bdate_range("2021-01-04", periods=5)
    market = _make_market(idx, [100, 101, 102, 103, 104])
    symbol_dfs = {
        "AAA": _make_symbol(idx, [8, 8, 9, 10, 12], [8, 8, 9, 10, math.nan]),
        "BBB": _make_symbol(idx, [8, 8, 9, 9, 9], [8, 8, 9, 9, 9]),
    }

    res = run_weekly_portfolio(
        symbol_dfs=symbol_dfs,
        market_df=market,
        start=idx[0],
        end=idx[-1],
        cfg=_make_cfg("next_open"),
    )

    assert res.ending_holdings == {}
    assert res.ending_cash == pytest.approx(100.0)
    assert res.skipped_execution_symbols == 1
    assert res.rebalance_records[0]["skipped_execution_symbols"] == 1

    gap = _gap_record(res, "AAA")
    assert gap["action"] == "skipped"
    assert pd.isna(gap["execution_price"])
    assert gap["pre_trade_weight"] == pytest.approx(0.0)
    assert gap["post_trade_weight"] == pytest.approx(0.0)
    assert gap["trade_weight_change"] == pytest.approx(0.0)
    assert gap["traded_notional_pct"] == pytest.approx(0.0)


def test_close_mode_behavior_remains_unchanged():
    idx = pd.bdate_range("2021-01-04", periods=5)
    market = _make_market(idx, [100, 101, 102, 103, 104])
    symbol_dfs = {
        "AAA": _make_symbol(idx, [8, 8, 9, 10, 12], [8, 8, 9, 10, 12]),
        "BBB": _make_symbol(idx, [8, 8, 9, 9, 9], [8, 8, 9, 9, 9]),
    }

    res = run_weekly_portfolio(
        symbol_dfs=symbol_dfs,
        market_df=market,
        start=idx[0],
        end=idx[-1],
        cfg=_make_cfg("close"),
    )

    assert res.rebalance_records[0]["decision_date"] == idx[3]
    assert res.rebalance_records[0]["execution_date"] == idx[3]
    assert res.rebalance_records[0]["selected_symbols"] == "AAA"
    assert res.ending_holdings["AAA"] == pytest.approx(10.0)
    assert res.equity_curve.loc[idx[-1]] == pytest.approx(120.0)
    assert res.gap_records == []


def test_next_open_risk_off_exits_at_next_open():
    idx = pd.bdate_range("2021-01-04", periods=5)
    market = _make_market(idx, [100, 100, 100, 90, 89])
    symbol_dfs = {
        "AAA": _make_symbol(idx, [8, 8, 9, 10, 7], [8, 8, 9, 10, 8]),
        "BBB": _make_symbol(idx, [8, 8, 8, 8, 8], [8, 8, 8, 8, 8]),
    }

    res = run_weekly_portfolio(
        symbol_dfs=symbol_dfs,
        market_df=market,
        start=idx[0],
        end=idx[-1],
        cfg=_make_cfg("next_open"),
        initial_cash=0.0,
        initial_holdings={"AAA": 10.0},
    )

    assert res.rebalance_records[0]["risk_on"] is False
    assert res.rebalance_records[0]["execution_date"] == idx[-1]
    assert res.ending_holdings == {}
    assert res.ending_cash == pytest.approx(80.0)

    gap = _gap_record(res, "AAA")
    assert gap["action"] == "sold"
    assert gap["pre_trade_weight"] == pytest.approx(1.0)
    assert gap["post_trade_weight"] == pytest.approx(0.0)
    assert gap["trade_weight_change"] == pytest.approx(-1.0)
    assert gap["traded_notional_pct"] == pytest.approx(1.0)


def test_next_open_partial_rebalance_has_exact_weight_changes():
    idx = pd.bdate_range("2021-01-04", periods=5)
    market = _make_market(idx, [100, 101, 102, 103, 104])
    symbol_dfs = {
        "AAA": _make_symbol(idx, [8, 9, 9, 10, 10], [8, 9, 9, 10, 10]),
        "BBB": _make_symbol(idx, [8, 9, 9, 10, 10], [8, 9, 9, 10, 10]),
    }

    cfg = WalkForwardConfig(**{**_make_cfg("next_open").__dict__, "positions": 2})
    res = run_weekly_portfolio(
        symbol_dfs=symbol_dfs,
        market_df=market,
        start=idx[0],
        end=idx[-1],
        cfg=cfg,
        initial_cash=0.0,
        initial_holdings={"AAA": 10.0},
    )

    gap_aaa = _gap_record(res, "AAA")
    gap_bbb = _gap_record(res, "BBB")

    assert gap_aaa["action"] == "sold"
    assert gap_aaa["pre_trade_weight"] == pytest.approx(1.0)
    assert gap_aaa["post_trade_weight"] == pytest.approx(0.5)
    assert gap_aaa["trade_weight_change"] == pytest.approx(-0.5)
    assert gap_aaa["traded_notional_pct"] == pytest.approx(0.5)

    assert gap_bbb["action"] == "bought"
    assert gap_bbb["pre_trade_weight"] == pytest.approx(0.0)
    assert gap_bbb["post_trade_weight"] == pytest.approx(0.5)
    assert gap_bbb["trade_weight_change"] == pytest.approx(0.5)
    assert gap_bbb["traded_notional_pct"] == pytest.approx(0.5)


def test_next_open_skips_final_rebalance_without_next_session():
    idx = pd.bdate_range("2021-01-04", periods=4)
    market = _make_market(idx, [100, 101, 102, 103])
    symbol_dfs = {
        "AAA": _make_symbol(idx, [8, 8, 9, 10], [8, 8, 9, 10]),
        "BBB": _make_symbol(idx, [8, 8, 9, 9], [8, 8, 9, 9]),
    }

    res = run_weekly_portfolio(
        symbol_dfs=symbol_dfs,
        market_df=market,
        start=idx[0],
        end=idx[-1],
        cfg=_make_cfg("next_open"),
    )

    assert res.ending_holdings == {}
    assert res.skipped_execution_rebalances == 1
    assert res.rebalance_records[0]["skip_reason"] == "no_next_execution_day"
    assert pd.isna(res.rebalance_records[0]["execution_date"])