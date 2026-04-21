from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Optional

from research.walk_forward_momentum import WalkForwardConfig


# Strategy baseline (locked)
BASELINE_POSITIONS = 12
BASELINE_REBALANCE_INTERVAL_WEEKS = 3  # ~15 trading days
BASELINE_W_3M = 0.60
BASELINE_W_6M = 0.30
BASELINE_W_12M = 0.10
BASELINE_COST_BPS = 5.0


@dataclass(frozen=True)
class LiveTradingConfig:
    state_file: Path = Path("live/alpaca_live_state.json")
    logs_dir: Path = Path("logs")

    # Execution window (ET)
    execution_window_start: time = time(hour=9, minute=50)
    execution_window_end: time = time(hour=10, minute=5)

    # Order behavior
    buy_limit_buffer: float = 1.001
    sell_limit_buffer: float = 0.999
    retry_enabled: bool = True
    retry_wait_seconds: int = 20
    retry_buy_limit_buffer: float = 1.002
    retry_sell_limit_buffer: float = 0.998

    # Trade filters / risk controls
    min_notional_trade: float = 5.0
    min_share_delta: float = 0.0001
    max_deployment_fraction: float = 0.95
    max_position_weight_tolerance: float = 0.03
    allow_margin: bool = False

    # Trading calendar lookback for cadence
    rebalance_calendar_days_back: int = 730

    # Signal-data freshness guardrail
    max_signal_staleness_days: int = 3


def build_baseline_cfg() -> WalkForwardConfig:
    return WalkForwardConfig(
        train_years=3,
        test_months=6,
        step_months=6,
        positions=BASELINE_POSITIONS,
        universe_top_n=800,
        rebalance_weekday=0,
        rebalance_interval_weeks=BASELINE_REBALANCE_INTERVAL_WEEKS,
        starting_cash=100_000.0,
        liq_lookback=60,
        mom_3m=63,
        mom_6m=126,
        mom_12m=252,
        w_3m=BASELINE_W_3M,
        w_6m=BASELINE_W_6M,
        w_12m=BASELINE_W_12M,
        use_strength_filter=False,
        percentile_filter_enabled=False,
        market_filter_mode="none",
        momentum_effectiveness_skip_threshold=None,
        veto_if_12m_return_below=0.0,
        market_symbol="SPY",
        market_sma_days=200,
        risk_on_buffer=0.0,
        cost_bps=BASELINE_COST_BPS,
        slippage_bps=2.0,
        min_exposure=0.25,
        max_exposure=1.0,
        exposure_slope=0.0,
        require_positive_sma_slope=True,
        sma_slope_lookback=20,
        stability_lookback_periods=1,
        min_rebalance_weight_change=0.0,
    )
