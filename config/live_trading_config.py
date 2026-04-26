from __future__ import annotations

from dataclasses import dataclass
from datetime import time
import os
from pathlib import Path
from typing import Optional

from config.strategy_public_defaults import PUBLIC_STRATEGY_DEFAULTS
from research.walk_forward_momentum import WalkForwardConfig


def _load_private_strategy_overrides() -> dict[str, object]:
    """Load optional local-only strategy overrides from config_private.py."""
    try:
        from config_private import STRATEGY_PRIVATE_OVERRIDES  # type: ignore
    except ModuleNotFoundError:
        return {}

    if not isinstance(STRATEGY_PRIVATE_OVERRIDES, dict):
        raise ValueError("STRATEGY_PRIVATE_OVERRIDES must be a dict if provided.")
    return dict(STRATEGY_PRIVATE_OVERRIDES)


def _build_strategy_defaults() -> dict[str, object]:
    cfg = dict(PUBLIC_STRATEGY_DEFAULTS)
    cfg.update(_load_private_strategy_overrides())
    return cfg


@dataclass(frozen=True)
class LiveTradingConfig:
    state_file: Path = Path("live/alpaca_live_state.json")
    logs_dir: Path = Path("logs")

    # Broker connectivity / mode
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    dry_run: bool = True

    # First deployment safety
    first_run_liquidate_all: bool = False
    cancel_open_orders_on_start: bool = False

    # Execution window (ET)
    execution_window_start: time = time(hour=9, minute=50)
    execution_window_end: time = time(hour=10, minute=5)

    # Order behavior
    buy_limit_buffer_bps: float = 10.0
    sell_limit_buffer_bps: float = 10.0
    retry_enabled: bool = True
    retry_wait_seconds: int = 20
    retry_buy_limit_buffer_bps: float = 20.0
    retry_sell_limit_buffer_bps: float = 20.0
    time_in_force: str = "day"

    # Trade filters / risk controls
    min_trade_notional: float = 10.0
    max_deployment_pct: float = 0.60
    max_positions: int = int(PUBLIC_STRATEGY_DEFAULTS.get("positions", 8))
    max_order_count: int = 40
    min_sell_qty: float = 0.000001
    max_position_weight_tolerance: float = 0.03
    allow_margin: bool = False

    # Trading calendar lookback for cadence
    rebalance_calendar_days_back: int = 730

    # Signal-data freshness guardrail
    max_signal_staleness_days: int = 3


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    return v in {"1", "true", "t", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_time(name: str, default: time) -> time:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default

    value = raw.strip()
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format for {name}: {value}; expected HH:MM")
    hour = int(parts[0])
    minute = int(parts[1])
    return time(hour=hour, minute=minute)


def load_live_trading_config() -> LiveTradingConfig:
    strat_defaults = _build_strategy_defaults()

    cfg = LiveTradingConfig(
        alpaca_api_key=os.getenv("ALPACA_API_KEY", "").strip(),
        alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY", "").strip(),
        alpaca_base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").strip(),
        dry_run=_env_bool("DRY_RUN", True),
        first_run_liquidate_all=_env_bool("FIRST_RUN_LIQUIDATE_ALL", False),
        cancel_open_orders_on_start=_env_bool("CANCEL_OPEN_ORDERS_ON_START", False),
        execution_window_start=_env_time("EXECUTION_WINDOW_START_ET", time(hour=9, minute=50)),
        execution_window_end=_env_time("EXECUTION_WINDOW_END_ET", time(hour=10, minute=5)),
        buy_limit_buffer_bps=_env_float("BUY_LIMIT_BUFFER_BPS", 10.0),
        sell_limit_buffer_bps=_env_float("SELL_LIMIT_BUFFER_BPS", 10.0),
        retry_enabled=_env_bool("RETRY_ENABLED", True),
        retry_wait_seconds=_env_int("RETRY_WAIT_SECONDS", 20),
        retry_buy_limit_buffer_bps=_env_float("RETRY_BUY_LIMIT_BUFFER_BPS", 20.0),
        retry_sell_limit_buffer_bps=_env_float("RETRY_SELL_LIMIT_BUFFER_BPS", 20.0),
        time_in_force=os.getenv("TIME_IN_FORCE", "day").strip().lower() or "day",
        min_trade_notional=_env_float("MIN_TRADE_NOTIONAL", 10.0),
        max_deployment_pct=_env_float("MAX_DEPLOYMENT_PCT", 0.60),
        max_positions=_env_int("MAX_POSITIONS", int(strat_defaults.get("positions", 8))),
        max_order_count=_env_int("MAX_ORDER_COUNT", 40),
        min_sell_qty=_env_float("MIN_SELL_QTY", 0.000001),
    )

    if not (0.0 < cfg.max_deployment_pct <= 1.0):
        raise ValueError("MAX_DEPLOYMENT_PCT must be in (0, 1]")
    if cfg.max_positions <= 0:
        raise ValueError("MAX_POSITIONS must be > 0")
    if cfg.max_order_count <= 0:
        raise ValueError("MAX_ORDER_COUNT must be > 0")
    if cfg.min_trade_notional <= 0:
        raise ValueError("MIN_TRADE_NOTIONAL must be > 0")

    return cfg


def build_baseline_cfg() -> WalkForwardConfig:
    return WalkForwardConfig(**_build_strategy_defaults())
