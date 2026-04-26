#!/usr/bin/env python3
"""
Production-safe Alpaca live runner for the equity momentum strategy.

Run:
    python3 -m live.run_alpaca_live_trader --dry-run --verbose
    python3 -m live.run_alpaca_live_trader --live --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List
from zoneinfo import ZoneInfo

import pandas as pd

from config.live_trading_config import LiveTradingConfig, build_baseline_cfg, load_live_trading_config
from live.alpaca_client import AlpacaBroker, AlpacaCredentials, AlpacaDependencyError
from live.execution_gate import (
    filter_symbols_already_pending,
    validate_account_for_trading,
    validate_assets_for_orders,
    validate_order_plan_shape,
    within_execution_window,
)
from live.rebalance_planner import PlannedOrder, build_rebalance_plan, positions_from_alpaca
from live.state_store import load_state, save_state
from research.run_walk_forward import STOOQ_DIR, build_universe_from_stooq, fetch_ohlcv
from research.walk_forward_momentum import (
    _ensure_datetime_index,
    _normalize_cols,
    _week_rebalance_dates,
    compute_rebalance_target,
)


ET = ZoneInfo("America/New_York")


def _load_local_env(env_path: Path = Path(".env")) -> None:
    try:
        from dotenv import load_dotenv as _load_dotenv  # type: ignore

        _load_dotenv(dotenv_path=env_path, override=False)
        return
    except Exception:
        pass

    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env(Path(".env"))


@dataclass(frozen=True)
class RuntimeContext:
    run_id: str
    dry_run: bool
    force: bool
    verbose: bool


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one Alpaca live rebalance cycle")
    p.add_argument("--dry-run", action="store_true", help="Compute and log only; do not place orders")
    p.add_argument("--live", action="store_true", help="Enable live order placement")
    p.add_argument("--force", action="store_true", help="Bypass execution window and duplicate-cycle checks")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def _setup_logging(cfg: LiveTradingConfig, run_id: str, verbose: bool) -> logging.Logger:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    log_fp = cfg.logs_dir / f"live_run_{run_id}.log"

    logger = logging.getLogger("alpaca_live_trader")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_fp, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(sh)

    logger.info("logging_path=%s", log_fp)
    return logger


def _log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info("%s", json.dumps(payload, default=str, sort_keys=True))


def _to_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _build_rebalance_calendar(
    broker: AlpacaBroker,
    cfg: LiveTradingConfig,
    today_et: date,
) -> pd.DatetimeIndex:
    start = today_et - timedelta(days=cfg.rebalance_calendar_days_back)
    days = broker.get_trading_days(start=start, end=today_et)
    if not days:
        raise RuntimeError("No Alpaca trading calendar days returned")

    cal = pd.DatetimeIndex(pd.to_datetime(days)).sort_values()
    rebals = _week_rebalance_dates(cal, weekday=0)
    if len(rebals) == 0:
        return pd.DatetimeIndex([cal[-1]])

    baseline_cfg = build_baseline_cfg()
    if baseline_cfg.rebalance_interval_weeks > 1:
        rebals = rebals[:: baseline_cfg.rebalance_interval_weeks]
    return pd.DatetimeIndex(rebals)


def _load_strategy_data(logger: logging.Logger) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    symbols = build_universe_from_stooq(STOOQ_DIR)
    _log_event(logger, "strategy_universe", symbol_count=len(symbols))

    market_df = _ensure_datetime_index(_normalize_cols(fetch_ohlcv("SPY")))

    symbol_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            symbol_dfs[sym] = _ensure_datetime_index(_normalize_cols(fetch_ohlcv(sym)))
        except Exception as exc:
            _log_event(logger, "symbol_load_failed", symbol=sym, error=str(exc))

    if not symbol_dfs:
        raise RuntimeError("No symbol data loaded from Stooq cache")

    _log_event(logger, "strategy_data_loaded", loaded_symbols=len(symbol_dfs))
    return symbol_dfs, market_df


def _get_signal_asof_date(market_df: pd.DataFrame, today_et: date) -> pd.Timestamp:
    cutoff = pd.Timestamp(today_et) - pd.Timedelta(days=1)
    hist = market_df.index[market_df.index <= cutoff]
    if len(hist) == 0:
        raise RuntimeError("No completed market history is available for signal generation")
    return pd.Timestamp(hist.max())


def _fetch_latest_prices(broker: AlpacaBroker, symbols: Iterable[str]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for sym in sorted({str(s).strip().upper() for s in symbols if str(s).strip()}):
        px = broker.get_latest_trade_price(sym)
        if px is None:
            continue
        if float(px) <= 0:
            continue
        prices[sym] = float(px)
    return prices


def _orders_to_rows(orders: List[PlannedOrder]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for o in orders:
        rows.append(asdict(o))
    rows.sort(key=lambda x: (0 if x["side"] == "sell" else 1, -float(x.get("target_notional", 0.0))))
    return rows


def _save_rebalance_summary(
    cfg: LiveTradingConfig,
    run_id: str,
    payload: Dict[str, Any],
    orders_rows: List[Dict[str, Any]],
    positions_rows: List[Dict[str, Any]],
) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)

    summary_json = cfg.logs_dir / f"rebalance_summary_{run_id}.json"
    summary_txt = cfg.logs_dir / f"rebalance_summary_{run_id}.txt"
    orders_csv = cfg.logs_dir / f"rebalance_orders_{run_id}.csv"
    positions_csv = cfg.logs_dir / f"rebalance_positions_{run_id}.csv"

    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    lines = [
        f"run_id: {run_id}",
        f"timestamp_et: {payload.get('timestamp_et')}",
        f"mode: {'DRY_RUN' if payload.get('dry_run') else 'LIVE'}",
        f"account_equity: {payload.get('account_equity')}",
        f"buying_power: {payload.get('buying_power')}",
        f"target_symbols: {','.join(payload.get('target_symbols', []))}",
        f"planned_orders: {len(orders_rows)}",
        f"liquidation_mode: {payload.get('liquidation_mode')}",
        "",
        "abort_reasons:",
    ]
    for reason in payload.get("abort_reasons", []):
        lines.append(f"- {reason}")

    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    pd.DataFrame(orders_rows).to_csv(orders_csv, index=False)
    pd.DataFrame(positions_rows).to_csv(positions_csv, index=False)


def _build_creds_from_cfg(cfg: LiveTradingConfig) -> AlpacaCredentials:
    missing = []
    if not cfg.alpaca_api_key:
        missing.append("ALPACA_API_KEY")
    if not cfg.alpaca_secret_key:
        missing.append("ALPACA_SECRET_KEY")
    if not cfg.alpaca_base_url:
        missing.append("ALPACA_BASE_URL")
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return AlpacaCredentials(
        api_key=cfg.alpaca_api_key,
        secret_key=cfg.alpaca_secret_key,
        base_url=cfg.alpaca_base_url,
    )


def _submit_orders(
    broker: AlpacaBroker,
    logger: logging.Logger,
    orders_rows: List[Dict[str, Any]],
    run_id: str,
    tif: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for idx, row in enumerate(orders_rows, start=1):
        symbol = str(row["symbol"]).upper()
        side = str(row["side"]).lower()
        limit_price = float(row["limit_price"])
        reason = str(row.get("reason", ""))
        cid = f"mom-{run_id}-{idx}"

        try:
            if side == "buy":
                notional = float(row.get("notional") or 0.0)
                if notional <= 0:
                    raise ValueError(f"Invalid buy notional for {symbol}: {notional}")
                _log_event(
                    logger,
                    "submit_buy_notional",
                    symbol=symbol,
                    notional=notional,
                    limit_price=limit_price,
                    client_order_id=cid,
                    reason=reason,
                )
                order = broker.submit_fractional_buy_notional(
                    symbol=symbol,
                    notional=notional,
                    limit_price=limit_price,
                    client_order_id=cid,
                    tif=tif,
                )
            elif side == "sell":
                qty = float(row.get("qty") or 0.0)
                if qty <= 0:
                    raise ValueError(f"Invalid sell qty for {symbol}: {qty}")
                _log_event(
                    logger,
                    "submit_sell_qty",
                    symbol=symbol,
                    qty=qty,
                    limit_price=limit_price,
                    client_order_id=cid,
                    reason=reason,
                )
                order = broker.submit_fractional_sell_qty(
                    symbol=symbol,
                    qty=qty,
                    limit_price=limit_price,
                    client_order_id=cid,
                    tif=tif,
                )
            else:
                raise ValueError(f"Invalid order side: {side}")

            order_id = str(getattr(order, "id", ""))
            latest = order
            if order_id:
                try:
                    latest = broker.get_order(order_id)
                except Exception:
                    latest = order

            status = str(getattr(latest, "status", "unknown"))
            filled_qty = _to_float(getattr(latest, "filled_qty", 0.0))
            filled_avg_price = _to_float(getattr(latest, "filled_avg_price", 0.0))
            results.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "order_id": order_id,
                    "status": status,
                    "client_order_id": cid,
                    "filled_qty": filled_qty,
                    "filled_avg_price": filled_avg_price,
                    "error": "",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "order_id": "",
                    "status": "submit_error",
                    "client_order_id": cid,
                    "filled_qty": 0.0,
                    "filled_avg_price": 0.0,
                    "error": str(exc),
                }
            )
            _log_event(logger, "order_submit_error", symbol=symbol, side=side, error=str(exc))

        time.sleep(0.25)

    return results


def main() -> None:
    args = _parse_args()

    cfg = load_live_trading_config()
    dry_run = cfg.dry_run
    if args.live:
        dry_run = False
    if args.dry_run:
        dry_run = True

    run_id = datetime.now(tz=ET).strftime("%Y%m%d_%H%M%S")
    logger = _setup_logging(cfg=cfg, run_id=run_id, verbose=args.verbose)
    ctx = RuntimeContext(run_id=run_id, dry_run=dry_run, force=args.force, verbose=args.verbose)

    try:
        creds = _build_creds_from_cfg(cfg)
        broker = AlpacaBroker(creds)
    except AlpacaDependencyError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    except Exception as exc:
        logger.error("broker_init_failed: %s", exc)
        sys.exit(2)

    try:
        account = broker.get_account()
        clock = broker.get_clock()
    except Exception as exc:
        logger.error("failed_to_fetch_account_or_clock: %s", exc)
        return

    now_et = getattr(clock, "timestamp", datetime.now(tz=ET))
    if now_et.tzinfo is None:
        now_et = now_et.replace(tzinfo=ET)
    now_et = now_et.astimezone(ET)
    today_et = now_et.date()
    cycle_key = f"{today_et.isoformat()}_{cfg.execution_window_start.strftime('%H%M')}_{cfg.execution_window_end.strftime('%H%M')}"

    account_equity = _to_float(getattr(account, "equity", 0.0))
    account_cash = _to_float(getattr(account, "cash", 0.0))
    buying_power = _to_float(getattr(account, "buying_power", 0.0))

    _log_event(
        logger,
        "account_snapshot",
        timestamp_et=now_et.isoformat(),
        account_equity=account_equity,
        cash=account_cash,
        buying_power=buying_power,
        paper_mode=("paper" in creds.base_url.lower()),
        dry_run=ctx.dry_run,
        first_run_liquidate_all=cfg.first_run_liquidate_all,
    )

    window_ok = within_execution_window(now_et, cfg.execution_window_start, cfg.execution_window_end)
    market_open = bool(getattr(clock, "is_open", False))

    state = load_state(cfg.state_file)
    duplicate_cycle = state.get("last_cycle_key") == cycle_key

    if not market_open:
        _log_event(logger, "skip_cycle", reason="market_closed")
        return

    if not window_ok and not ctx.force:
        _log_event(
            logger,
            "skip_cycle",
            reason="outside_execution_window",
            window_start=str(cfg.execution_window_start),
            window_end=str(cfg.execution_window_end),
            now_et=now_et.isoformat(),
        )
        return

    if duplicate_cycle and not ctx.force:
        _log_event(logger, "skip_cycle", reason="duplicate_cycle", cycle_key=cycle_key)
        return

    account_gate = validate_account_for_trading(account)
    if not account_gate.ok:
        for reason in account_gate.reasons:
            _log_event(logger, "abort", reason=reason)
        return

    if cfg.cancel_open_orders_on_start:
        try:
            broker.cancel_open_orders()
            _log_event(logger, "cancel_open_orders", status="requested")
        except Exception as exc:
            _log_event(logger, "abort", reason=f"cancel_open_orders_failed: {exc}")
            return

    try:
        symbol_dfs, market_df = _load_strategy_data(logger)
        signal_asof = _get_signal_asof_date(market_df, today_et=today_et)
    except Exception as exc:
        _log_event(logger, "abort", reason=f"strategy_data_load_failed: {exc}")
        return

    signal_staleness_days = (today_et - signal_asof.date()).days
    if signal_staleness_days > cfg.max_signal_staleness_days:
        _log_event(logger, "abort", reason=f"signal_staleness_days={signal_staleness_days}")
        return

    try:
        rebals = _build_rebalance_calendar(broker=broker, cfg=cfg, today_et=today_et)
    except Exception as exc:
        _log_event(logger, "abort", reason=f"rebalance_calendar_failed: {exc}")
        return

    if pd.Timestamp(today_et) not in rebals:
        _log_event(logger, "skip_cycle", reason="not_rebalance_day")
        return

    baseline_cfg = build_baseline_cfg()

    # Strategy target source is currently compute_rebalance_target(); update here if you switch target provider.
    target = compute_rebalance_target(
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        asof=signal_asof,
        cfg=baseline_cfg,
        ranking_history=[],
    )

    selected_symbols = [str(s).strip().upper() for s in target.get("selected_symbols", [])]
    target_weights = {str(k).strip().upper(): float(v) for k, v in dict(target.get("target_weights", {})).items()}
    target_exposure = float(target.get("target_exposure", 0.0))

    if not selected_symbols:
        _log_event(logger, "abort", reason="empty_target_symbols")
        return

    if len(selected_symbols) > cfg.max_positions:
        _log_event(
            logger,
            "target_trimmed",
            requested_positions=len(selected_symbols),
            max_positions=cfg.max_positions,
        )
        selected_symbols = selected_symbols[: cfg.max_positions]

    try:
        positions = broker.get_positions()
    except Exception as exc:
        _log_event(logger, "abort", reason=f"positions_fetch_failed: {exc}")
        return

    position_snapshots = positions_from_alpaca(positions)
    positions_rows = [asdict(p) for p in position_snapshots]

    price_universe = set(selected_symbols) | {p.symbol for p in position_snapshots}
    latest_prices = _fetch_latest_prices(broker, price_universe)

    liquidation_active = bool(cfg.first_run_liquidate_all and not state.get("first_run_liquidation_done", False))

    plan = build_rebalance_plan(
        selected_symbols=selected_symbols,
        target_weights=target_weights,
        target_exposure=target_exposure,
        account_equity=account_equity,
        max_deployment_pct=cfg.max_deployment_pct,
        current_positions=position_snapshots,
        latest_prices=latest_prices,
        min_trade_notional=cfg.min_trade_notional,
        buy_limit_buffer_bps=cfg.buy_limit_buffer_bps,
        sell_limit_buffer_bps=cfg.sell_limit_buffer_bps,
        min_sell_qty=cfg.min_sell_qty,
        max_positions=cfg.max_positions,
        first_run_liquidate_all=liquidation_active,
    )

    orders_rows = _orders_to_rows(plan.orders)

    if plan.abort_reasons:
        for reason in plan.abort_reasons:
            _log_event(logger, "abort", reason=reason)
        summary_payload = {
            "timestamp_et": now_et.isoformat(),
            "dry_run": ctx.dry_run,
            "account_equity": account_equity,
            "buying_power": buying_power,
            "target_symbols": selected_symbols,
            "liquidation_mode": plan.liquidation_mode,
            "abort_reasons": plan.abort_reasons,
        }
        _save_rebalance_summary(cfg, run_id, summary_payload, orders_rows, positions_rows)
        return

    shape_gate = validate_order_plan_shape(orders_rows, cfg.max_order_count)
    if not shape_gate.ok:
        for reason in shape_gate.reasons:
            _log_event(logger, "abort", reason=reason)
        summary_payload = {
            "timestamp_et": now_et.isoformat(),
            "dry_run": ctx.dry_run,
            "account_equity": account_equity,
            "buying_power": buying_power,
            "target_symbols": selected_symbols,
            "liquidation_mode": plan.liquidation_mode,
            "abort_reasons": shape_gate.reasons,
        }
        _save_rebalance_summary(cfg, run_id, summary_payload, orders_rows, positions_rows)
        return

    try:
        open_orders = broker.get_open_orders()
    except Exception as exc:
        _log_event(logger, "abort", reason=f"open_orders_fetch_failed: {exc}")
        return

    filtered_orders, skipped_pending = filter_symbols_already_pending(orders_rows, open_orders)
    for reason in skipped_pending:
        _log_event(logger, "idempotency_skip", reason=reason)

    symbols_for_assets = sorted({str(r["symbol"]).upper() for r in filtered_orders})
    assets: Dict[str, Any] = {}
    for sym in symbols_for_assets:
        try:
            assets[sym] = broker.get_asset(sym)
        except Exception as exc:
            _log_event(logger, "abort", reason=f"asset_lookup_failed_{sym}: {exc}")
            return

    asset_gate = validate_assets_for_orders(filtered_orders, assets)
    if not asset_gate.ok:
        for reason in asset_gate.reasons:
            _log_event(logger, "abort", reason=reason)
        summary_payload = {
            "timestamp_et": now_et.isoformat(),
            "dry_run": ctx.dry_run,
            "account_equity": account_equity,
            "buying_power": buying_power,
            "target_symbols": selected_symbols,
            "liquidation_mode": plan.liquidation_mode,
            "abort_reasons": asset_gate.reasons,
        }
        _save_rebalance_summary(cfg, run_id, summary_payload, filtered_orders, positions_rows)
        return

    _log_event(
        logger,
        "rebalance_plan",
        current_positions=positions_rows,
        target_notional=plan.target_notional_by_symbol,
        order_count=len(filtered_orders),
        orders=filtered_orders,
        liquidation_mode=plan.liquidation_mode,
    )

    summary_payload = {
        "timestamp_et": now_et.isoformat(),
        "dry_run": ctx.dry_run,
        "account_equity": account_equity,
        "buying_power": buying_power,
        "target_symbols": selected_symbols,
        "target_notional": plan.target_notional_by_symbol,
        "liquidation_mode": plan.liquidation_mode,
        "abort_reasons": [],
    }
    _save_rebalance_summary(cfg, run_id, summary_payload, filtered_orders, positions_rows)

    if ctx.dry_run:
        _log_event(logger, "dry_run_complete", submitted_orders=0)
        return

    if not filtered_orders:
        _log_event(logger, "live_complete", submitted_orders=0, reason="no_orders_after_filters")
        return

    sells = [r for r in filtered_orders if str(r["side"]).lower() == "sell"]
    buys = [r for r in filtered_orders if str(r["side"]).lower() == "buy"]
    ordered = sells + buys

    exec_rows = _submit_orders(
        broker=broker,
        logger=logger,
        orders_rows=ordered,
        run_id=run_id,
        tif=cfg.time_in_force,
    )

    exec_fp = cfg.logs_dir / f"live_execution_{run_id}.csv"
    pd.DataFrame(exec_rows).to_csv(exec_fp, index=False)
    _log_event(logger, "execution_results", report_path=str(exec_fp), results=exec_rows)

    state["last_rebalance_date"] = str(today_et)
    state["last_successful_run_ts"] = datetime.now(tz=ET).isoformat()
    state["last_target_symbols"] = selected_symbols
    state["last_mode"] = "dry_run" if ctx.dry_run else "live"
    state["last_cycle_key"] = cycle_key

    if plan.liquidation_mode:
        try:
            time.sleep(2.0)
            remaining_positions = positions_from_alpaca(broker.get_positions())
            nonzero = [p for p in remaining_positions if p.qty > cfg.min_sell_qty]
            state["first_run_liquidation_done"] = len(nonzero) == 0
            _log_event(
                logger,
                "first_run_liquidation_status",
                liquidation_done=state["first_run_liquidation_done"],
                remaining_symbols=[p.symbol for p in nonzero],
            )
        except Exception as exc:
            _log_event(logger, "first_run_liquidation_status_error", error=str(exc))

    save_state(cfg.state_file, state)


if __name__ == "__main__":
    main()
