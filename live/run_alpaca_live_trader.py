#!/usr/bin/env python3
"""
Production-safe Alpaca live runner for the validated baseline momentum strategy.

Run:
    python3 -m live.run_alpaca_live_trader --dry-run --verbose
    python3 -m live.run_alpaca_live_trader --live --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config.live_trading_config import LiveTradingConfig, build_baseline_cfg
from live.alpaca_client import AlpacaBroker, AlpacaCredentials, AlpacaDependencyError
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
    """Load .env values into process environment without overriding existing vars."""
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


@dataclass
class RuntimeContext:
    run_id: str
    logger: logging.Logger
    config: LiveTradingConfig
    dry_run: bool
    force: bool
    verbose: bool


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Alpaca live trader for baseline momentum strategy")
    p.add_argument("--dry-run", action="store_true", help="Compute targets/orders but do not place orders")
    p.add_argument("--live", action="store_true", help="Enable live order placement (must be explicit)")
    p.add_argument("--force", action="store_true", help="Bypass preferred execution window guardrail")
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

    logger.info("Logging to %s", log_fp)
    return logger


def _load_creds_from_env() -> AlpacaCredentials:
    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()
    base_url = os.getenv("ALPACA_BASE_URL", "").strip()

    missing = [
        name
        for name, value in [
            ("ALPACA_API_KEY", api_key),
            ("ALPACA_SECRET_KEY", secret_key),
            ("ALPACA_BASE_URL", base_url),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return AlpacaCredentials(api_key=api_key, secret_key=secret_key, base_url=base_url)


def _to_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _within_execution_window(now_et: datetime, cfg: LiveTradingConfig) -> bool:
    t = now_et.time()
    return cfg.execution_window_start <= t <= cfg.execution_window_end


def _load_strategy_data(logger: logging.Logger) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    logger.info("Loading strategy inputs from cache")
    symbols = build_universe_from_stooq(STOOQ_DIR)
    logger.info("Universe loaded from cache: %d symbols", len(symbols))

    market_df = fetch_ohlcv("SPY")
    market_df = _ensure_datetime_index(_normalize_cols(market_df))

    symbol_dfs: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            symbol_dfs[sym] = _ensure_datetime_index(_normalize_cols(fetch_ohlcv(sym)))
        except Exception as exc:
            logger.warning("Symbol load failed: %s (%s)", sym, exc)

    if not symbol_dfs:
        raise RuntimeError("No symbol data loaded successfully.")

    logger.info("Loaded OHLCV for %d symbols", len(symbol_dfs))
    return symbol_dfs, market_df


def _get_signal_asof_date(market_df: pd.DataFrame, today_et: date) -> pd.Timestamp:
    cutoff = pd.Timestamp(today_et) - pd.Timedelta(days=1)
    hist = market_df.index[market_df.index <= cutoff]
    if len(hist) == 0:
        raise RuntimeError("No completed market history is available for signal generation.")
    return pd.Timestamp(hist.max())


def _build_rebalance_calendar(
    broker: AlpacaBroker,
    cfg: LiveTradingConfig,
    today_et: date,
) -> pd.DatetimeIndex:
    start = today_et - timedelta(days=cfg.rebalance_calendar_days_back)
    days = broker.get_trading_days(start=start, end=today_et)
    if not days:
        raise RuntimeError("No Alpaca trading calendar days returned.")

    cal = pd.DatetimeIndex(pd.to_datetime(days)).sort_values()
    rebals = _week_rebalance_dates(cal, weekday=0)
    if len(rebals) == 0:
        return pd.DatetimeIndex([cal[-1]])

    baseline_cfg = build_baseline_cfg()
    if baseline_cfg.rebalance_interval_weeks > 1:
        rebals = rebals[:: baseline_cfg.rebalance_interval_weeks]
    return pd.DatetimeIndex(rebals)


def _build_target_table(
    target: Dict[str, Any],
    deployment_capital: float,
    prices: Dict[str, float],
) -> pd.DataFrame:
    rows = []
    for sym in target["selected_symbols"]:
        w = float(target["target_weights"].get(sym, 0.0))
        port_w = float(target["target_exposure"] * w)
        px = prices.get(sym)
        if px is None or not np.isfinite(px) or px <= 0:
            raise RuntimeError(f"Missing/invalid price for target symbol {sym}")
        target_dollar = deployment_capital * port_w
        target_shares = target_dollar / px
        rows.append(
            {
                "symbol": sym,
                "target_weight": port_w,
                "target_dollar": target_dollar,
                "reference_price": px,
                "target_shares": target_shares,
            }
        )
    return pd.DataFrame(rows).sort_values("target_weight", ascending=False).reset_index(drop=True)


def _current_positions_df(positions: Iterable[Any]) -> pd.DataFrame:
    rows = []
    for p in positions:
        rows.append(
            {
                "symbol": str(getattr(p, "symbol", "")),
                "current_shares": _to_float(getattr(p, "qty", 0.0)),
                "market_value": _to_float(getattr(p, "market_value", 0.0)),
                "avg_entry_price": _to_float(getattr(p, "avg_entry_price", 0.0)),
                "side": str(getattr(p, "side", "long")),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["symbol", "current_shares", "market_value", "avg_entry_price", "side"])
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def _build_order_plan(
    current_df: pd.DataFrame,
    target_df: pd.DataFrame,
    prices: Dict[str, float],
    cfg: LiveTradingConfig,
) -> pd.DataFrame:
    cur = {r.symbol: float(r.current_shares) for r in current_df.itertuples(index=False)}
    tgt = {r.symbol: float(r.target_shares) for r in target_df.itertuples(index=False)}

    all_symbols = sorted(set(cur.keys()) | set(tgt.keys()))
    rows = []

    for sym in all_symbols:
        current_shares = float(cur.get(sym, 0.0))
        target_shares = float(tgt.get(sym, 0.0))
        delta = target_shares - current_shares
        px = prices.get(sym)
        if px is None or not np.isfinite(px) or px <= 0:
            continue

        notional = abs(delta) * px
        if abs(delta) < cfg.min_share_delta or notional < cfg.min_notional_trade:
            continue

        side = "buy" if delta > 0 else "sell"
        limit_price = px * (cfg.buy_limit_buffer if side == "buy" else cfg.sell_limit_buffer)

        if current_shares == 0 and target_shares > 0:
            reason = "new position"
        elif current_shares > 0 and target_shares == 0:
            reason = "exit"
        else:
            reason = "rebalance resize"

        rows.append(
            {
                "symbol": sym,
                "side": side,
                "current_shares": current_shares,
                "target_shares": target_shares,
                "delta_shares": delta,
                "reference_price": px,
                "limit_price": limit_price,
                "estimated_notional": notional,
                "reason": reason,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "side",
                "current_shares",
                "target_shares",
                "delta_shares",
                "reference_price",
                "limit_price",
                "estimated_notional",
                "reason",
            ]
        )

    orders = pd.DataFrame(rows)
    orders["side_rank"] = np.where(orders["side"] == "sell", 0, 1)
    orders = orders.sort_values(["side_rank", "estimated_notional"], ascending=[True, False]).drop(columns=["side_rank"])
    return orders.reset_index(drop=True)


def _log_table(logger: logging.Logger, title: str, df: pd.DataFrame) -> None:
    logger.info("=== %s ===", title)
    if df.empty:
        logger.info("(empty)")
    else:
        logger.info("\n%s", df.to_string(index=False))


def _checklist_item(name: str, ok: bool) -> str:
    return f"- [{'OK' if ok else 'FAIL'}] {name}"


def _submit_orders(
    broker: AlpacaBroker,
    orders_df: pd.DataFrame,
    run_id: str,
    cfg: LiveTradingConfig,
    logger: logging.Logger,
) -> pd.DataFrame:
    results: list[dict[str, Any]] = []

    for idx, row in enumerate(orders_df.itertuples(index=False), start=1):
        cid = f"mom-{run_id}-{idx}"
        symbol = str(row.symbol)
        side = str(row.side)
        qty = abs(float(row.delta_shares))
        limit_price = float(row.limit_price)

        logger.info("Submitting %s %s qty=%.6f @ %.4f", side.upper(), symbol, qty, limit_price)

        order = broker.submit_limit_order(
            symbol=symbol,
            side=side,
            qty=qty,
            limit_price=limit_price,
            client_order_id=cid,
        )

        order_id = str(getattr(order, "id", ""))
        time.sleep(max(cfg.retry_wait_seconds, 1))
        latest = broker.get_order(order_id)
        status = str(getattr(latest, "status", "unknown"))

        retried = False
        if cfg.retry_enabled and status.lower() in {"new", "accepted", "pending_new", "partially_filled"}:
            retried = True
            try:
                broker.cancel_order(order_id)
            except Exception as exc:
                logger.warning("Cancel failed for %s (%s): %s", symbol, order_id, exc)

            widened = float(
                row.reference_price
                * (cfg.retry_buy_limit_buffer if side == "buy" else cfg.retry_sell_limit_buffer)
            )
            cid2 = f"mom-{run_id}-{idx}-r1"
            logger.info("Retrying %s %s qty=%.6f @ %.4f", side.upper(), symbol, qty, widened)
            order2 = broker.submit_limit_order(
                symbol=symbol,
                side=side,
                qty=qty,
                limit_price=widened,
                client_order_id=cid2,
            )
            order2_id = str(getattr(order2, "id", ""))
            time.sleep(max(cfg.retry_wait_seconds, 1))
            latest = broker.get_order(order2_id)
            status = str(getattr(latest, "status", "unknown"))
            order_id = order2_id
            limit_price = widened

        results.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "limit_price": limit_price,
                "order_id": order_id,
                "status": status,
                "retried": retried,
            }
        )

    return pd.DataFrame(results)


def main() -> None:
    args = _parse_args()

    dry_run = True
    if args.live and not args.dry_run:
        dry_run = False

    cfg = LiveTradingConfig()
    run_id = datetime.now(tz=ET).strftime("%Y%m%d_%H%M%S")
    logger = _setup_logging(cfg=cfg, run_id=run_id, verbose=args.verbose)
    ctx = RuntimeContext(run_id=run_id, logger=logger, config=cfg, dry_run=dry_run, force=args.force, verbose=args.verbose)

    try:
        creds = _load_creds_from_env()
        broker = AlpacaBroker(creds)
    except AlpacaDependencyError as exc:
        logger.error("%s", exc)
        sys.exit(2)
    except Exception as exc:
        logger.error("Client initialization failed: %s", exc)
        sys.exit(2)

    account = broker.get_account()
    clock = broker.get_clock()
    now_et = getattr(clock, "timestamp", datetime.now(tz=ET))
    if now_et.tzinfo is None:
        now_et = now_et.replace(tzinfo=ET)
    now_et = now_et.astimezone(ET)
    today_et = now_et.date()

    account_equity = _to_float(getattr(account, "equity", 0.0))
    account_cash = _to_float(getattr(account, "cash", 0.0))
    buying_power = _to_float(getattr(account, "buying_power", 0.0))

    account_mode = "paper" if "paper" in creds.base_url.lower() else "live"

    logger.info("=== ACCOUNT SUMMARY ===")
    logger.info("mode=%s dry_run=%s force=%s", account_mode, ctx.dry_run, ctx.force)
    logger.info("equity=%.2f cash=%.2f buying_power=%.2f", account_equity, account_cash, buying_power)

    window_ok = _within_execution_window(now_et, cfg)
    market_open = bool(getattr(clock, "is_open", False))

    cal_start = today_et - timedelta(days=cfg.rebalance_calendar_days_back)
    trading_days = broker.get_trading_days(cal_start, today_et)
    trading_day_set = {pd.Timestamp(d).date() for d in trading_days}
    is_trading_day = today_et in trading_day_set

    rebals = _build_rebalance_calendar(broker, cfg, today_et)
    is_rebalance_day = pd.Timestamp(today_et) in rebals

    state = load_state(cfg.state_file)
    duplicate_run = state.get("last_rebalance_date") == str(today_et)

    symbol_dfs, market_df = _load_strategy_data(logger)
    signal_asof = _get_signal_asof_date(market_df, today_et=today_et)
    signal_staleness_days = (today_et - signal_asof.date()).days
    data_fresh = signal_staleness_days <= cfg.max_signal_staleness_days

    checklist = [
        _checklist_item("credentials loaded", True),
        _checklist_item("account mode confirmed", account_mode in {"paper", "live"}),
        _checklist_item("market open", market_open),
        _checklist_item("execution window valid (or --force)", window_ok or ctx.force),
        _checklist_item("rebalance day confirmed", is_rebalance_day),
        _checklist_item("data fresh", data_fresh),
        _checklist_item("no duplicate execution today", (not duplicate_run) or ctx.force),
        _checklist_item("run mode explicit", (ctx.dry_run or args.live)),
    ]

    logger.info("=== LIVE-READINESS CHECKLIST ===")
    for line in checklist:
        logger.info(line)

    if not is_trading_day:
        logger.warning("Rebalance skipped: non-trading day")
        return

    if not market_open:
        logger.warning("Rebalance skipped: market closed")
        return

    if not window_ok and not ctx.force:
        logger.warning("Execution blocked: outside window (%s-%s ET)", cfg.execution_window_start, cfg.execution_window_end)
        return

    if duplicate_run and not ctx.force:
        logger.warning("Execution blocked: duplicate run detected")
        return

    if not data_fresh:
        logger.error("Freshness check failed: signal data is %d days old", signal_staleness_days)
        return

    if not is_rebalance_day:
        logger.info("=== STRATEGY SUMMARY ===")
        logger.info("Rebalance skipped: cadence not due")
        positions_df = _current_positions_df(broker.get_positions())
        _log_table(logger, "CURRENT HOLDINGS", positions_df)
        return

    baseline_cfg = build_baseline_cfg()
    target = compute_rebalance_target(
        symbol_dfs=symbol_dfs,
        market_df=market_df,
        asof=signal_asof,
        cfg=baseline_cfg,
        ranking_history=[],
    )

    selected_symbols = list(target["selected_symbols"])
    target_weights = dict(target["target_weights"])
    target_exposure = float(target["target_exposure"])

    logger.info("=== STRATEGY SUMMARY ===")
    logger.info("rebalance_day=%s signal_asof=%s risk_on=%s target_exposure=%.2f", is_rebalance_day, signal_asof.date(), target["risk_on"], target_exposure)
    logger.info("selected_symbols=%s", "|".join(selected_symbols))

    if not selected_symbols:
        logger.error("Target generation failed: empty selection")
        return

    expected_weight = target_exposure / len(selected_symbols)
    for sym, w in target_weights.items():
        port_weight = target_exposure * float(w)
        if abs(port_weight - expected_weight) > cfg.max_position_weight_tolerance:
            raise RuntimeError(
                f"Unexpected target weight distortion for {sym}: {port_weight:.4f} vs expected {expected_weight:.4f}"
            )

    positions_df = _current_positions_df(broker.get_positions())
    _log_table(logger, "CURRENT HOLDINGS", positions_df)

    price_symbols = sorted(set(selected_symbols) | set(positions_df["symbol"].tolist()))
    prices: Dict[str, float] = {}
    missing_price_symbols: list[str] = []
    for sym in price_symbols:
        px = broker.get_latest_trade_price(sym)
        if px is None or not np.isfinite(px) or px <= 0:
            missing_price_symbols.append(sym)
            continue
        prices[sym] = float(px)

    if any(sym in selected_symbols for sym in missing_price_symbols):
        logger.error("Missing market prices for target symbols: %s", ", ".join(sorted(set(missing_price_symbols) & set(selected_symbols))))
        return

    deployment_capital = account_equity * cfg.max_deployment_fraction
    target_df = _build_target_table(target=target, deployment_capital=deployment_capital, prices=prices)
    _log_table(logger, "TARGET PORTFOLIO", target_df)

    orders_df = _build_order_plan(current_df=positions_df, target_df=target_df, prices=prices, cfg=cfg)
    _log_table(logger, "PROPOSED ORDERS", orders_df)

    total_buy_notional = float(orders_df.loc[orders_df["side"] == "buy", "estimated_notional"].sum()) if not orders_df.empty else 0.0
    total_sell_notional = float(orders_df.loc[orders_df["side"] == "sell", "estimated_notional"].sum()) if not orders_df.empty else 0.0
    est_post_trade_cash = account_cash + total_sell_notional - total_buy_notional

    logger.info("Order sanity: count=%d buy_notional=%.2f sell_notional=%.2f est_post_trade_cash=%.2f", len(orders_df), total_buy_notional, total_sell_notional, est_post_trade_cash)

    if (not cfg.allow_margin) and est_post_trade_cash < -1e-6:
        logger.error("Estimated post-trade cash would be negative (margin disabled); aborting.")
        return

    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    targets_fp = cfg.logs_dir / f"live_targets_{run_id}.csv"
    orders_fp = cfg.logs_dir / f"live_orders_{run_id}.csv"
    target_df.to_csv(targets_fp, index=False)
    orders_df.to_csv(orders_fp, index=False)
    logger.info("Saved targets to %s", targets_fp)
    logger.info("Saved proposed orders to %s", orders_fp)

    if ctx.dry_run:
        logger.info("=== EXECUTION RESULT ===")
        logger.info("Dry run complete: no orders submitted")
        return

    if account_mode == "live" and not args.live:
        logger.error("Live account detected but --live was not provided; aborting.")
        return

    if orders_df.empty:
        logger.info("=== EXECUTION RESULT ===")
        logger.info("No orders to place after thresholds.")
        return

    logger.info("=== EXECUTION RESULT ===")
    logger.info("Submitting %d orders (sells first, then buys)...", len(orders_df))

    exec_df = _submit_orders(
        broker=broker,
        orders_df=orders_df,
        run_id=run_id,
        cfg=cfg,
        logger=logger,
    )

    exec_fp = cfg.logs_dir / f"live_execution_{run_id}.csv"
    exec_df.to_csv(exec_fp, index=False)
    _log_table(logger, "ORDER STATUS", exec_df)
    logger.info("Saved execution report to %s", exec_fp)

    state["last_rebalance_date"] = str(today_et)
    state["last_successful_run_ts"] = datetime.now(tz=ET).isoformat()
    state["last_target_symbols"] = selected_symbols
    state["last_mode"] = account_mode
    save_state(cfg.state_file, state)
    logger.info("Updated state file: %s", cfg.state_file)


if __name__ == "__main__":
    main()
