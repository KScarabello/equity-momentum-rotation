from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class GateCheckResult:
    ok: bool
    reasons: List[str]


def within_execution_window(now_et: datetime, start: time, end: time) -> bool:
    t = now_et.time()
    return start <= t <= end


def validate_account_for_trading(account: Any) -> GateCheckResult:
    reasons: List[str] = []

    if account is None:
        return GateCheckResult(ok=False, reasons=["Account payload missing"])

    if not bool(getattr(account, "trading_blocked", False)):
        pass
    else:
        reasons.append("Account is trading_blocked")

    if not bool(getattr(account, "account_blocked", False)):
        pass
    else:
        reasons.append("Account is account_blocked")

    status = str(getattr(account, "status", "")).lower()
    if status not in {"active", "accountstatus.active"} and status:
        reasons.append(f"Unexpected account status={status}")

    return GateCheckResult(ok=len(reasons) == 0, reasons=reasons)


def validate_order_plan_shape(orders: Iterable[Dict[str, Any]], max_order_count: int) -> GateCheckResult:
    reasons: List[str] = []
    rows = list(orders)

    if len(rows) > max_order_count:
        reasons.append(f"Order count {len(rows)} exceeds MAX_ORDER_COUNT={max_order_count}")

    for row in rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        side = str(row.get("side", "")).strip().lower()
        qty = row.get("qty")
        notional = row.get("notional")

        if not symbol:
            reasons.append("Order contains empty symbol")
        if side not in {"buy", "sell"}:
            reasons.append(f"Order {symbol} has invalid side={side}")

        if side == "buy":
            if notional is None or float(notional) <= 0:
                reasons.append(f"Buy order {symbol} missing positive notional")
        if side == "sell":
            if qty is None or float(qty) <= 0:
                reasons.append(f"Sell order {symbol} missing positive qty")

    return GateCheckResult(ok=len(reasons) == 0, reasons=reasons)


def filter_symbols_already_pending(
    orders: Iterable[Dict[str, Any]],
    open_orders: Iterable[Any],
) -> tuple[List[Dict[str, Any]], List[str]]:
    pending_keys = set()
    for o in open_orders:
        sym = str(getattr(o, "symbol", "")).strip().upper()
        side = str(getattr(o, "side", "")).strip().lower()
        status = str(getattr(o, "status", "")).strip().lower()
        if sym and side and status in {"new", "accepted", "pending_new", "partially_filled", "held", "pending_replace"}:
            pending_keys.add((sym, side))

    kept: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for row in orders:
        key = (str(row.get("symbol", "")).strip().upper(), str(row.get("side", "")).strip().lower())
        if key in pending_keys:
            skipped.append(f"Skipped {key[1]} {key[0]} due to existing open order")
            continue
        kept.append(row)

    return kept, skipped


def validate_assets_for_orders(
    orders: Iterable[Dict[str, Any]],
    assets_by_symbol: Dict[str, Any],
) -> GateCheckResult:
    reasons: List[str] = []

    for row in orders:
        symbol = str(row.get("symbol", "")).strip().upper()
        side = str(row.get("side", "")).strip().lower()
        asset = assets_by_symbol.get(symbol)
        if asset is None:
            reasons.append(f"Missing asset metadata for {symbol}")
            continue

        tradable = bool(getattr(asset, "tradable", False))
        fractionable = bool(getattr(asset, "fractionable", False))
        if not tradable:
            reasons.append(f"Asset {symbol} is not tradable")

        if side == "buy" and not fractionable:
            reasons.append(f"Asset {symbol} is not fractionable for notional buy")

    return GateCheckResult(ok=len(reasons) == 0, reasons=reasons)
