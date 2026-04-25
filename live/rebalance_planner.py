from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    qty: float
    market_value: float


@dataclass(frozen=True)
class PlannedOrder:
    symbol: str
    side: str
    qty: Optional[float]
    notional: Optional[float]
    limit_price: float
    reason: str
    current_qty: float
    target_qty: float
    current_notional: float
    target_notional: float


@dataclass(frozen=True)
class PlanResult:
    target_notional_by_symbol: Dict[str, float]
    orders: List[PlannedOrder]
    liquidation_mode: bool
    abort_reasons: List[str]


def _to_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def positions_from_alpaca(raw_positions: Iterable[Any]) -> List[PositionSnapshot]:
    out: List[PositionSnapshot] = []
    for p in raw_positions:
        symbol = str(getattr(p, "symbol", "")).strip().upper()
        if not symbol:
            continue
        qty = _to_float(getattr(p, "qty", 0.0))
        market_value = _to_float(getattr(p, "market_value", 0.0))
        out.append(PositionSnapshot(symbol=symbol, qty=qty, market_value=market_value))
    return out


def _build_price(symbol: str, latest_prices: Dict[str, float], abort_reasons: List[str]) -> Optional[float]:
    px = latest_prices.get(symbol)
    if px is None or px <= 0:
        abort_reasons.append(f"Missing or invalid latest price for {symbol}")
        return None
    return float(px)


def _build_limit_price(reference_price: float, side: str, buy_limit_buffer_bps: float, sell_limit_buffer_bps: float) -> float:
    if side == "buy":
        return round(reference_price * (1.0 + buy_limit_buffer_bps / 10000.0), 4)
    return round(reference_price * (1.0 - sell_limit_buffer_bps / 10000.0), 4)


def _normalize_target_weights(selected_symbols: List[str], target_weights: Dict[str, float], max_positions: int) -> tuple[List[str], Dict[str, float], List[str]]:
    abort_reasons: List[str] = []

    symbols_upper = [str(s).strip().upper() for s in selected_symbols if str(s).strip()]
    if len(symbols_upper) != len(set(symbols_upper)):
        abort_reasons.append("Duplicate symbols found in target portfolio")

    if len(symbols_upper) > max_positions:
        symbols_upper = symbols_upper[:max_positions]

    filtered = {s: float(target_weights.get(s, target_weights.get(s.upper(), 0.0))) for s in symbols_upper}
    total = sum(max(0.0, w) for w in filtered.values())
    if total <= 0:
        abort_reasons.append("Target weights are empty or non-positive")
        return symbols_upper, filtered, abort_reasons

    normalized = {s: max(0.0, w) / total for s, w in filtered.items()}
    return symbols_upper, normalized, abort_reasons


def build_rebalance_plan(
    selected_symbols: List[str],
    target_weights: Dict[str, float],
    target_exposure: float,
    account_equity: float,
    max_deployment_pct: float,
    current_positions: List[PositionSnapshot],
    latest_prices: Dict[str, float],
    min_trade_notional: float,
    buy_limit_buffer_bps: float,
    sell_limit_buffer_bps: float,
    min_sell_qty: float,
    max_positions: int,
    first_run_liquidate_all: bool,
) -> PlanResult:
    abort_reasons: List[str] = []

    if account_equity <= 0:
        abort_reasons.append("Account equity must be positive")

    if not (0 < max_deployment_pct <= 1.0):
        abort_reasons.append("Max deployment must be in (0, 1]")

    if target_exposure < 0 or target_exposure > 1.0:
        abort_reasons.append(f"Target exposure out of range: {target_exposure}")

    target_symbols, norm_weights, target_abort = _normalize_target_weights(
        selected_symbols=selected_symbols,
        target_weights=target_weights,
        max_positions=max_positions,
    )
    abort_reasons.extend(target_abort)

    deployment_dollars = float(account_equity) * float(max_deployment_pct)
    intended_total = deployment_dollars * float(target_exposure)
    if intended_total - deployment_dollars > 1e-6:
        abort_reasons.append("Target weights exceed configured deployment cap")

    current_by_symbol = {p.symbol: p for p in current_positions}

    target_notional_by_symbol: Dict[str, float] = {}
    for sym in target_symbols:
        target_notional_by_symbol[sym] = intended_total * norm_weights.get(sym, 0.0)

    orders: List[PlannedOrder] = []

    if first_run_liquidate_all:
        for sym, pos in current_by_symbol.items():
            if pos.qty <= min_sell_qty:
                continue
            px = _build_price(sym, latest_prices, abort_reasons)
            if px is None:
                continue
            orders.append(
                PlannedOrder(
                    symbol=sym,
                    side="sell",
                    qty=pos.qty,
                    notional=None,
                    limit_price=_build_limit_price(
                        reference_price=px,
                        side="sell",
                        buy_limit_buffer_bps=buy_limit_buffer_bps,
                        sell_limit_buffer_bps=sell_limit_buffer_bps,
                    ),
                    reason="first_run_liquidation",
                    current_qty=pos.qty,
                    target_qty=0.0,
                    current_notional=max(0.0, pos.market_value),
                    target_notional=0.0,
                )
            )
        return PlanResult(
            target_notional_by_symbol=target_notional_by_symbol,
            orders=orders,
            liquidation_mode=True,
            abort_reasons=abort_reasons,
        )

    all_symbols = sorted(set(current_by_symbol.keys()) | set(target_notional_by_symbol.keys()))

    for sym in all_symbols:
        pos = current_by_symbol.get(sym)
        current_qty = pos.qty if pos else 0.0
        current_notional = max(0.0, pos.market_value) if pos else 0.0

        px = _build_price(sym, latest_prices, abort_reasons)
        if px is None:
            continue

        if current_notional <= 0.0 and current_qty > 0.0:
            current_notional = current_qty * px

        target_notional = float(target_notional_by_symbol.get(sym, 0.0))
        delta_notional = target_notional - current_notional

        if abs(delta_notional) < min_trade_notional:
            continue

        if delta_notional > 0:
            # Fractional buy path must use notional order values.
            orders.append(
                PlannedOrder(
                    symbol=sym,
                    side="buy",
                    qty=None,
                    notional=round(delta_notional, 2),
                    limit_price=_build_limit_price(
                        reference_price=px,
                        side="buy",
                        buy_limit_buffer_bps=buy_limit_buffer_bps,
                        sell_limit_buffer_bps=sell_limit_buffer_bps,
                    ),
                    reason="new_entry" if current_qty <= 0 else "position_increase",
                    current_qty=current_qty,
                    target_qty=target_notional / px,
                    current_notional=current_notional,
                    target_notional=target_notional,
                )
            )
            continue

        target_qty = target_notional / px if px > 0 else 0.0
        if target_notional <= 0:
            sell_qty = current_qty
        else:
            sell_qty = max(0.0, current_qty - target_qty)

        if sell_qty <= min_sell_qty:
            continue

        orders.append(
            PlannedOrder(
                symbol=sym,
                side="sell",
                qty=round(sell_qty, 6),
                notional=None,
                limit_price=_build_limit_price(
                    reference_price=px,
                    side="sell",
                    buy_limit_buffer_bps=buy_limit_buffer_bps,
                    sell_limit_buffer_bps=sell_limit_buffer_bps,
                ),
                reason="full_exit" if target_notional <= 0 else "position_reduce",
                current_qty=current_qty,
                target_qty=target_qty,
                current_notional=current_notional,
                target_notional=target_notional,
            )
        )

    return PlanResult(
        target_notional_by_symbol=target_notional_by_symbol,
        orders=orders,
        liquidation_mode=False,
        abort_reasons=abort_reasons,
    )
