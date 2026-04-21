from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional


class AlpacaDependencyError(RuntimeError):
    pass


def _import_alpaca() -> Dict[str, Any]:
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetCalendarRequest, GetOrdersRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        return {
            "TradingClient": TradingClient,
            "GetCalendarRequest": GetCalendarRequest,
            "GetOrdersRequest": GetOrdersRequest,
            "LimitOrderRequest": LimitOrderRequest,
            "OrderSide": OrderSide,
            "TimeInForce": TimeInForce,
            "QueryOrderStatus": QueryOrderStatus,
            "StockHistoricalDataClient": StockHistoricalDataClient,
            "StockLatestTradeRequest": StockLatestTradeRequest,
        }
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise AlpacaDependencyError(
            "Missing Python dependency for Alpaca client: "
            f"{missing}. Install with: pip install alpaca-py pytz"
        ) from exc
    except Exception as exc:
        raise AlpacaDependencyError(
            "Unable to initialize Alpaca SDK imports. "
            f"Original error: {exc}"
        ) from exc


@dataclass
class AlpacaCredentials:
    api_key: str
    secret_key: str
    base_url: str


class AlpacaBroker:
    def __init__(self, creds: AlpacaCredentials) -> None:
        m = _import_alpaca()
        self._m = m
        self._trading = m["TradingClient"](
            api_key=creds.api_key,
            secret_key=creds.secret_key,
            paper="paper" in creds.base_url.lower(),
            url_override=creds.base_url,
        )
        self._data = m["StockHistoricalDataClient"](
            api_key=creds.api_key,
            secret_key=creds.secret_key,
        )

    def get_account(self) -> Any:
        return self._trading.get_account()

    def get_clock(self) -> Any:
        return self._trading.get_clock()

    def get_positions(self) -> List[Any]:
        return list(self._trading.get_all_positions())

    def get_latest_trade_price(self, symbol: str) -> Optional[float]:
        req = self._m["StockLatestTradeRequest"](symbol_or_symbols=symbol)
        out = self._data.get_stock_latest_trade(req)
        trade = out.get(symbol)
        if trade is None:
            return None
        price = getattr(trade, "price", None)
        return float(price) if price is not None else None

    def get_trading_days(self, start: date, end: date) -> List[date]:
        req = self._m["GetCalendarRequest"](start=start, end=end)
        rows = self._trading.get_calendar(req)
        return [r.date for r in rows]

    def submit_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        limit_price: float,
        client_order_id: str,
    ) -> Any:
        if side not in {"buy", "sell"}:
            raise ValueError(f"Unsupported side: {side}")

        order_side = self._m["OrderSide"].BUY if side == "buy" else self._m["OrderSide"].SELL
        req = self._m["LimitOrderRequest"](
            symbol=symbol,
            qty=round(float(qty), 6),
            side=order_side,
            time_in_force=self._m["TimeInForce"].DAY,
            limit_price=round(float(limit_price), 4),
            client_order_id=client_order_id,
        )
        return self._trading.submit_order(order_data=req)

    def get_order(self, order_id: str) -> Any:
        return self._trading.get_order_by_id(order_id)

    def cancel_order(self, order_id: str) -> None:
        self._trading.cancel_order_by_id(order_id)

    def list_open_orders(self) -> List[Any]:
        req = self._m["GetOrdersRequest"](status=self._m["QueryOrderStatus"].OPEN)
        return list(self._trading.get_orders(filter=req))
