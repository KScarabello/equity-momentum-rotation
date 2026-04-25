from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional


class AlpacaDependencyError(RuntimeError):
    pass


def _import_alpaca() -> Dict[str, Any]:
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetCalendarRequest, GetOrdersRequest, LimitOrderRequest, ClosePositionRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        return {
            "TradingClient": TradingClient,
            "GetCalendarRequest": GetCalendarRequest,
            "GetOrdersRequest": GetOrdersRequest,
            "LimitOrderRequest": LimitOrderRequest,
            "ClosePositionRequest": ClosePositionRequest,
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

    def get_asset(self, symbol: str) -> Any:
        return self._trading.get_asset(symbol)

    def get_positions(self) -> List[Any]:
        return list(self._trading.get_all_positions())

    def get_position(self, symbol: str) -> Any:
        return self._trading.get_open_position(symbol)

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
        qty: Optional[float],
        notional: Optional[float],
        limit_price: float,
        client_order_id: str,
        tif: str = "day",
    ) -> Any:
        if side not in {"buy", "sell"}:
            raise ValueError(f"Unsupported side: {side}")
        if qty is None and notional is None:
            raise ValueError("One of qty or notional must be provided")
        if qty is not None and notional is not None:
            raise ValueError("Only one of qty or notional can be provided")
        if qty is not None and float(qty) <= 0:
            raise ValueError("qty must be positive")
        if notional is not None and float(notional) <= 0:
            raise ValueError("notional must be positive")
        if float(limit_price) <= 0:
            raise ValueError("limit_price must be positive")

        order_side = self._m["OrderSide"].BUY if side == "buy" else self._m["OrderSide"].SELL
        tif_value = self._m["TimeInForce"].DAY if tif.lower() == "day" else self._m["TimeInForce"].GTC

        req_kwargs: Dict[str, Any] = {
            "symbol": symbol,
            "side": order_side,
            "time_in_force": tif_value,
            "limit_price": round(float(limit_price), 4),
            "client_order_id": client_order_id,
        }
        if qty is not None:
            req_kwargs["qty"] = round(float(qty), 6)
        if notional is not None:
            req_kwargs["notional"] = round(float(notional), 2)

        req = self._m["LimitOrderRequest"](
            **req_kwargs,
        )
        return self._trading.submit_order(order_data=req)

    def submit_fractional_buy_notional(
        self,
        symbol: str,
        notional: float,
        limit_price: float,
        client_order_id: str,
        tif: str = "day",
    ) -> Any:
        return self.submit_limit_order(
            symbol=symbol,
            side="buy",
            qty=None,
            notional=notional,
            limit_price=limit_price,
            client_order_id=client_order_id,
            tif=tif,
        )

    def submit_fractional_sell_qty(
        self,
        symbol: str,
        qty: float,
        limit_price: float,
        client_order_id: str,
        tif: str = "day",
    ) -> Any:
        return self.submit_limit_order(
            symbol=symbol,
            side="sell",
            qty=qty,
            notional=None,
            limit_price=limit_price,
            client_order_id=client_order_id,
            tif=tif,
        )

    def close_position(
        self,
        symbol: str,
        qty: float,
        limit_price: float,
        tif: str = "day",
        client_order_id: Optional[str] = None,
    ) -> Any:
        tif_value = self._m["TimeInForce"].DAY if tif.lower() == "day" else self._m["TimeInForce"].GTC
        req = self._m["ClosePositionRequest"](
            qty=round(float(qty), 6),
            limit_price=round(float(limit_price), 4),
            time_in_force=tif_value,
            client_order_id=client_order_id,
        )
        return self._trading.close_position(symbol_or_asset_id=symbol, close_options=req)

    def close_all_positions(self, cancel_orders: bool = False) -> Any:
        return self._trading.close_all_positions(cancel_orders=cancel_orders)

    def get_order(self, order_id: str) -> Any:
        return self._trading.get_order_by_id(order_id)

    def cancel_order(self, order_id: str) -> None:
        self._trading.cancel_order_by_id(order_id)

    def get_open_orders(self) -> List[Any]:
        req = self._m["GetOrdersRequest"](status=self._m["QueryOrderStatus"].OPEN)
        return list(self._trading.get_orders(filter=req))

    def list_open_orders(self) -> List[Any]:
        return self.get_open_orders()

    def cancel_open_orders(self) -> None:
        self._trading.cancel_orders()
