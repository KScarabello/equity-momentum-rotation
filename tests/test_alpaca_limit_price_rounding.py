from __future__ import annotations

from types import SimpleNamespace

import pytest

from live.alpaca_client import round_alpaca_limit_price
from live.run_alpaca_live_trader import _submit_orders


class _FakeBroker:
    def __init__(self) -> None:
        self.calls = []

    def submit_fractional_buy_notional(self, *, symbol, notional, limit_price, client_order_id, tif):
        self.calls.append(
            {
                "method": "buy",
                "symbol": symbol,
                "notional": notional,
                "limit_price": limit_price,
                "client_order_id": client_order_id,
                "tif": tif,
            }
        )
        return SimpleNamespace(id="buy-order-1", status="accepted", filled_qty=0.0, filled_avg_price=0.0)

    def submit_fractional_sell_qty(self, *, symbol, qty, limit_price, client_order_id, tif):
        self.calls.append(
            {
                "method": "sell",
                "symbol": symbol,
                "qty": qty,
                "limit_price": limit_price,
                "client_order_id": client_order_id,
                "tif": tif,
            }
        )
        return SimpleNamespace(id="sell-order-1", status="accepted", filled_qty=0.0, filled_avg_price=0.0)

    def get_order(self, order_id):
        if order_id == "buy-order-1":
            return SimpleNamespace(id="buy-order-1", status="accepted", filled_qty=0.0, filled_avg_price=0.0)
        if order_id == "sell-order-1":
            return SimpleNamespace(id="sell-order-1", status="accepted", filled_qty=0.0, filled_avg_price=0.0)
        raise AssertionError(f"Unexpected order_id: {order_id}")


class _FakeLogger:
    def info(self, *args, **kwargs):
        return None


def test_round_alpaca_limit_price_above_one_rounds_to_two_decimals():
    assert round_alpaca_limit_price(443.1827) == pytest.approx(443.18)
    assert round_alpaca_limit_price(905.905) == pytest.approx(905.91)


def test_round_alpaca_limit_price_below_one_rounds_to_four_decimals():
    assert round_alpaca_limit_price(0.98765) == pytest.approx(0.9877)


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan"), float("inf")])
def test_round_alpaca_limit_price_rejects_non_positive_or_non_finite_prices(price):
    with pytest.raises(ValueError):
        round_alpaca_limit_price(price)


def test_submit_orders_uses_rounded_limit_prices(monkeypatch):
    broker = _FakeBroker()
    logger = _FakeLogger()
    monkeypatch.setattr("live.run_alpaca_live_trader.time.sleep", lambda _: None)

    rows = [
        {
            "symbol": "AAPL",
            "side": "buy",
            "notional": 1000.0,
            "limit_price": 443.1827,
            "reason": "new_entry",
        },
        {
            "symbol": "XYZ",
            "side": "sell",
            "qty": 12.5,
            "limit_price": 0.98765,
            "reason": "position_reduce",
        },
    ]

    results = _submit_orders(
        broker=broker,
        logger=logger,
        orders_rows=rows,
        run_id="test123",
        tif="day",
    )

    assert [call["limit_price"] for call in broker.calls] == [443.18, 0.9877]
    assert results[0]["raw_limit_price"] == pytest.approx(443.1827)
    assert results[0]["limit_price"] == pytest.approx(443.18)
    assert results[1]["raw_limit_price"] == pytest.approx(0.98765)
    assert results[1]["limit_price"] == pytest.approx(0.9877)
