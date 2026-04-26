from __future__ import annotations

import pandas as pd

from research.data_stooq import load_stooq_price_matrix
from research.locked_experiment import (
    LOCKED_START_DATE,
    LOCKED_END_DATE,
    LOCKED_SYMBOLS,
    MIN_SYMBOL_COUNT,
)
from research.backtest_v0 import backtest_rotation_v0


def _compute_metrics_from_prices(price_series: pd.Series) -> dict:
    price_series = price_series.dropna().sort_index()
    if len(price_series) < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "final_equity": 1.0}

    rets = price_series.pct_change().fillna(0.0)
    equity = (1.0 + rets).cumprod()

    running_max = equity.cummax()
    dd = (equity - running_max) / running_max

    years = len(equity) / 252.0
    cagr = (float(equity.iloc[-1]) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    std = float(rets.std())
    sharpe = float(rets.mean() / std * (252.0**0.5)) if std > 0 else 0.0

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
        "final_equity": float(equity.iloc[-1]),
        "equity_curve": equity,
    }


def main() -> None:
    from research.locked_experiment import (
        LOCKED_START_DATE,
        LOCKED_END_DATE,
        LOCKED_SYMBOLS,
        MIN_SYMBOL_COUNT,
    )

    prices_all = load_stooq_price_matrix(
        stooq_dir="data_cache/stooq",
        start=LOCKED_START_DATE,
        end=LOCKED_END_DATE,
    )

    prices_all = prices_all.sort_index().ffill(limit=3)

    # --- Apply configured universe ---
    present = set(prices_all.columns)
    required = LOCKED_SYMBOLS
    missing = required - present

    if missing:
        raise RuntimeError(f"Missing required symbols: {sorted(missing)}")

    prices_all = prices_all[list(required)].copy()

    if prices_all.shape[1] < MIN_SYMBOL_COUNT:
        raise RuntimeError("Too few symbols after universe lock")

    # --- Separate SPY for benchmark ---
    spy = prices_all["SPY"].copy()
    prices = prices_all.drop(columns=["SPY"])

    print("Locked universe size (ex SPY):", prices.shape[1])
    print(
        "Locked date range:",
        prices.index.min().date(),
        "->",
        prices.index.max().date(),
    )

    # --- Strategy backtest ---
    out = backtest_rotation_v0(
        prices,
        config_path="config/alpha_v1.yaml",
        cost_bps=10,
    )
    m = out["metrics"]

    # --- Benchmark (align to strategy dates) ---
    spy_aligned = spy.reindex(out["equity_curve"].index).ffill()
    spy_metrics = _compute_metrics_from_prices(spy_aligned)

    print("=" * 60)
    print("STRATEGY RESULTS (REAL STOOQ DATA)")
    print("=" * 60)
    print(f"CAGR:                         {m['cagr']*100:.2f}%")
    print(f"Sharpe Ratio:                 {m['sharpe']:.2f}")
    print(f"Max Drawdown:                 {m['max_drawdown']*100:.2f}%")
    print(f"Avg Turnover per Rebalance:   {m['avg_turnover_per_rebalance']*100:.2f}%")
    print("=" * 60)
    print(f"Final Equity:                 ${out['equity_curve'].iloc[-1]:.2f}")
    print(f"Number of Rebalances:         {len(out['holdings_history'])}")
    print("=" * 60)

    print("=" * 60)
    print("BENCHMARK: SPY BUY & HOLD (STOOQ)")
    print("=" * 60)
    print(f"CAGR:                         {spy_metrics['cagr']*100:.2f}%")
    print(f"Sharpe Ratio:                 {spy_metrics['sharpe']:.2f}")
    print(f"Max Drawdown:                 {spy_metrics['max_drawdown']*100:.2f}%")
    print("=" * 60)
    print(f"Final Equity:                 ${spy_metrics['final_equity']:.2f}")
    print("=" * 60)

    excess_cagr = m["cagr"] - spy_metrics["cagr"]
    print("=" * 60)
    print("QUICK COMPARISON")
    print("=" * 60)
    print(f"Strategy CAGR - benchmark CAGR: {excess_cagr*100:.2f}%")
    print("Note: This is not a full alpha model (no factor regression),")
    print("but it tells us whether we're plausibly beating SPY net of costs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
