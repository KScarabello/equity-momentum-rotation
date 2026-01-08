# Equity Momentum Rotation (12–1)

This repository contains a research backtest of a classic 12–1 equity momentum
strategy, evaluated on historical U.S. equity data (Stooq) and benchmarked
against SPY buy-and-hold.

## Strategy

- Signal: 12–1 momentum (12-month lookback, 1-month skip)
- Universe: U.S. equities (Stooq parquet cache)
- Portfolio: Top-N ranked equities, equal-weighted
- Rebalance: Every N trading days (biweekly by default)
- Costs: Applied on rebalance days, proportional to turnover
- Risk-off: Optional absolute momentum filter

## Results (2009–2025)

- Strategy CAGR: ~21%
- SPY CAGR: ~14%
- Excess return: ~7% annualized
- Sharpe: Higher than SPY
- Drawdown: Comparable

## Correctness Guarantees

This project includes unit tests that lock:

- Exact momentum math (no indexing drift)
- Rebalance timing (no lookahead)
- Transaction cost application
- Absolute momentum risk-off behavior

Any unintended logic change will fail tests.

## Disclaimer

This is a research project, not investment advice.
