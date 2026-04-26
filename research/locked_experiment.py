from __future__ import annotations

# Demo/example values only — not tuned or optimized.
LOCKED_START_DATE: str = "2015-01-01"
LOCKED_END_DATE: str = "2023-12-31"

LOCKED_SYMBOLS: set[str] = {
    "AAPL", "ABBV", "ADBE", "AMAT", "AMD", "AMGN", "AMZN", "AVGO",
    "BA", "BAC", "BKNG", "BLK", "CAT", "COST", "CRM", "CSCO", "CVX",
    "DE", "DIS", "GE", "GOOG", "GOOGL", "HD", "HON", "IBM", "INTC",
    "INTU", "JPM", "KO", "LIN", "LLY", "LMT", "MA", "MCD", "META",
    "MRK", "MSFT", "MU", "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP",
    "PG", "PM", "QCOM", "SPY", "TMO", "TSLA", "TXN", "UNH", "UPS",
    "V", "WMT", "XOM",
}

MIN_SYMBOL_COUNT: int = 10
