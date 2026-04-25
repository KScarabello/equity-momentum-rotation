from __future__ import annotations

from io import StringIO
from pathlib import Path
import os
import socket
import time
from typing import Iterable, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd


SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA", "LLY",
    "JPM", "V", "WMT", "XOM", "MA", "UNH", "PG", "COST", "HD", "NFLX", "AMD", "CRM",
    "BAC", "KO", "PEP", "ADBE", "CSCO", "ORCL", "INTC", "QCOM", "TXN", "AMAT", "IBM",
    "DIS", "NKE", "MCD", "PM", "GE", "CAT", "BA", "HON", "UPS", "LMT", "BLK", "NOW",
    "TMO", "DE", "CVX", "MRK", "ABBV", "LIN", "BKNG", "MU", "SPY",
]

OUT = Path("data_cache/stooq")

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


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


def _build_stooq_url(symbol: str, api_key: Optional[str]) -> str:
    stooq_sym = f"{symbol.lower()}.us"
    base = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
    if api_key:
        return f"{base}&apikey={api_key}"
    return base


def _response_looks_like_csv(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    header = lines[0].lower().replace(" ", "")
    expected = "date,open,high,low,close,volume"
    return header == expected


def _validate_csv_response_or_raise(text: str, symbol: str) -> None:
    preview = text[:500]
    print(f"PREVIEW {symbol}: {preview!r}")
    if _response_looks_like_csv(text):
        return

    snippet = "\n".join(text.splitlines()[:5])
    raise ValueError(
        f"Non-CSV or unexpected response for {symbol}. "
        f"Expected header Date,Open,High,Low,Close,Volume. First lines:\n{snippet}"
    )


def _fetch_text_with_retries(
    symbol: str,
    api_key: Optional[str],
    attempts: int = 4,
    timeout_seconds: int = 20,
) -> str:
    url = _build_stooq_url(symbol=symbol, api_key=api_key)
    last_exc: Optional[Exception] = None

    for attempt in range(1, attempts + 1):
        try:
            req = Request(url, headers={"User-Agent": "equity-momentum-rotation/1.0"})
            with urlopen(req, timeout=timeout_seconds) as response:
                raw = response.read()
            text = raw.decode("utf-8", errors="replace")
            _validate_csv_response_or_raise(text=text, symbol=symbol)
            return text
        except (URLError, TimeoutError, socket.timeout, ValueError) as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            backoff = min(8.0, 0.75 * (2 ** (attempt - 1)))
            print(
                f"RETRY {symbol} attempt={attempt}/{attempts} after error={type(exc).__name__}: {exc}. "
                f"Sleeping {backoff:.2f}s"
            )
            time.sleep(backoff)

    raise RuntimeError(f"Failed to fetch valid CSV for {symbol} after {attempts} attempts: {last_exc}")


def _parse_stooq_csv(text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(text))
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if df.empty:
        raise ValueError("CSV payload is empty")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    if df.empty:
        raise ValueError("Parsed DataFrame is empty after Date index normalization")
    return df


def update_symbol(symbol: str, out_dir: Path, api_key: Optional[str]) -> tuple[bool, str]:
    out_path = out_dir / f"{symbol}.US.parquet"
    text = _fetch_text_with_retries(symbol=symbol, api_key=api_key)
    df = _parse_stooq_csv(text)

    # Write only after full validation so existing files are never clobbered by bad responses.
    df.to_parquet(out_path)
    return True, f"WROTE {out_path.name} through {df.index.max().date()}"


def latest_date_across_cache(out_dir: Path) -> Optional[pd.Timestamp]:
    latest: Optional[pd.Timestamp] = None
    for fp in sorted(out_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(fp)
            if df.empty:
                continue
            idx = pd.to_datetime(df.index)
            if len(idx) == 0:
                continue
            candidate = pd.Timestamp(idx.max())
            if latest is None or candidate > latest:
                latest = candidate
        except Exception:
            continue
    return latest


def update_cache(symbols: Iterable[str] = SYMBOLS, out_dir: Path = OUT) -> int:
    _load_local_env(Path(".env"))
    api_key = os.getenv("STOOQ_API_KEY", "").strip() or None

    out_dir.mkdir(parents=True, exist_ok=True)

    ok_symbols: list[str] = []
    failed_symbols: list[str] = []

    for sym in symbols:
        try:
            ok, msg = update_symbol(symbol=sym, out_dir=out_dir, api_key=api_key)
            if ok:
                ok_symbols.append(sym)
                print(msg)
        except Exception as exc:
            failed_symbols.append(sym)
            print(f"FAIL {sym}: {exc}")

        time.sleep(1.2)

    latest = latest_date_across_cache(out_dir)
    print("\n=== STOOQ CACHE UPDATE SUMMARY ===")
    print(f"successful symbols ({len(ok_symbols)}): {ok_symbols}")
    print(f"failed symbols ({len(failed_symbols)}): {failed_symbols}")
    print(f"latest date across cache: {latest.date() if latest is not None else 'n/a'}")

    if latest is not None:
        age_days = (pd.Timestamp.now(tz=None).normalize() - latest.normalize()).days
        if age_days > 5:
            print(f"WARNING: latest cache date is stale by {age_days} calendar days")

    return 0 if not failed_symbols else 1


def main() -> None:
    raise SystemExit(update_cache())


if __name__ == "__main__":
    main()
