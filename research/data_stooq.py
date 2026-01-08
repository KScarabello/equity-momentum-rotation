from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


def list_stooq_parquets(
    stooq_dir: Union[str, Path], limit: Optional[int] = None
) -> List[Path]:
    stooq_dir = Path(stooq_dir)
    if not stooq_dir.exists():
        raise FileNotFoundError(f"Stooq dir not found: {stooq_dir}")

    files = sorted(stooq_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {stooq_dir}")

    if limit is not None:
        files = files[: int(limit)]
    return files


def _infer_symbol_from_filename(p: Path) -> str:
    # e.g. AAPL.US.parquet -> AAPL
    name = p.stem  # AAPL.US
    return name.split(".")[0] if "." in name else name


def load_stooq_price_matrix(
    stooq_dir: Union[str, Path] = "data_cache/stooq",
    field_preference: Optional[List[str]] = None,
    limit_symbols: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads per-symbol Stooq parquet files and returns:
      index = datetime
      columns = symbols
      values = price series (prefers close-like field)
    """
    if field_preference is None:
        # Stooq parquet you showed: Open/High/Low/Close/Volume (no adj close)
        field_preference = ["Adj Close", "AdjClose", "adj_close", "close", "Close"]

    files = list_stooq_parquets(stooq_dir, limit=limit_symbols)

    series_list = []
    for fp in files:
        df = pd.read_parquet(fp)

        # Normalize date index
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        elif "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        else:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

        df = df.sort_index()

        price_col = None
        for c in field_preference:
            if c in df.columns:
                price_col = c
                break
        if price_col is None:
            continue

        sym = _infer_symbol_from_filename(fp)
        s = df[price_col].rename(sym)

        if start:
            s = s.loc[pd.to_datetime(start) :]
        if end:
            s = s.loc[: pd.to_datetime(end)]

        series_list.append(s)

    if not series_list:
        raise ValueError("No usable price series found.")

    prices = pd.concat(series_list, axis=1).sort_index()
    prices = prices.dropna(axis=1, how="all")
    return prices


if __name__ == "__main__":
    prices = load_stooq_price_matrix(limit_symbols=20)
    print(prices.shape)
    print(prices.tail())
