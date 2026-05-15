from pathlib import Path
from typing import Dict, List, Optional, Union

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
    name = p.stem
    return name.split(".")[0] if "." in name else name


def normalize_stooq_ohlcv(
    df: pd.DataFrame,
    require_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    if "date" in df.columns:
        df = df.set_index(pd.to_datetime(df["date"])).drop(columns=["date"])
    elif "Date" in df.columns:
        df = df.set_index(pd.to_datetime(df["Date"])).drop(columns=["Date"])
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})

    for alias in ["adj close", "adjclose", "adj_close"]:
        if alias in df.columns and "close" not in df.columns:
            df = df.rename(columns={alias: "close"})

    if require_columns is not None:
        missing = [col for col in require_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns {missing}. Have: {list(df.columns)}")

    return df.copy()


def load_stooq_ohlcv_bundle(
    stooq_dir: Union[str, Path] = "data_cache/stooq",
    limit_symbols: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    require_open: bool = False,
) -> Dict[str, pd.DataFrame]:
    files = list_stooq_parquets(stooq_dir, limit=limit_symbols)
    required = ["close", "volume"]
    if require_open:
        required.insert(0, "open")

    symbol_dfs: Dict[str, pd.DataFrame] = {}
    for fp in files:
        sym = _infer_symbol_from_filename(fp)
        df = normalize_stooq_ohlcv(pd.read_parquet(fp), require_columns=required)

        if start:
            df = df.loc[pd.to_datetime(start) :]
        if end:
            df = df.loc[: pd.to_datetime(end)]

        symbol_dfs[sym] = df

    if not symbol_dfs:
        raise ValueError("No usable OHLCV series found.")

    return symbol_dfs


def load_stooq_price_matrix(
    stooq_dir: Union[str, Path] = "data_cache/stooq",
    field_preference: Optional[List[str]] = None,
    limit_symbols: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load close-like price series from per-symbol Stooq parquet files."""
    if field_preference is None:
        field_preference = ["Adj Close", "AdjClose", "adj_close", "close", "Close"]

    files = list_stooq_parquets(stooq_dir, limit=limit_symbols)

    series_list = []
    for fp in files:
        df = pd.read_parquet(fp)

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
