import pandas as pd
from pathlib import Path
import numpy as np

# ---- PATHS ----
STRATEGY_PATH = Path("walk_forward_equity_oos.csv")

STOOQ_DIR = Path("/Users/kimscarabello/Desktop/Repos/QuantApp/data_cache/stooq")
SPY_PATH = STOOQ_DIR / "SPY.US.parquet"
START = "2022-11-07"
END = "2025-12-24"

# ---- LOAD STRATEGY ----
strategy = pd.read_csv(STRATEGY_PATH, index_col=0, parse_dates=True)
strategy = strategy.rename(columns={strategy.columns[0]: "strategy"})

print("\n[DEBUG] strategy columns:", strategy.columns.tolist())
print(strategy.head())

# ---- LOAD SPY ----
spy_df = pd.read_parquet(SPY_PATH)

if "date" in spy_df.columns:
    spy_df["date"] = pd.to_datetime(spy_df["date"])
    spy_df = spy_df.set_index("date")
elif "Date" in spy_df.columns:
    spy_df["Date"] = pd.to_datetime(spy_df["Date"])
    spy_df = spy_df.set_index("Date")

spy_df = spy_df.sort_index()

spy_close = spy_df["Close"] if "Close" in spy_df.columns else spy_df["close"]

# ---- FILTER WINDOW (2022) ----
strategy_2022 = strategy.loc[START:END, "strategy"]
spy_2022 = spy_close.loc[START:END]

# ---- ALIGN ----
combined = pd.concat([strategy_2022, spy_2022], axis=1, sort=False).dropna()
combined.columns = ["strategy", "spy"]
if combined.empty:
    raise ValueError("No overlapping strategy/SPY data in selected range.")

# ---- NORMALIZE ----
combined = combined / combined.iloc[0]

print("\n[DEBUG] strategy_2022 rows:", len(strategy_2022))
print("[DEBUG] spy_2022 rows:", len(spy_2022))
print("[DEBUG] combined rows:", len(combined))

print("[DEBUG] combined start:", combined.index.min())
print("[DEBUG] combined end:", combined.index.max())

print("\n[DEBUG] combined head:")
print(combined.head())

print("\n[DEBUG] combined tail:")
print(combined.tail())

# ---- RETURNS + METRICS ----
returns = combined.pct_change().dropna()


def compute_metrics(r: pd.Series) -> dict:
    if len(r) == 0:
        return {
            "final": np.nan,
            "cagr": np.nan,
            "vol": np.nan,
            "sharpe": np.nan,
            "max_dd": np.nan,
        }

    final = float((1 + r).prod())
    cagr = float(final ** (252 / len(r)) - 1)
    vol = float(r.std() * np.sqrt(252))

    std = float(r.std())
    sharpe = float((r.mean() / std) * np.sqrt(252)) if std > 0 else np.nan

    cumulative = (1 + r).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = float(drawdown.min())

    return {
        "final": final,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


strategy_stats = compute_metrics(returns["strategy"])
spy_stats = compute_metrics(returns["spy"])

# ---- PRINT ----
print("\n=== STRATEGY (2022) ===")
for k, v in strategy_stats.items():
    print(f"{k}: {v:,.4f}")

print("\n=== SPY (2022) ===")
for k, v in spy_stats.items():
    print(f"{k}: {v:,.4f}")

# ---- SAVE ----
combined.to_csv("comparison_strategy_vs_spy_2022.csv")
print("\n[INFO] Saved comparison_strategy_vs_spy_2022.csv")