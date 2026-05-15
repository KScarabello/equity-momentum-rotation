#!/usr/bin/env python3
"""Analyze weighted next-open gap impact from an existing diagnostics CSV.

Run:
    python -m research.analyze_execution_gap_impact
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("research/execution_timing_gap_diagnostics.csv")
DEFAULT_OUTPUT = Path("research/execution_timing_gap_analysis_weighted.csv")
REQUIRED_COLUMNS = {
    "overnight_gap_return",
    "action",
    "trade_weight_change",
    "traded_notional_pct",
}


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "Missing required columns in diagnostics CSV: "
            + ", ".join(missing)
            + ". Required columns are: "
            + ", ".join(sorted(REQUIRED_COLUMNS))
        )


def compute_event_impacts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _validate_required_columns(out)

    optional_columns = [
        "close_t",
        "execution_price",
        "pre_trade_weight",
        "post_trade_weight",
        "target_weight",
        "executed_weight",
        "decision_date",
        "execution_date",
        "symbol",
    ]
    for c in optional_columns:
        if c not in out.columns:
            out[c] = np.nan

    for c in [
        "overnight_gap_return",
        "trade_weight_change",
        "traded_notional_pct",
        "close_t",
        "execution_price",
        "pre_trade_weight",
        "post_trade_weight",
        "target_weight",
        "executed_weight",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in ["decision_date", "execution_date"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")

    out["action"] = out["action"].astype(str).str.lower().str.strip()

    gap = out["overnight_gap_return"].fillna(0.0)
    twc = out["trade_weight_change"].fillna(0.0)
    tnp = out["traded_notional_pct"].fillna(0.0)

    positive_change = twc.clip(lower=0.0)
    negative_change_abs = (-twc).clip(lower=0.0)

    signed_impact = np.zeros(len(out), dtype=float)

    bought = out["action"].eq("bought")
    sold = out["action"].eq("sold")
    held = out["action"].eq("held")
    skipped = out["action"].eq("skipped")
    unknown = ~(bought | sold | held | skipped)

    signed_impact[bought] = (-gap[bought] * positive_change[bought]).to_numpy()
    signed_impact[sold] = (+gap[sold] * negative_change_abs[sold]).to_numpy()

    # Conservative handling for held rows that appear to have a trade footprint.
    held_with_trade = held & (tnp > 0) & (twc != 0)
    out["impact_note"] = ""
    out.loc[held_with_trade, "impact_note"] = (
        "held row with non-zero traded_notional_pct/trade_weight_change; impact set to 0 conservatively"
    )
    out.loc[unknown, "impact_note"] = "unknown action; impact set to 0"

    out["signed_weighted_gap_impact"] = signed_impact
    out["abs_signed_weighted_gap_impact"] = np.abs(signed_impact)

    out["impact_direction"] = "neutral"
    out.loc[out["signed_weighted_gap_impact"] > 0, "impact_direction"] = "favorable"
    out.loc[out["signed_weighted_gap_impact"] < 0, "impact_direction"] = "unfavorable"

    return out


def _build_overall_summary(events: pd.DataFrame) -> pd.DataFrame:
    row = {
        "section": "overall_summary",
        "total_events": int(len(events)),
        "total_signed_weighted_gap_impact": float(events["signed_weighted_gap_impact"].sum()),
        "avg_signed_weighted_gap_impact": float(events["signed_weighted_gap_impact"].mean())
        if len(events)
        else np.nan,
        "favorable_event_count": int((events["impact_direction"] == "favorable").sum()),
        "unfavorable_event_count": int((events["impact_direction"] == "unfavorable").sum()),
        "neutral_event_count": int((events["impact_direction"] == "neutral").sum()),
    }
    return pd.DataFrame([row])


def _build_action_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for action in ["bought", "sold", "held", "skipped"]:
        s = events[events["action"] == action]
        rows.append(
            {
                "section": "action_summary",
                "action": action,
                "count": int(len(s)),
                "total_signed_weighted_gap_impact": float(s["signed_weighted_gap_impact"].sum())
                if len(s)
                else 0.0,
                "avg_signed_weighted_gap_impact": float(s["signed_weighted_gap_impact"].mean())
                if len(s)
                else np.nan,
                "avg_overnight_gap_return": float(s["overnight_gap_return"].mean()) if len(s) else np.nan,
                "total_traded_notional_pct": float(s["traded_notional_pct"].sum()) if len(s) else 0.0,
                "avg_traded_notional_pct": float(s["traded_notional_pct"].mean()) if len(s) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _build_top_events(events: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    favorable = events.sort_values("signed_weighted_gap_impact", ascending=False).head(top_n).copy()
    unfavorable = events.sort_values("signed_weighted_gap_impact", ascending=True).head(top_n).copy()

    favorable["section"] = "top_events_weighted"
    favorable["favorability"] = "favorable"
    favorable["rank"] = np.arange(1, len(favorable) + 1)

    unfavorable["section"] = "top_events_weighted"
    unfavorable["favorability"] = "unfavorable"
    unfavorable["rank"] = np.arange(1, len(unfavorable) + 1)

    cols = [
        "section",
        "rank",
        "favorability",
        "decision_date",
        "execution_date",
        "symbol",
        "action",
        "close_t",
        "execution_price",
        "overnight_gap_return",
        "pre_trade_weight",
        "post_trade_weight",
        "trade_weight_change",
        "traded_notional_pct",
        "target_weight",
        "executed_weight",
        "signed_weighted_gap_impact",
        "abs_signed_weighted_gap_impact",
        "impact_direction",
        "impact_note",
    ]

    return pd.concat([favorable[cols], unfavorable[cols]], ignore_index=True)


def build_weighted_gap_analysis(events: pd.DataFrame) -> pd.DataFrame:
    overall = _build_overall_summary(events)
    action = _build_action_summary(events)
    top = _build_top_events(events)

    out = pd.concat([overall, action, top], ignore_index=True, sort=False)
    return out


def run_analysis(input_path: Path, output_path: Path) -> Dict[str, float]:
    raw = pd.read_csv(input_path)
    events = compute_event_impacts(raw)
    report = build_weighted_gap_analysis(events)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    stats = {
        "total_signed_weighted_gap_impact": float(events["signed_weighted_gap_impact"].sum()),
        "avg_signed_weighted_gap_impact": float(events["signed_weighted_gap_impact"].mean())
        if len(events)
        else 0.0,
        "favorable_event_count": int((events["impact_direction"] == "favorable").sum()),
        "unfavorable_event_count": int((events["impact_direction"] == "unfavorable").sum()),
        "neutral_event_count": int((events["impact_direction"] == "neutral").sum()),
    }

    action_totals = {}
    for action in ["bought", "sold", "held", "skipped"]:
        s = events[events["action"] == action]
        action_totals[action] = float(s["signed_weighted_gap_impact"].sum()) if len(s) else 0.0
    stats["action_totals"] = action_totals

    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze weighted next-open gap impact from an existing diagnostics CSV."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input diagnostics CSV path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output analysis CSV path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = run_analysis(args.input, args.output)

    print("=== WEIGHTED EXECUTION GAP IMPACT ===")
    print(f"Input path:  {args.input}")
    print(f"Output path: {args.output}")
    print(f"Total weighted signed impact:   {stats['total_signed_weighted_gap_impact']:.10f}")
    print(f"Average weighted signed impact: {stats['avg_signed_weighted_gap_impact']:.10f}")
    print("Action totals:")
    for action in ["bought", "sold", "held", "skipped"]:
        print(f"  {action}: {stats['action_totals'][action]:.10f}")
    print(
        "Event direction counts: "
        f"favorable={stats['favorable_event_count']}, "
        f"unfavorable={stats['unfavorable_event_count']}, "
        f"neutral={stats['neutral_event_count']}"
    )


if __name__ == "__main__":
    main()
