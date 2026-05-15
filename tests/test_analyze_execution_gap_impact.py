from pathlib import Path

import pandas as pd
import pytest

from research.analyze_execution_gap_impact import (
    build_weighted_gap_analysis,
    compute_event_impacts,
    run_analysis,
)


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "decision_date": "2024-01-01",
                "execution_date": "2024-01-02",
                "symbol": "AAA",
                "action": "bought",
                "overnight_gap_return": 0.10,
                "trade_weight_change": 0.20,
                "traded_notional_pct": 0.20,
            },
            {
                "decision_date": "2024-01-08",
                "execution_date": "2024-01-09",
                "symbol": "BBB",
                "action": "sold",
                "overnight_gap_return": 0.08,
                "trade_weight_change": -0.15,
                "traded_notional_pct": 0.15,
            },
            {
                "decision_date": "2024-01-15",
                "execution_date": "2024-01-16",
                "symbol": "CCC",
                "action": "held",
                "overnight_gap_return": -0.05,
                "trade_weight_change": 0.0,
                "traded_notional_pct": 0.0,
            },
            {
                "decision_date": "2024-01-22",
                "execution_date": "2024-01-23",
                "symbol": "DDD",
                "action": "skipped",
                "overnight_gap_return": 0.03,
                "trade_weight_change": 0.0,
                "traded_notional_pct": 0.0,
            },
            {
                "decision_date": "2024-01-29",
                "execution_date": "2024-01-30",
                "symbol": "EEE",
                "action": "bought",
                "overnight_gap_return": -0.04,
                "trade_weight_change": 0.10,
                "traded_notional_pct": 0.10,
            },
            {
                "decision_date": "2024-02-05",
                "execution_date": "2024-02-06",
                "symbol": "FFF",
                "action": "sold",
                "overnight_gap_return": -0.02,
                "trade_weight_change": -0.20,
                "traded_notional_pct": 0.20,
            },
        ]
    )


def test_bought_impact_uses_negative_gap_times_positive_change():
    events = compute_event_impacts(_toy_df())
    row = events.loc[events["symbol"] == "AAA"].iloc[0]
    assert row["signed_weighted_gap_impact"] == pytest.approx(-0.10 * 0.20)


def test_sold_impact_uses_positive_gap_times_abs_negative_change():
    events = compute_event_impacts(_toy_df())
    row = events.loc[events["symbol"] == "BBB"].iloc[0]
    assert row["signed_weighted_gap_impact"] == pytest.approx(0.08 * 0.15)


def test_held_and_skipped_impacts_are_zero():
    events = compute_event_impacts(_toy_df())
    held_row = events.loc[events["symbol"] == "CCC"].iloc[0]
    skipped_row = events.loc[events["symbol"] == "DDD"].iloc[0]
    assert held_row["signed_weighted_gap_impact"] == pytest.approx(0.0)
    assert skipped_row["signed_weighted_gap_impact"] == pytest.approx(0.0)


def test_missing_required_columns_raises_clear_error():
    bad = pd.DataFrame(
        [{"action": "bought", "overnight_gap_return": 0.01, "trade_weight_change": 0.02}]
    )
    with pytest.raises(ValueError) as exc:
        compute_event_impacts(bad)
    assert "Missing required columns" in str(exc.value)
    assert "traded_notional_pct" in str(exc.value)


def test_output_contains_required_sections(tmp_path: Path):
    inp = tmp_path / "diag.csv"
    out = tmp_path / "analysis.csv"
    _toy_df().to_csv(inp, index=False)

    run_analysis(inp, out)

    result = pd.read_csv(out)
    sections = set(result["section"].dropna().astype(str).tolist())
    assert "overall_summary" in sections
    assert "action_summary" in sections
    assert "top_events_weighted" in sections


def test_top_favorable_unfavorable_ranking_on_toy_data():
    events = compute_event_impacts(_toy_df())
    report = build_weighted_gap_analysis(events)
    top = report[report["section"] == "top_events_weighted"].copy()

    top_fav = top[top["favorability"] == "favorable"].sort_values("rank")
    top_unfav = top[top["favorability"] == "unfavorable"].sort_values("rank")

    assert top_fav.iloc[0]["symbol"] == "BBB"  # +(0.08)*abs(-0.15) = +0.012
    assert top_unfav.iloc[0]["symbol"] == "AAA"  # -(0.10)*0.20 = -0.020
