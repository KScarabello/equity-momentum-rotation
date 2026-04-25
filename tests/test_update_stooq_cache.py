from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research import update_stooq_cache


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_error_page_response_rejected_and_existing_parquet_preserved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "stooq"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "AAPL.US.parquet"

    original_df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [101.0],
            "Low": [99.5],
            "Close": [100.5],
            "Volume": [12345],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-02")], name="Date"),
    )
    original_df.to_parquet(out_path)

    before_bytes = out_path.read_bytes()

    def _fake_urlopen(req, timeout=20):  # noqa: ANN001
        body = b"Get your apikey to access this endpoint"
        return _FakeResponse(body=body)

    monkeypatch.setattr(update_stooq_cache, "urlopen", _fake_urlopen)

    with pytest.raises(RuntimeError):
        update_stooq_cache.update_symbol(symbol="AAPL", out_dir=out_dir, api_key="bad-key")

    after_bytes = out_path.read_bytes()
    assert after_bytes == before_bytes

    roundtrip = pd.read_parquet(out_path)
    assert not roundtrip.empty
    assert pd.Timestamp("2026-01-02") == pd.Timestamp(roundtrip.index.max())
