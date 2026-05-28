"""EmbeddingStore + PineconeStore — query semantics, no real Pinecone calls."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pytest

from xyz.forecast.embedding_store import PineconeStore


def _make_store(mock_index) -> PineconeStore:
    return PineconeStore(index_d=mock_index, index_w=mock_index, namespace="test")


def test_query_d_returns_period_date_pairs():
    mock_index = MagicMock()
    mock_index.query.return_value = {
        "matches": [
            {"id": "AAPL_2024-03-15", "score": 0.91,
             "metadata": {"period_start": "2024-03-15", "period_end": "2024-03-16"}},
            {"id": "AAPL_2024-06-01", "score": 0.87,
             "metadata": {"period_start": "2024-06-01", "period_end": "2024-06-02"}},
        ]
    }
    store = _make_store(mock_index)
    out = store.query(
        query_vector=np.zeros(1536),
        granularity="D",
        top_k=2,
        symbol="AAPL",
        as_of=date(2025, 1, 1),
    )
    assert out == [
        (date(2024, 3, 15), date(2024, 3, 16), 0.91),
        (date(2024, 6, 1), date(2024, 6, 2), 0.87),
    ]


def test_query_excludes_periods_after_as_of():
    """Filter `period_end < as_of` must be in the Pinecone query filter."""
    mock_index = MagicMock()
    mock_index.query.return_value = {"matches": []}
    store = _make_store(mock_index)
    store.query(
        query_vector=np.zeros(1536),
        granularity="D",
        top_k=5,
        symbol="AAPL",
        as_of=date(2025, 1, 1),
    )
    # Verify filter was applied
    call_args = mock_index.query.call_args
    filter_arg = call_args.kwargs.get("filter") or call_args[1].get("filter")
    assert filter_arg is not None
    assert filter_arg.get("symbol") == "AAPL"
    assert "period_end" in filter_arg


def test_granularity_w_uses_index_w():
    """Granularity 'W' must route to the W-tier index, not D."""
    mock_d = MagicMock()
    mock_w = MagicMock()
    mock_d.query.return_value = {"matches": []}
    mock_w.query.return_value = {"matches": []}
    store = PineconeStore(index_d=mock_d, index_w=mock_w, namespace="test")
    store.query(
        query_vector=np.zeros(1536),
        granularity="W",
        top_k=3,
        symbol="AAPL",
        as_of=date(2025, 1, 1),
    )
    mock_w.query.assert_called_once()
    mock_d.query.assert_not_called()


def test_granularity_30T_raises():
    """The forecast must NOT query 30T (orthogonal granularity)."""
    store = _make_store(MagicMock())
    with pytest.raises(ValueError, match="30T"):
        store.query(
            query_vector=np.zeros(1536),
            granularity="30T",   # type: ignore[arg-type]
            top_k=3,
            symbol="AAPL",
            as_of=date(2025, 1, 1),
        )
