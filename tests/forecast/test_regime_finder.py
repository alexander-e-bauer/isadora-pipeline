"""SimilarRegimeFinder tests — RecentWindowFinder + EmbeddingRegimeFinder."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pytest

from xyz.forecast.regime_finder import (
    EmbeddingRegimeFinder,
    RecentWindowFinder,
)


def test_recent_window_finder_returns_single_window_ending_at_t0():
    finder = RecentWindowFinder()
    windows = finder.find(symbol="AAPL", t0=date(2025, 6, 1), k=3, days=30)
    assert windows == [(date(2025, 5, 3), date(2025, 6, 1))]   # t0 - 29 days


def test_embedding_finder_returns_topk_windows_from_store():
    store = MagicMock()
    store.query.return_value = [
        (date(2024, 3, 15), date(2024, 3, 16), 0.91),
        (date(2024, 6, 1),  date(2024, 6, 1),  0.87),
        (date(2024, 9, 4),  date(2024, 9, 4),  0.80),
    ]
    finder = EmbeddingRegimeFinder(
        store=store,
        query_vector_provider=lambda symbol, t0: np.zeros(1536),
    )
    windows = finder.find(symbol="AAPL", t0=date(2025, 1, 1), k=3, days=30)

    # Each match expanded into a 30-day window ending at the matched period_end
    assert len(windows) == 3
    assert windows[0] == (date(2024, 2, 16), date(2024, 3, 16))
    assert windows[1] == (date(2024, 5, 3),  date(2024, 6, 1))
    assert windows[2] == (date(2024, 8, 6),  date(2024, 9, 4))


def test_embedding_finder_passes_as_of_to_store():
    store = MagicMock()
    store.query.return_value = []
    finder = EmbeddingRegimeFinder(
        store=store,
        query_vector_provider=lambda symbol, t0: np.zeros(1536),
    )
    finder.find(symbol="AAPL", t0=date(2025, 1, 1), k=3, days=30)
    call = store.query.call_args.kwargs
    assert call["as_of"] == date(2025, 1, 1)
    assert call["granularity"] == "D"
    assert call["symbol"] == "AAPL"
    assert call["top_k"] == 3
