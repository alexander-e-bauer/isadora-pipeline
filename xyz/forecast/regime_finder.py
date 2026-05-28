"""SimilarRegimeFinder — picks calibration windows for the Calibrator.

Two implementations:

* ``RecentWindowFinder`` — naive fallback that returns the single
  ``days``-long window ending at ``t0``. Used when no embedding
  coverage exists for the symbol or as a manual override.

* ``EmbeddingRegimeFinder`` — queries an ``EmbeddingStore`` for the
  top-K similar daily regimes and expands each match into a
  ``days``-long window ending at the matched ``period_end``.

The forecast's lookback windows are recorded as concrete dates in
``CalibratedParams.lookback_windows``, so determinism survives
embedding-index changes — the artifact still re-verifies because
the windows themselves are frozen.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Callable, Literal, Protocol

import numpy as np

from xyz.forecast.embedding_store import EmbeddingStore


class SimilarRegimeFinder(Protocol):
    def find(
        self, *,
        symbol: str,
        t0: date,
        k: int,
        days: int,
        granularity: Literal["D", "W"] = "D",
    ) -> list[tuple[date, date]]:
        """Returns a list of (window_start, window_end) tuples, K items long."""
        ...


class RecentWindowFinder:
    """Naive: one window of length ``days`` ending at ``t0``."""

    def find(
        self, *,
        symbol: str,
        t0: date,
        k: int,
        days: int,
        granularity: Literal["D", "W"] = "D",
    ) -> list[tuple[date, date]]:
        start = t0 - timedelta(days=days - 1)
        return [(start, t0)]


class EmbeddingRegimeFinder:
    """Embedding-conditioned: top-K similar regimes expanded to windows."""

    def __init__(
        self, *,
        store: EmbeddingStore,
        query_vector_provider: Callable[[str, date], np.ndarray],
    ):
        self._store = store
        self._provider = query_vector_provider

    def find(
        self, *,
        symbol: str,
        t0: date,
        k: int,
        days: int,
        granularity: Literal["D", "W"] = "D",
    ) -> list[tuple[date, date]]:
        vec = self._provider(symbol, t0)
        matches = self._store.query(
            query_vector=vec,
            granularity=granularity,
            top_k=k,
            symbol=symbol,
            as_of=t0,
        )
        windows: list[tuple[date, date]] = []
        for _, period_end, _ in matches:
            start = period_end - timedelta(days=days - 1)
            windows.append((start, period_end))
        return windows
