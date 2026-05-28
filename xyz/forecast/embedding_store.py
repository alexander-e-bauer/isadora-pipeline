"""EmbeddingStore — vector-search interface scoped to the forecast feature.

v1 implementation wraps the existing Pinecone setup. Two indexes are
held: one for the daily MarketEmbDay grain, one for the weekly
MarketEmbWeek grain. The 30T tier is deliberately not exposed —
intraday similarity is orthogonal to daily-bar forecasting.

The deferred follow-up spec swaps PineconeStore → PgVectorStore with
no other code change required; only the implementation class differs.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Literal, Protocol

import numpy as np


class EmbeddingStore(Protocol):
    def query(
        self, *,
        query_vector: np.ndarray,
        granularity: Literal["D", "W"],
        top_k: int,
        symbol: str,
        as_of: date,
    ) -> list[tuple[date, date, float]]:
        """Top-K similar periods.

        Returns [(period_start, period_end, similarity), ...] sorted
        by similarity desc. Periods with period_end >= as_of are
        excluded (no lookahead leakage into the calibration).
        """
        ...


class PineconeStore:
    """v1 EmbeddingStore against the existing Pinecone setup.

    Two index handles are required at construction — one for the daily
    grain, one for the weekly grain. The 30T tier is rejected.
    """

    def __init__(self, *, index_d: Any, index_w: Any, namespace: str = ""):
        self._index_d = index_d
        self._index_w = index_w
        self._namespace = namespace

    def query(
        self, *,
        query_vector: np.ndarray,
        granularity: Literal["D", "W"],
        top_k: int,
        symbol: str,
        as_of: date,
    ) -> list[tuple[date, date, float]]:
        if granularity == "30T":
            raise ValueError(
                "EmbeddingStore.query: granularity='30T' is not supported by "
                "the forecast feature (intraday is orthogonal to daily-bar "
                "forecasting)"
            )
        if granularity not in ("D", "W"):
            raise ValueError(f"unknown granularity: {granularity!r}")

        index = self._index_d if granularity == "D" else self._index_w
        filter_ = {
            "symbol": symbol,
            "period_end": {"$lt": as_of.isoformat()},
        }
        res = index.query(
            vector=query_vector.tolist() if isinstance(query_vector, np.ndarray) else query_vector,
            top_k=top_k,
            namespace=self._namespace or None,
            filter=filter_,
            include_metadata=True,
        )
        out: list[tuple[date, date, float]] = []
        for m in res.get("matches", []):
            meta = m.get("metadata") or {}
            ps = date.fromisoformat(meta["period_start"])
            pe = date.fromisoformat(meta["period_end"])
            out.append((ps, pe, float(m["score"])))
        return out
