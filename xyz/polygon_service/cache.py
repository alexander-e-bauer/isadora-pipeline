"""TTL cache for Polygon option-chain snapshots.

Key is a 4-tuple:
    (underlying: str,
     asof_minute: datetime truncated to the minute or None,
     expiration_gte: date | None,
     expiration_lte: date | None)

Thread-safe via an internal lock wrapping every read/write.
"""
from __future__ import annotations

import threading
from datetime import datetime
from typing import TYPE_CHECKING, Any

import cachetools

if TYPE_CHECKING:
    from xyz.polygon_service.options_client import ChainSnapshot


class ChainCache:
    """Thread-safe TTL cache for ChainSnapshot objects."""

    def __init__(self, ttl_seconds: int = 300, maxsize: int = 128) -> None:
        self._cache: cachetools.TTLCache = cachetools.TTLCache(
            maxsize=maxsize, ttl=ttl_seconds
        )
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(
        underlying: str,
        asof: datetime | None = None,
        expiration_gte: Any = None,
        expiration_lte: Any = None,
    ) -> tuple:
        """Return a hashable cache key."""
        asof_minute = (
            asof.replace(second=0, microsecond=0) if asof is not None else None
        )
        return (underlying.upper(), asof_minute, expiration_gte, expiration_lte)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: tuple) -> "ChainSnapshot | None":
        with self._lock:
            return self._cache.get(key)

    def put(self, key: tuple, value: "ChainSnapshot") -> None:
        with self._lock:
            self._cache[key] = value
