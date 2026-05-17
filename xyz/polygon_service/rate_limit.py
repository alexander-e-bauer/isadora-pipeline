"""Token-bucket rate limiter for Polygon REST API calls.

When the bucket is empty, ``acquire()`` sleeps until enough tokens have
refilled rather than raising.  The refill rate and capacity are both set to
``rate_per_min`` tokens, so a burst of up to that many requests goes through
immediately and then the caller is throttled to the steady-state rate.

``POLYGON_RATE_PER_MIN`` env var overrides the default of 100 req/min.
"""
from __future__ import annotations

import os
import time
import threading


class RateLimiter:
    """Thread-safe token-bucket rate limiter."""

    def __init__(self, rate_per_min: int | None = None) -> None:
        if rate_per_min is None:
            rate_per_min = int(os.environ.get("POLYGON_RATE_PER_MIN", 100))
        self._capacity: float = float(rate_per_min)
        self._tokens: float = float(rate_per_min)
        self._refill_per_sec: float = rate_per_min / 60.0
        self._last_refill: float = time.monotonic()
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Add tokens accrued since the last refill (must hold _lock)."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_sec)
        self._last_refill = now

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self, n: int = 1) -> None:
        """Block until *n* tokens are available, then consume them."""
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= n:
                    self._tokens -= n
                    return
                # Calculate how long until enough tokens are available.
                deficit = n - self._tokens
                wait_secs = deficit / self._refill_per_sec

            time.sleep(wait_secs)
