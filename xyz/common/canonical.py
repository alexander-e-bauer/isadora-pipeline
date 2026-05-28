"""Canonical serialization helper for content-hash inputs.

Promoted from engine/xyz/backtest/engine.py so both BacktestAgent and
ForecastAgent can produce byte-stable content hashes using the same
6-decimal float canonicalization.

The hashing layer in callers uses these strings (not the raw floats)
so floating-point repr drift in the LSBs cannot change the hash.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any


def quantize_floats(obj: Any) -> Any:
    """Recursively walk ``obj`` rendering every float as a 6-decimal string.

    Lists and tuples become lists. Dicts recurse on values. Dates and
    datetimes are isoformatted. Other types pass through untouched.

    The 6-decimal precision is the cross-platform-stable choice the
    audit chain relies on.
    """
    if isinstance(obj, float):
        return f"{obj:.6f}"
    if isinstance(obj, dict):
        return {k: quantize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [quantize_floats(v) for v in obj]
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return obj
