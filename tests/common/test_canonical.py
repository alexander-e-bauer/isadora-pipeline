"""Tests for engine.xyz.common.canonical — promoted from backtest engine."""
from __future__ import annotations

from datetime import date, datetime

from xyz.common.canonical import quantize_floats


def test_float_renders_to_6_decimal_string():
    assert quantize_floats(1.234567891) == "1.234568"
    assert quantize_floats(0.1) == "0.100000"
    assert quantize_floats(-1.0) == "-1.000000"


def test_int_passes_through():
    assert quantize_floats(42) == 42
    assert quantize_floats(0) == 0


def test_str_passes_through():
    assert quantize_floats("hello") == "hello"


def test_date_isoformats():
    assert quantize_floats(date(2026, 5, 28)) == "2026-05-28"


def test_datetime_isoformats():
    assert quantize_floats(datetime(2026, 5, 28, 12, 0, 0)) == "2026-05-28T12:00:00"


def test_recurses_into_dict():
    assert quantize_floats({"a": 1.5, "b": 2}) == {"a": "1.500000", "b": 2}


def test_recurses_into_list():
    assert quantize_floats([1.5, 2.5]) == ["1.500000", "2.500000"]


def test_recurses_into_tuple():
    assert quantize_floats((1.5, 2.5)) == ["1.500000", "2.500000"]


def test_handles_nested_structures():
    obj = {"nav": [1.0, 1.123456789], "meta": {"sigma": 0.2, "date": date(2026, 1, 1)}}
    assert quantize_floats(obj) == {
        "nav": ["1.000000", "1.123457"],
        "meta": {"sigma": "0.200000", "date": "2026-01-01"},
    }
