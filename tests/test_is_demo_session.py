"""Engine reads X-Demo-Session header forwarded by the server."""
from __future__ import annotations

from fastapi import Request

from xyz.agents.auth import is_demo_session


def _make_request(headers: list[tuple[bytes, bytes]]) -> Request:
    scope = {"type": "http", "headers": headers}
    return Request(scope)


def test_returns_true_when_header_present_lowercase():
    req = _make_request([(b"x-demo-session", b"true")])
    assert is_demo_session(req) is True


def test_returns_true_when_header_present_mixed_case():
    """HTTP headers are case-insensitive — the dep must accept any casing."""
    req = _make_request([(b"X-Demo-Session", b"true")])
    assert is_demo_session(req) is True


def test_returns_false_when_header_absent():
    req = _make_request([])
    assert is_demo_session(req) is False


def test_returns_false_when_header_value_not_true():
    """Only the literal 'true' (case-insensitive) flips the flag."""
    req = _make_request([(b"x-demo-session", b"false")])
    assert is_demo_session(req) is False


def test_returns_false_when_header_value_random_string():
    req = _make_request([(b"x-demo-session", b"yes")])
    assert is_demo_session(req) is False


def test_case_insensitive_true_value():
    """Server might forward 'True' or 'TRUE' — accept any casing."""
    req = _make_request([(b"x-demo-session", b"True")])
    assert is_demo_session(req) is True
