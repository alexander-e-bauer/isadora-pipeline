"""Engine-side tests for the request-id correlation infrastructure (Task 5.2).

Mirrors ``server-fastapi-wt/tests/test_request_id.py`` so the
spec-compliance reviewer can map server tests 1-to-1 against engine tests.

Test inventory
--------------
1. test_request_with_no_inbound_header_gets_generated_id
2. test_inbound_request_id_is_echoed_back
3. test_two_consecutive_requests_get_distinct_ids
4. test_emit_event_during_request_stamps_request_id_into_payload
5. test_emit_event_outside_request_scope_omits_request_id
6. test_json_formatter_emits_required_keys
7. test_json_formatter_includes_request_id_when_set
8. test_json_formatter_serialises_exception_traceback
9. test_oversized_inbound_request_id_is_rejected
10. test_set_firm_id_propagates_into_log_records
11. test_configure_logging_is_idempotent

We build a minimal FastAPI app per-test (mirroring helpers_app.py) so we
don't drag in the real engine app.py (which imports Pinecone, OpenAI, etc.).
The middleware behaviour is independent of the rest of the engine.
"""
from __future__ import annotations

import json
import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from xyz.observability import (
    HEADER_NAME,
    JsonFormatter,
    RequestIdMiddleware,
    configure_logging,
    firm_id_var,
    get_firm_id,
    get_request_id,
    request_id_var,
    set_firm_id,
)
from xyz.tenant.events import emit_event
from xyz.tenant.models import Event, Firm


def _make_app() -> FastAPI:
    """Minimal FastAPI app with only the request-id middleware mounted."""
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/ping")
    def _ping():
        return {"ok": True, "request_id": get_request_id()}

    return app


# ---------------------------------------------------------------------------
# 1. test_request_with_no_inbound_header_gets_generated_id
# ---------------------------------------------------------------------------

def test_request_with_no_inbound_header_gets_generated_id():
    client = TestClient(_make_app())
    resp = client.get("/ping")
    assert resp.status_code == 200
    rid = resp.headers.get(HEADER_NAME)
    assert rid is not None
    assert len(rid) == 32
    assert all(c in "0123456789abcdef" for c in rid)
    # The handler also saw the same id via the ContextVar.
    assert resp.json()["request_id"] == rid


# ---------------------------------------------------------------------------
# 2. test_inbound_request_id_is_echoed_back
# ---------------------------------------------------------------------------

def test_inbound_request_id_is_echoed_back():
    client = TestClient(_make_app())
    inbound = "server-issued-correlation-id"
    resp = client.get("/ping", headers={HEADER_NAME: inbound})
    assert resp.status_code == 200
    assert resp.headers[HEADER_NAME] == inbound
    # The handler picks up the inbound id from the ContextVar — proving
    # the server→engine hop carries the correlation id end-to-end.
    assert resp.json()["request_id"] == inbound


# ---------------------------------------------------------------------------
# 3. test_two_consecutive_requests_get_distinct_ids
# ---------------------------------------------------------------------------

def test_two_consecutive_requests_get_distinct_ids():
    client = TestClient(_make_app())
    r1 = client.get("/ping")
    r2 = client.get("/ping")
    assert r1.headers[HEADER_NAME] != r2.headers[HEADER_NAME]


# ---------------------------------------------------------------------------
# 4. test_emit_event_during_request_stamps_request_id_into_payload
# ---------------------------------------------------------------------------

def test_emit_event_during_request_stamps_request_id_into_payload(db_session):
    firm = Firm(name="ACME Capital")
    db_session.add(firm)
    db_session.flush()

    rid = "engine-side-rid-xyz"
    token = request_id_var.set(rid)
    try:
        event_id = emit_event(
            db=db_session,
            kind="pipeline.run_started",
            firm_id=firm.id,
            actor_user_id=None,
            payload={"ticker": "AAPL"},
        )
        db_session.commit()
    finally:
        request_id_var.reset(token)

    row = db_session.get(Event, event_id)
    assert row is not None
    assert row.payload.get("_request_id") == rid
    assert row.payload.get("ticker") == "AAPL"


# ---------------------------------------------------------------------------
# 5. test_emit_event_outside_request_scope_omits_request_id
# ---------------------------------------------------------------------------

def test_emit_event_outside_request_scope_omits_request_id(db_session):
    firm = Firm(name="ACME Capital")
    db_session.add(firm)
    db_session.flush()

    assert get_request_id() is None
    event_id = emit_event(
        db=db_session,
        kind="pipeline.run_started",
        firm_id=firm.id,
        actor_user_id=None,
        payload={"ticker": "AAPL"},
    )
    db_session.commit()
    row = db_session.get(Event, event_id)
    assert row is not None
    assert "_request_id" not in row.payload


# ---------------------------------------------------------------------------
# 6. test_json_formatter_emits_required_keys
# ---------------------------------------------------------------------------

def test_json_formatter_emits_required_keys():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="xyz.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello world", args=(), exc_info=None,
    )
    payload = json.loads(formatter.format(record))
    for key in ("ts", "level", "logger", "msg", "request_id", "firm_id"):
        assert key in payload, f"missing key: {key}"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "xyz.test"
    assert payload["msg"] == "hello world"
    assert payload["request_id"] is None
    assert payload["firm_id"] is None


# ---------------------------------------------------------------------------
# 7. test_json_formatter_includes_request_id_when_set
# ---------------------------------------------------------------------------

def test_json_formatter_includes_request_id_when_set():
    formatter = JsonFormatter()
    rid = "engine-ctx-rid"
    fid = 99

    rid_token = request_id_var.set(rid)
    fid_token = firm_id_var.set(fid)
    try:
        record = logging.LogRecord(
            name="xyz.test", level=logging.INFO, pathname=__file__, lineno=1,
            msg="with context", args=(), exc_info=None,
        )
        payload = json.loads(formatter.format(record))
    finally:
        request_id_var.reset(rid_token)
        firm_id_var.reset(fid_token)

    assert payload["request_id"] == rid
    assert payload["firm_id"] == fid


# ---------------------------------------------------------------------------
# 8. test_json_formatter_serialises_exception_traceback
# ---------------------------------------------------------------------------

def test_json_formatter_serialises_exception_traceback():
    formatter = JsonFormatter()
    try:
        raise ValueError("kaboom")
    except ValueError:
        import sys
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="xyz.test", level=logging.ERROR, pathname=__file__, lineno=1,
        msg="failure", args=(), exc_info=exc_info,
    )
    payload = json.loads(formatter.format(record))
    assert "trace" in payload
    assert "ValueError" in payload["trace"]
    assert "kaboom" in payload["trace"]


# ---------------------------------------------------------------------------
# 9. test_oversized_inbound_request_id_is_rejected
# ---------------------------------------------------------------------------

def test_oversized_inbound_request_id_is_rejected():
    client = TestClient(_make_app())
    huge = "x" * 5000
    resp = client.get("/ping", headers={HEADER_NAME: huge})
    assert resp.status_code == 200
    echoed = resp.headers[HEADER_NAME]
    assert echoed != huge
    assert len(echoed) == 32


# ---------------------------------------------------------------------------
# 10. test_set_firm_id_propagates_into_log_records
# ---------------------------------------------------------------------------

def test_set_firm_id_propagates_into_log_records():
    formatter = JsonFormatter()
    token = set_firm_id(11)
    try:
        assert get_firm_id() == 11
        record = logging.LogRecord(
            name="xyz.test", level=logging.INFO, pathname=__file__, lineno=1,
            msg="hi", args=(), exc_info=None,
        )
        payload = json.loads(formatter.format(record))
        assert payload["firm_id"] == 11
    finally:
        firm_id_var.reset(token)


# ---------------------------------------------------------------------------
# 11. test_configure_logging_is_idempotent
# ---------------------------------------------------------------------------

def test_configure_logging_is_idempotent():
    configure_logging()
    handlers_after_first = len(logging.getLogger().handlers)
    configure_logging()
    handlers_after_second = len(logging.getLogger().handlers)
    assert handlers_after_first == handlers_after_second == 1
