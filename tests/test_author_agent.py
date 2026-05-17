"""Tests for the AUTHOR subagent + DSL validator + route auth (Task 4.2).

All Anthropic calls are mocked — no live API in CI.

Inventory
---------
 1. test_author_covered_call_brief_returns_cc_dsl
 2. test_author_csp_brief_returns_csp_dsl
 3. test_author_collar_brief_returns_collar_dsl
 4. test_author_emits_strategy_draft_event
 5. test_author_stamps_firm_id_and_author_user_id
 6. test_author_rejects_invalid_dsl_from_claude
 7. test_validate_dsl_rejects_missing_required_sections
 8. test_validate_dsl_rejects_scripted_kind
 9. test_validate_dsl_accepts_full_declarative_dsl
10. test_validate_dsl_catches_typo_in_risk_box
11. test_author_route_rejects_missing_bearer
12. test_author_route_with_invalid_bearer_returns_401
13. test_validate_dsl_route_returns_200_with_errors_for_invalid
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "anthropic"


# ---------------------------------------------------------------------------
# Fixture / mock helpers
# ---------------------------------------------------------------------------

def _load_fixture_text(name: str) -> str:
    """Load a fixture as raw text — what Claude would emit as response.content[0].text."""
    return (FIXTURES_DIR / name).read_text()


def _load_fixture(name: str) -> dict:
    return json.loads(_load_fixture_text(name))


def _make_mock_anthropic(fixture_name: str):
    """Return a mock AnthropicClient whose complete() returns a canned response."""
    client = MagicMock()
    text = _load_fixture_text(fixture_name)
    client.complete.return_value = SimpleNamespace(content=[SimpleNamespace(text=text)])
    return client


def _make_tenant_db():
    """Create an in-memory SQLite DB with tenant tables only.

    AUTHOR does not read from the finazon/polygon tables, so we keep this
    helper minimal (compared to the RESEARCH variant that also wires up
    FinazonBase).  Keeps tests fast.
    """
    from xyz.tenant.models import Base as TenantBase

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TenantBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return engine, Session


def _make_agent(Session, fixture_name: str):
    from xyz.agents.author import AuthorAgent

    return AuthorAgent(
        anthropic_client=_make_mock_anthropic(fixture_name),
        db_session_factory=Session,
    )


# ---------------------------------------------------------------------------
# 1 — Covered-call brief returns a covered-call DSL with sensible defaults
# ---------------------------------------------------------------------------

def test_author_covered_call_brief_returns_cc_dsl():
    """Acceptance: brief 'income on AAPL, low assignment risk, monthlies,
    IRA suitable' returns a covered-call DSL with sensible defaults."""
    from xyz.agents.schemas import AuthorArtifact, AuthorInput

    _, Session = _make_tenant_db()
    agent = _make_agent(Session, "author_aapl_covered_call.json")

    artifact = agent.run(AuthorInput(
        firm_id=1,
        brief="income on AAPL, low assignment risk, monthlies, IRA suitable",
        actor_user_id=10,
    ))

    assert isinstance(artifact, AuthorArtifact)
    assert artifact.template == "covered_call"
    # Sensible defaults baked in by the fixture (modeling the spec defaults).
    assert artifact.dsl["action"]["delta_short_target"] == 0.30
    assert artifact.dsl["action"]["dte_min"] == 30
    assert artifact.dsl["action"]["dte_max"] == 45
    assert artifact.dsl["exit"]["max_profit_pct_close"] == 0.50
    assert artifact.dsl["risk_box"]["time_windows"]["earnings_blackout_days"] == 7
    # Content hash should be deterministic and 64 chars (sha256 hex).
    assert artifact.content_hash and len(artifact.content_hash) == 64
    # Rationale should resemble a Reg-BI stub (non-trivial length).
    assert len(artifact.rationale) > 50


# ---------------------------------------------------------------------------
# 2 — Cash-secured-put brief returns a CSP DSL
# ---------------------------------------------------------------------------

def test_author_csp_brief_returns_csp_dsl():
    from xyz.agents.schemas import AuthorInput

    _, Session = _make_tenant_db()
    agent = _make_agent(Session, "author_msft_cash_secured_put.json")

    artifact = agent.run(AuthorInput(
        firm_id=1,
        brief="entry ladder on MSFT via short puts",
        actor_user_id=11,
    ))

    assert artifact.template == "cash_secured_put"
    assert artifact.dsl["action"]["leg"] == "short_put"


# ---------------------------------------------------------------------------
# 3 — Collar brief returns a collar DSL
# ---------------------------------------------------------------------------

def test_author_collar_brief_returns_collar_dsl():
    from xyz.agents.schemas import AuthorInput

    _, Session = _make_tenant_db()
    agent = _make_agent(Session, "author_spy_collar.json")

    artifact = agent.run(AuthorInput(
        firm_id=1,
        brief="defensive collar on SPY for downside protection",
        actor_user_id=12,
    ))

    assert artifact.template == "collar"
    assert "short_call_delta" in artifact.dsl["action"]
    assert "long_put_delta" in artifact.dsl["action"]


# ---------------------------------------------------------------------------
# 4 — Event emission
# ---------------------------------------------------------------------------

def test_author_emits_strategy_draft_event():
    """After run(), the events table must have one row with kind='strategy.draft'."""
    from xyz.agents.schemas import AuthorInput
    from xyz.tenant.models import Event

    _, Session = _make_tenant_db()
    agent = _make_agent(Session, "author_aapl_covered_call.json")

    agent.run(AuthorInput(firm_id=42, brief="income on AAPL", actor_user_id=7))

    session = Session()
    try:
        events = session.query(Event).filter_by(kind="strategy.draft", firm_id=42).all()
        assert len(events) == 1
        ev = events[0]
        assert ev.actor_user_id == 7
        assert ev.payload is not None
        assert ev.payload.get("template") == "covered_call"
        assert ev.payload.get("dsl", {}).get("kind") == "declarative"
        assert "content_hash" in ev.payload
    finally:
        session.close()


# ---------------------------------------------------------------------------
# 5 — Agent stamps firm_id and author_user_id even if Claude omits them
# ---------------------------------------------------------------------------

def test_author_stamps_firm_id_and_author_user_id():
    """The agent must overwrite firm_id and author_user_id in the DSL with
    the authoritative values from AuthorInput, regardless of what Claude returned."""
    from xyz.agents.schemas import AuthorInput

    _, Session = _make_tenant_db()

    # Build a response with WRONG firm_id / author_user_id; the agent must overwrite.
    bad_fixture = _load_fixture("author_aapl_covered_call.json")
    bad_fixture["dsl"]["firm_id"] = 999    # wrong on purpose
    bad_fixture["dsl"]["author_user_id"] = 999   # wrong on purpose

    mock_client = MagicMock()
    mock_client.complete.return_value = SimpleNamespace(
        content=[SimpleNamespace(text=json.dumps(bad_fixture))]
    )

    from xyz.agents.author import AuthorAgent
    agent = AuthorAgent(anthropic_client=mock_client, db_session_factory=Session)

    artifact = agent.run(AuthorInput(firm_id=42, brief="x", actor_user_id=7))

    assert artifact.dsl["firm_id"] == 42, "firm_id must come from AuthorInput, not the model"
    assert artifact.dsl["author_user_id"] == 7


# ---------------------------------------------------------------------------
# 6 — Claude returning a malformed DSL raises ValueError + emits no event
# ---------------------------------------------------------------------------

def test_author_rejects_invalid_dsl_from_claude():
    """If Claude's DSL fails JSON-Schema validation, AUTHOR must raise
    ValueError and emit no event (audit log stays clean)."""
    from xyz.agents.author import AuthorAgent
    from xyz.agents.schemas import AuthorInput
    from xyz.tenant.models import Event

    _, Session = _make_tenant_db()

    # Build a response with a missing required section (exit removed).
    bad = _load_fixture("author_aapl_covered_call.json")
    del bad["dsl"]["exit"]

    mock_client = MagicMock()
    mock_client.complete.return_value = SimpleNamespace(
        content=[SimpleNamespace(text=json.dumps(bad))]
    )
    agent = AuthorAgent(anthropic_client=mock_client, db_session_factory=Session)

    with pytest.raises(ValueError, match="invalid DSL"):
        agent.run(AuthorInput(firm_id=1, brief="x", actor_user_id=5))

    # Audit log must not contain the malformed draft.
    session = Session()
    try:
        events = session.query(Event).filter_by(kind="strategy.draft").all()
        assert events == []
    finally:
        session.close()


# ---------------------------------------------------------------------------
# 7 — validate_dsl rejects missing required top-level sections
# ---------------------------------------------------------------------------

def test_validate_dsl_rejects_missing_required_sections():
    """Acceptance: DSL fails validation if any of selection/trigger/action/
    exit/risk_box are missing."""
    from xyz.dsl.validate import validate_dsl

    base = {
        "kind": "declarative",
        "name": "x",
        "version": 1,
        "author_user_id": 1,
        "firm_id": 1,
        "selection": {},
        "trigger": {},
        "action": {},
        "exit": {},
        "risk_box": {},
    }

    # Valid baseline.
    valid, errors = validate_dsl(base)
    assert valid, f"baseline must validate: {errors}"

    # Each missing top-level section must fail.
    for missing in ("selection", "trigger", "action", "exit", "risk_box"):
        bad = {k: v for k, v in base.items() if k != missing}
        valid, errors = validate_dsl(bad)
        assert not valid, f"DSL missing {missing} must fail"
        assert any(missing in e for e in errors), f"errors must name {missing}: {errors}"


# ---------------------------------------------------------------------------
# 8 — validate_dsl rejects kind=scripted with the literal error string
# ---------------------------------------------------------------------------

def test_validate_dsl_rejects_scripted_kind():
    """Acceptance: DSL with ``kind: 'scripted'`` is rejected with the
    literal string 'scripted_not_supported_in_v1' in the errors list."""
    from xyz.dsl.validate import validate_dsl

    dsl = {
        "kind": "scripted",
        "name": "x",
        "version": 1,
        "author_user_id": 1,
        "firm_id": 1,
        "selection": {},
        "trigger": {},
        "action": {},
        "exit": {},
        "risk_box": {},
    }
    valid, errors = validate_dsl(dsl)
    assert not valid
    assert "scripted_not_supported_in_v1" in errors


# ---------------------------------------------------------------------------
# 9 — validate_dsl accepts a full declarative DSL from our CC fixture
# ---------------------------------------------------------------------------

def test_validate_dsl_accepts_full_declarative_dsl():
    """A realistic, fully populated DSL must validate without errors."""
    from xyz.dsl.validate import validate_dsl

    fixture = _load_fixture("author_aapl_covered_call.json")
    valid, errors = validate_dsl(fixture["dsl"])
    assert valid, f"covered-call fixture must validate: {errors}"


# ---------------------------------------------------------------------------
# 10 — validate_dsl catches a typo in risk_box (additionalProperties: false)
# ---------------------------------------------------------------------------

def test_validate_dsl_catches_typo_in_risk_box():
    """A typo in a risk_box sub-key must be flagged — additionalProperties:false."""
    from xyz.dsl.validate import validate_dsl

    dsl = {
        "kind": "declarative",
        "name": "x",
        "version": 1,
        "author_user_id": 1,
        "firm_id": 1,
        "selection": {},
        "trigger": {},
        "action": {},
        "exit": {},
        "risk_box": {
            # typo: should be regime_gates → min_iv_rank, not min_iv_rnk
            "regime_gates": {"min_iv_rnk": 0.30},
        },
    }
    valid, errors = validate_dsl(dsl)
    assert not valid
    assert any("min_iv_rnk" in e or "Additional properties" in e for e in errors), errors


# ---------------------------------------------------------------------------
# 11 — Route auth: missing bearer returns 401/403
# ---------------------------------------------------------------------------

def test_author_route_rejects_missing_bearer():
    from unittest.mock import patch
    with patch.dict(os.environ, {"KEY": "test-secret", "ANTHROPIC_API_KEY": "fake"}):
        from tests.helpers_app import get_test_client
        client = get_test_client()
        resp = client.post("/agents/author", json={"firm_id": 1, "brief": "x"})
    assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# 12 — Route auth: invalid bearer returns 401
# ---------------------------------------------------------------------------

def test_author_route_with_invalid_bearer_returns_401():
    from unittest.mock import patch
    with patch.dict(os.environ, {"KEY": "test-secret", "ANTHROPIC_API_KEY": "fake"}):
        from tests.helpers_app import get_test_client
        client = get_test_client()
        resp = client.post(
            "/agents/author",
            json={"firm_id": 1, "brief": "income on AAPL"},
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 13 — validate-dsl route returns 200 with valid=false + errors on invalid input
# ---------------------------------------------------------------------------

def test_validate_dsl_route_returns_200_with_errors_for_invalid():
    """The validate-dsl route must always 200 — the body carries valid/errors."""
    from unittest.mock import patch
    with patch.dict(os.environ, {"KEY": "test-secret"}):
        from tests.helpers_app import get_test_client
        client = get_test_client()
        resp = client.post(
            "/agents/validate-dsl",
            json={"dsl": {"kind": "scripted"}},
            headers={"Authorization": "Bearer test-secret"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is False
    assert "scripted_not_supported_in_v1" in body["errors"]
