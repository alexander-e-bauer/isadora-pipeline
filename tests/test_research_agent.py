"""Tests for the RESEARCH subagent.

All Anthropic and Polygon HTTP calls are mocked — no live API calls in CI.

Test cases
----------
1. test_research_with_symbol_returns_artifact
2. test_research_with_brief_only_skips_iv_and_news
3. test_research_requires_symbol_or_brief
4. test_research_emits_research_artifact_event
5. test_research_content_hash_is_deterministic
6. test_research_payload_includes_citations
7. test_research_route_rejects_missing_bearer
8. test_research_route_with_invalid_bearer_returns_401
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# ---------------------------------------------------------------------------
# Path / fixture helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "anthropic"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


# ---------------------------------------------------------------------------
# In-memory SQLite helpers (reuse conftest's tenant Base)
# ---------------------------------------------------------------------------

def _make_tenant_db():
    """Create an in-memory SQLite DB with tenant + polygon tables."""
    from xyz.tenant.models import Base as TenantBase
    from xyz.finazon_service.base import Base as FinazonBase
    import xyz.polygon_service.models  # noqa: register polygon tables

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TenantBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return engine, Session


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_llm_response(fixture_name: str = "research_aapl_response.json"):
    """Return an object that quacks like anthropic.types.Message."""
    raw = _load_fixture(fixture_name)
    text = json.dumps(raw)
    content_block = SimpleNamespace(text=text)
    return SimpleNamespace(content=[content_block])


def _make_mock_options_client(symbol: str = "AAPL", iv: float = 0.285):
    """Return a mock OptionsClient that returns a minimal chain snapshot."""
    from xyz.polygon_service.options_client import ChainSnapshot, OptionContract

    contract = OptionContract(
        contract_ticker=f"O:{symbol}250117C00185000",
        underlying=symbol,
        expiry=date(2025, 1, 17),
        strike=185.0,
        option_type="CALL",
        bid=2.50,
        ask=2.60,
        mid=2.55,
        implied_vol=iv,
        delta=0.52,
    )
    snapshot = ChainSnapshot(
        underlying=symbol,
        underlying_price=185.0,
        asof_at=datetime(2025, 1, 17, 20, 0, 0),
        contracts=[contract],
    )
    client = MagicMock()
    client.get_chain.return_value = snapshot
    return client


def _make_mock_anthropic_client(fixture_name: str = "research_aapl_response.json"):
    """Return a mock AnthropicClient whose complete() returns a canned response."""
    client = MagicMock()
    client.complete.return_value = _make_mock_llm_response(fixture_name)
    return client


def _make_agent(db_session_factory, fixture_name: str = "research_aapl_response.json"):
    """Convenience: build a ResearchAgent with all external deps mocked."""
    from xyz.agents.research import ResearchAgent

    return ResearchAgent(
        anthropic_client=_make_mock_anthropic_client(fixture_name),
        options_client=_make_mock_options_client(),
        db_session_factory=db_session_factory,
    )


# ---------------------------------------------------------------------------
# Test 1 — symbol run returns a populated ResearchArtifact
# ---------------------------------------------------------------------------

def test_research_with_symbol_returns_artifact(db_session):
    """ResearchAgent.run with a symbol produces summary + iv_regime + news_headlines."""
    from xyz.agents.research import ResearchAgent
    from xyz.agents.schemas import ResearchInput, ResearchArtifact

    _, Session = _make_tenant_db()

    # Patch _gather_news_context so it doesn't hit the real DB
    agent = ResearchAgent(
        anthropic_client=_make_mock_anthropic_client(),
        options_client=_make_mock_options_client(),
        db_session_factory=Session,
    )

    with patch.object(agent, "_gather_news_context", return_value=[
        {"title": "Test headline", "citation": {"kind": "db_row", "source": "documents:1", "excerpt": "Test headline"}}
    ]):
        artifact = agent.run(ResearchInput(firm_id=1, symbol="AAPL", actor_user_id=10))

    assert isinstance(artifact, ResearchArtifact)
    assert artifact.symbol == "AAPL"
    assert artifact.summary and len(artifact.summary) > 10
    assert artifact.iv_regime is not None
    assert artifact.news_headlines is not None
    assert artifact.content_hash and len(artifact.content_hash) == 64


# ---------------------------------------------------------------------------
# Test 2 — brief-only run skips IV and news
# ---------------------------------------------------------------------------

def test_research_with_brief_only_skips_iv_and_news():
    """When only brief is set, iv_regime and news_headlines must be None."""
    from xyz.agents.research import ResearchAgent
    from xyz.agents.schemas import ResearchInput

    _, Session = _make_tenant_db()

    # Claude response for a brief-only request
    brief_response = {
        "symbol": None,
        "summary": "The advisor is asking about general portfolio rebalancing context.",
        "iv_regime": None,
        "earnings_calendar": None,
        "news_headlines": None,
        "peer_comparison": None,
    }
    mock_llm = MagicMock()
    mock_llm.complete.return_value = SimpleNamespace(
        content=[SimpleNamespace(text=json.dumps(brief_response))]
    )

    agent = ResearchAgent(
        anthropic_client=mock_llm,
        options_client=_make_mock_options_client(),
        db_session_factory=Session,
    )

    artifact = agent.run(ResearchInput(
        firm_id=1,
        brief="General portfolio rebalancing context for Q2 2026",
    ))

    assert artifact.symbol is None
    assert artifact.iv_regime is None
    assert artifact.news_headlines is None
    # Polygon client must NOT have been called
    agent._options.get_chain.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3 — missing symbol AND brief raises ValueError
# ---------------------------------------------------------------------------

def test_research_requires_symbol_or_brief():
    """ResearchAgent.run raises ValueError when neither symbol nor brief is set."""
    from xyz.agents.research import ResearchAgent
    from xyz.agents.schemas import ResearchInput

    _, Session = _make_tenant_db()
    agent = ResearchAgent(
        anthropic_client=_make_mock_anthropic_client(),
        options_client=_make_mock_options_client(),
        db_session_factory=Session,
    )

    with pytest.raises(ValueError, match="symbol.*brief"):
        agent.run(ResearchInput(firm_id=1))


# ---------------------------------------------------------------------------
# Test 4 — run emits a research.artifact event row
# ---------------------------------------------------------------------------

def test_research_emits_research_artifact_event():
    """After run(), the events table must have one row with kind='research.artifact'."""
    from xyz.agents.research import ResearchAgent
    from xyz.agents.schemas import ResearchInput
    from xyz.tenant.models import Event

    _, Session = _make_tenant_db()

    agent = ResearchAgent(
        anthropic_client=_make_mock_anthropic_client(),
        options_client=_make_mock_options_client(),
        db_session_factory=Session,
    )

    with patch.object(agent, "_gather_news_context", return_value=[]):
        agent.run(ResearchInput(firm_id=42, symbol="AAPL", actor_user_id=7))

    # Inspect the event table directly
    session = Session()
    try:
        events = session.query(Event).filter_by(kind="research.artifact", firm_id=42).all()
        assert len(events) == 1
        ev = events[0]
        assert ev.actor_user_id == 7
        assert ev.payload is not None
        assert ev.payload.get("symbol") == "AAPL"
        assert "content_hash" in ev.payload
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Test 5 — content hash is deterministic
# ---------------------------------------------------------------------------

def test_research_content_hash_is_deterministic():
    """_hash_artifact must return the same hash for the same dict and different hash for different dict."""
    from xyz.agents.research import _hash_artifact

    d = {
        "symbol": "AAPL",
        "summary": "Test summary",
        "iv_regime": None,
        "earnings_calendar": None,
        "news_headlines": None,
        "peer_comparison": None,
    }

    h1 = _hash_artifact(d)
    h2 = _hash_artifact(d)
    assert h1 == h2, "Same dict → same hash"
    assert len(h1) == 64, "SHA-256 hex digest is 64 chars"

    d2 = dict(d)
    d2["summary"] = "Different summary"
    h3 = _hash_artifact(d2)
    assert h3 != h1, "Different dict → different hash"

    # generated_at and content_hash are excluded from the hash
    d_with_meta = dict(d)
    d_with_meta["generated_at"] = "2026-05-17T00:00:00+00:00"
    d_with_meta["content_hash"] = "deadbeef"
    h4 = _hash_artifact(d_with_meta)
    assert h4 == h1, "generated_at / content_hash must be excluded from hash"


# ---------------------------------------------------------------------------
# Test 6 — emitted payload includes citations
# ---------------------------------------------------------------------------

def test_research_payload_includes_citations():
    """The persisted event payload must include at least one non-empty citations list."""
    from xyz.agents.research import ResearchAgent
    from xyz.agents.schemas import ResearchInput
    from xyz.tenant.models import Event

    _, Session = _make_tenant_db()

    agent = ResearchAgent(
        anthropic_client=_make_mock_anthropic_client("research_aapl_response.json"),
        options_client=_make_mock_options_client(),
        db_session_factory=Session,
    )

    with patch.object(agent, "_gather_news_context", return_value=[]):
        agent.run(ResearchInput(firm_id=99, symbol="AAPL"))

    session = Session()
    try:
        ev = session.query(Event).filter_by(kind="research.artifact", firm_id=99).one()
        payload = ev.payload

        # At least one section with citations
        has_citations = any(
            isinstance(payload.get(sec), dict) and payload[sec].get("citations")
            for sec in ("iv_regime", "earnings_calendar", "news_headlines", "peer_comparison")
        )
        assert has_citations, f"Expected at least one non-empty citations list. Payload: {payload}"
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Test 7 — route rejects missing bearer
# ---------------------------------------------------------------------------

def test_research_route_rejects_missing_bearer():
    """POST /agents/research without Authorization header returns 401."""
    # We need to import the FastAPI app; patch env vars to avoid live clients
    with patch.dict(os.environ, {"KEY": "test-secret", "ANTHROPIC_API_KEY": "fake", "POLYGON_KEY": "fake"}):
        # Import app with patched env — use a lazy import inside the test
        from tests.helpers_app import get_test_client
        client = get_test_client()
        resp = client.post("/agents/research", json={"firm_id": 1, "symbol": "AAPL"})
    # FastAPI HTTPBearer raises 401 (missing creds) or 403 (bad scheme) depending on version
    assert resp.status_code in (401, 403)


# ---------------------------------------------------------------------------
# Test 8 — route rejects invalid bearer
# ---------------------------------------------------------------------------

def test_research_route_with_invalid_bearer_returns_401():
    """POST /agents/research with wrong bearer token returns 401."""
    with patch.dict(os.environ, {"KEY": "test-secret", "ANTHROPIC_API_KEY": "fake", "POLYGON_KEY": "fake"}):
        from tests.helpers_app import get_test_client
        client = get_test_client()
        resp = client.post(
            "/agents/research",
            json={"firm_id": 1, "symbol": "AAPL"},
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert resp.status_code == 401
