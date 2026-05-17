"""Tests for the PROPOSE subagent (Task 4.4).

All market data is synthetic — no live Polygon calls in CI.

Inventory
---------
 1. test_propose_emits_ticket_with_required_fields           (acceptance criterion 2)
 2. test_propose_emits_ticket_proposed_event                 (acceptance criterion 3 — engine side)
 3. test_propose_with_inactive_deployment_returns_empty      (spec §9 ACTIVE-gate)
 4. test_propose_with_no_long_shares_returns_empty           (acceptance criterion 1)
 5. test_propose_with_existing_short_call_returns_empty      (acceptance criterion 1)
 6. test_propose_with_no_chain_data_returns_empty            (acceptance criterion 1)
 7. test_propose_with_unsupported_template_returns_empty
 8. test_propose_route_rejects_missing_bearer
 9. test_propose_picks_nearest_delta_with_tiebreak
10. test_propose_records_account_autonomy_level

Acceptance criteria → tests mapping (Task 4.4, engine side)
-----------------------------------------------------------
1. trigger miss → {tickets: []}     : tests 3, 4, 5, 6, 7
2. trigger hit → ticket with fields : tests 1, 9
3. ticket.proposed event emitted    : test 2  (server-side: test_trades.py 17, 18)
4. trigger-hit + trigger-miss tests : covered above
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


# ---------------------------------------------------------------------------
# Stub xyz.finazon_service.sql_service before any propose import (test path).
# The engine venv lacks psycopg2 and the real module imports it at module
# load.  PROPOSE never calls the production session factory in tests, but
# importing xyz.tenant.db at module load would fail without the stub.
# ---------------------------------------------------------------------------

if "xyz.finazon_service.sql_service" not in sys.modules:
    try:
        import xyz.finazon_service.sql_service  # noqa: F401
    except Exception:
        stub = MagicMock()
        stub.SessionLocal = MagicMock()
        sys.modules["xyz.finazon_service.sql_service"] = stub


# ---------------------------------------------------------------------------
# DB helpers — in-memory SQLite with tenant + polygon tables.
# ---------------------------------------------------------------------------

def _make_db():
    """Create an in-memory SQLite DB with tenant + polygon tables."""
    from xyz.tenant.models import Base as TenantBase
    from xyz.finazon_service.base import Base as FinazonBase
    import xyz.polygon_service.models  # noqa: register tables

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TenantBase.metadata.create_all(engine)
    FinazonBase.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return engine, Session


# ---------------------------------------------------------------------------
# Seeding helpers — keep tests readable.
# ---------------------------------------------------------------------------

def _seed_firm_and_account(session, *, firm_id: int = 1, account_id: int = 10) -> int:
    """Insert a Firm → Client → Account chain.

    Encrypted columns (Client.name) get sealed by the EncryptedString
    type adapter — the autouse Fernet fixture in conftest.py provides
    the key.
    """
    from xyz.tenant.models import Account, Client, Firm

    firm = Firm(id=firm_id, name=f"Firm {firm_id}")
    session.add(firm)
    session.flush()

    client = Client(id=firm_id * 100, firm_id=firm_id, name="Test Client")
    session.add(client)
    session.flush()

    account = Account(
        id=account_id,
        client_id=client.id,
        nickname="Test Account",
        account_type="TAXABLE",
    )
    session.add(account)
    session.commit()
    return account.id


def _seed_strategy_and_deployment(
    session,
    *,
    firm_id: int,
    account_id: int,
    deployment_state: str = "ACTIVE",
    dsl: dict | None = None,
    strategy_id: int = 1,
    deployment_id: int = 1,
) -> tuple[int, int]:
    """Insert a Strategy + Deployment.

    The User row for ``author_user_id`` is also seeded so the FK holds.
    Returns (strategy_id, deployment_id).
    """
    from xyz.tenant.models import (
        Deployment,
        DeploymentState,
        Strategy,
        StrategyKind,
        StrategyState,
        User,
        UserRole,
        hash_email,
    )

    user_email = f"user{firm_id}@example.com"
    user = User(
        id=firm_id * 1000,
        firm_id=firm_id,
        email=user_email,
        email_hash=hash_email(user_email),
        hashed_password="x",
        role=UserRole.ADVISOR,
    )
    session.add(user)
    session.flush()

    strategy = Strategy(
        id=strategy_id,
        firm_id=firm_id,
        author_user_id=user.id,
        name="Test CC Strategy",
        version=1,
        kind=StrategyKind.DECLARATIVE,
        dsl_json=dsl or _covered_call_dsl(firm_id, user.id),
        state=StrategyState.PUBLISHED,
    )
    session.add(strategy)
    session.flush()

    deployment = Deployment(
        id=deployment_id,
        strategy_id=strategy.id,
        strategy_version=1,
        account_id=account_id,
        state=DeploymentState(deployment_state),
    )
    session.add(deployment)
    session.commit()
    return strategy.id, deployment.id


def _covered_call_dsl(firm_id: int, author_user_id: int) -> dict:
    """Minimal valid covered-call DSL."""
    return {
        "kind": "declarative",
        "name": "Test CC Strategy",
        "version": 1,
        "author_user_id": author_user_id,
        "firm_id": firm_id,
        "template": "covered_call",
        "selection": {"universe": ["AAPL"]},
        "trigger": {},
        "action": {
            "leg": "short_call",
            "delta_short_target": 0.30,
            "dte_min": 25,
            "dte_max": 50,
        },
        "exit": {"max_profit_pct_close": 0.50},
        "risk_box": {
            "execution_microstructure": {
                "max_slippage_tolerance_pct": 0.005,
            },
        },
        "autonomy_requirement": {
            "OPEN": "L2",
            "CLOSE": "L3",
            "ROLL": "L3",
            "HEDGE": "L2",
            "PORTFOLIO": "L2",
            "EVENT": "L4",
            "STATE": "L4",
        },
    }


def _seed_long_shares(session, account_id: int, symbol: str = "AAPL", qty: int = 100) -> None:
    """Insert an EQUITY position so the trigger's "≥100 long shares" passes."""
    from xyz.tenant.models import AssetClass, LotMethod, Position

    position = Position(
        account_id=account_id,
        symbol=symbol,
        asset_class=AssetClass.EQUITY,
        qty=Decimal(qty),
        cost_basis=Decimal("150.00"),
        lot_method=LotMethod.FIFO,
    )
    session.add(position)
    session.commit()


def _seed_short_call(
    session,
    account_id: int,
    symbol: str = "AAPL",
    *,
    expiry: date,
    strike: float = 180.0,
) -> None:
    """Insert an OPTION position representing an open short call."""
    from xyz.tenant.models import AssetClass, OptionType, LotMethod, Position

    position = Position(
        account_id=account_id,
        symbol=symbol,
        asset_class=AssetClass.OPTION,
        qty=Decimal("-1"),
        cost_basis=Decimal("2.00"),
        lot_method=LotMethod.FIFO,
        option_type=OptionType.CALL,
        strike=Decimal(str(strike)),
        expiry=expiry,
    )
    session.add(position)
    session.commit()


def _seed_chain_snapshot(
    session,
    *,
    symbol: str = "AAPL",
    asof: datetime,
    contracts: list[dict] | None = None,
) -> None:
    """Insert a small option_chains snapshot.

    Each ``contracts`` dict carries the fields the agent reads:
    contract_ticker, expiry, strike, delta, bid, ask, mid.  ``contracts``
    defaults to a single near-30Δ tradable call.
    """
    from xyz.polygon_service.models import OptionChains

    if contracts is None:
        contracts = [
            {
                "contract_ticker": "O:AAPL250117C00185000",
                "expiry": asof.date() + timedelta(days=35),
                "strike": 185.0,
                "delta": 0.30,
                "bid": 2.40,
                "ask": 2.60,
                "mid": 2.50,
            }
        ]

    for c in contracts:
        row = OptionChains(
            underlying=symbol,
            asof_at=asof,
            contract_ticker=c["contract_ticker"],
            expiry=c["expiry"],
            strike=Decimal(str(c["strike"])),
            option_type="CALL",
            bid=Decimal(str(c["bid"])),
            ask=Decimal(str(c["ask"])),
            mid=Decimal(str(c["mid"])),
            delta=Decimal(str(c["delta"])),
        )
        session.add(row)
    session.commit()


def _seed_autonomy(
    session,
    account_id: int,
    *,
    family: str = "OPEN",
    level: str = "L2",
) -> None:
    """Insert an AccountAutonomy row for the given family."""
    from xyz.tenant.models import AccountAutonomy, ActionFamily, AutonomyLevel

    row = AccountAutonomy(
        account_id=account_id,
        action_family=ActionFamily(family),
        level=AutonomyLevel(level),
    )
    session.add(row)
    session.commit()


def _make_agent(Session, now: datetime | None = None):
    """Build a ProposeAgent with an in-memory session factory + frozen clock."""
    from xyz.agents.propose import ProposeAgent

    return ProposeAgent(
        db_session_factory=Session,
        now=(lambda: now) if now is not None else None,
    )


# ===========================================================================
# 1. Trigger hit — ticket has all required fields  (acceptance criterion 2)
# ===========================================================================

def test_propose_emits_ticket_with_required_fields():
    """When the trigger is satisfied, the response contains exactly one
    ticket with every spec-required field populated:
      - leaf_action, action_family, risk_class
      - account_id, deployment_id
      - order_ticket_json with contract, limit_price, side, qty
      - autonomy_level_required + autonomy_level_account
      - reg_bi_rationale (string, non-empty)
    """
    from xyz.agents.schemas import ProposeArtifact, ProposeInput

    _, Session = _make_db()
    db = Session()
    try:
        account_id = _seed_firm_and_account(db, firm_id=1, account_id=10)
        _, deployment_id = _seed_strategy_and_deployment(
            db, firm_id=1, account_id=account_id, deployment_state="ACTIVE"
        )
        _seed_long_shares(db, account_id)
        _seed_autonomy(db, account_id, family="OPEN", level="L2")
        now = datetime(2025, 1, 15, 14, 30, tzinfo=timezone.utc)
        _seed_chain_snapshot(db, asof=now)
    finally:
        db.close()

    agent = _make_agent(Session, now=now)
    artifact = agent.run(ProposeInput(
        firm_id=1,
        deployment_id=deployment_id,
        actor_user_id=None,
    ))

    assert isinstance(artifact, ProposeArtifact)
    assert artifact.reason is None
    assert len(artifact.tickets) == 1

    t = artifact.tickets[0]
    assert t.leaf_action == "OPEN_NEW"
    assert t.action_family == "OPEN"
    assert t.risk_class == "RISK_INCREASING"
    assert t.account_id == account_id
    assert t.deployment_id == deployment_id
    assert t.autonomy_level_required == "L2"
    assert t.autonomy_level_account == "L2"
    assert "Covered call against existing long position in AAPL" in t.reg_bi_rationale

    o = t.order_ticket_json
    assert o["symbol"] == "AAPL"
    assert o["contract_ticker"] == "O:AAPL250117C00185000"
    assert o["option_type"] == "CALL"
    assert o["side"] == "sell_to_open"
    assert o["qty"] == 1
    assert o["strike"] == 185.0
    assert o["delta"] == pytest.approx(0.30)
    # limit_price = mid - slippage_pct * mid = 2.50 - 0.005*2.50 = 2.4875.
    assert o["limit_price"] == pytest.approx(2.4875, rel=1e-4)
    assert o["limit_side"] == "sell"
    assert o["max_slippage_tolerance_pct"] == 0.005
    assert o["strategy_version"] == 1


# ===========================================================================
# 2. Engine event chain: ticket.proposed emitted (acceptance criterion 3)
# ===========================================================================

def test_propose_emits_ticket_proposed_event():
    """After a trigger hit, exactly one ``ticket.proposed`` event lands
    in the engine's audit chain for the firm, with payload mirroring
    the artifact's ticket.

    Per spec §5, ``ticket.proposed`` is owned by the engine-side PROPOSE
    subagent — the server's POST /trades emits ``ticket.persisted``
    instead (covered by tests/test_trades.py).  This split keeps audit
    replay queries returning one row per proposal, not two.
    """
    from xyz.agents.schemas import ProposeInput
    from xyz.tenant.models import Event

    _, Session = _make_db()
    db = Session()
    try:
        account_id = _seed_firm_and_account(db, firm_id=99, account_id=10)
        _, deployment_id = _seed_strategy_and_deployment(
            db, firm_id=99, account_id=account_id
        )
        _seed_long_shares(db, account_id)
        now = datetime(2025, 1, 15, 14, 30, tzinfo=timezone.utc)
        _seed_chain_snapshot(db, asof=now)
    finally:
        db.close()

    agent = _make_agent(Session, now=now)
    agent.run(ProposeInput(
        firm_id=99,
        deployment_id=deployment_id,
        actor_user_id=7,
    ))

    sess = Session()
    try:
        events = sess.query(Event).filter_by(
            kind="ticket.proposed", firm_id=99
        ).all()
        assert len(events) == 1
        ev = events[0]
        assert ev.actor_user_id == 7
        assert ev.payload["account_id"] == account_id
        assert ev.payload["deployment_id"] == deployment_id
        assert ev.payload["leaf_action"] == "OPEN_NEW"
        assert ev.payload["action_family"] == "OPEN"
        assert ev.payload["risk_class"] == "RISK_INCREASING"
        assert ev.payload["order_ticket_json"]["symbol"] == "AAPL"
    finally:
        sess.close()


# ===========================================================================
# 3. Trigger miss — deployment not ACTIVE (acceptance criterion 1)
# ===========================================================================

def test_propose_with_inactive_deployment_returns_empty():
    """Spec §9: only ACTIVE deployments propose.  A PAUSED deployment
    returns ``tickets=[]`` with reason ``deployment_not_active`` — NOT
    an error.  No event is emitted (audit log stays clean of no-ops)."""
    from xyz.agents.schemas import ProposeInput
    from xyz.tenant.models import Event

    _, Session = _make_db()
    db = Session()
    try:
        account_id = _seed_firm_and_account(db, firm_id=1, account_id=10)
        _, deployment_id = _seed_strategy_and_deployment(
            db, firm_id=1, account_id=account_id, deployment_state="PAUSED"
        )
        _seed_long_shares(db, account_id)
        now = datetime(2025, 1, 15, 14, 30, tzinfo=timezone.utc)
        _seed_chain_snapshot(db, asof=now)
    finally:
        db.close()

    agent = _make_agent(Session, now=now)
    artifact = agent.run(ProposeInput(firm_id=1, deployment_id=deployment_id))

    assert artifact.tickets == []
    assert artifact.reason == "deployment_not_active"

    # No ticket.proposed event for a no-op.
    sess = Session()
    try:
        events = sess.query(Event).filter_by(kind="ticket.proposed").all()
        assert events == []
    finally:
        sess.close()


# ===========================================================================
# 4. Trigger miss — no long shares (acceptance criterion 1)
# ===========================================================================

def test_propose_with_no_long_shares_returns_empty():
    """The covered-call trigger requires ≥100 long shares of the
    underlying.  Without that, the agent returns ``tickets=[]`` with
    reason ``insufficient_long_shares``.
    """
    from xyz.agents.schemas import ProposeInput

    _, Session = _make_db()
    db = Session()
    try:
        account_id = _seed_firm_and_account(db, firm_id=1, account_id=10)
        _, deployment_id = _seed_strategy_and_deployment(
            db, firm_id=1, account_id=account_id
        )
        # No shares seeded.
        now = datetime(2025, 1, 15, 14, 30, tzinfo=timezone.utc)
        _seed_chain_snapshot(db, asof=now)
    finally:
        db.close()

    agent = _make_agent(Session, now=now)
    artifact = agent.run(ProposeInput(firm_id=1, deployment_id=deployment_id))

    assert artifact.tickets == []
    assert artifact.reason == "insufficient_long_shares"


# ===========================================================================
# 5. Trigger miss — existing open short call (acceptance criterion 1)
# ===========================================================================

def test_propose_with_existing_short_call_returns_empty():
    """If the account already holds an open short call against the
    underlying (qty<0, expiry>=today), the trigger does NOT fire — we
    don't stack inadvertent shorts on the same underlying."""
    from xyz.agents.schemas import ProposeInput

    _, Session = _make_db()
    db = Session()
    try:
        account_id = _seed_firm_and_account(db, firm_id=1, account_id=10)
        _, deployment_id = _seed_strategy_and_deployment(
            db, firm_id=1, account_id=account_id
        )
        _seed_long_shares(db, account_id)
        # Existing short call expires in 30 days — still open.
        now = datetime(2025, 1, 15, 14, 30, tzinfo=timezone.utc)
        _seed_short_call(db, account_id, expiry=now.date() + timedelta(days=30))
        _seed_chain_snapshot(db, asof=now)
    finally:
        db.close()

    agent = _make_agent(Session, now=now)
    artifact = agent.run(ProposeInput(firm_id=1, deployment_id=deployment_id))

    assert artifact.tickets == []
    assert artifact.reason == "existing_open_short_call"


# ===========================================================================
# 6. Trigger miss — no chain data for symbol (acceptance criterion 1)
# ===========================================================================

def test_propose_with_no_chain_data_returns_empty():
    """No ``option_chains`` rows for the underlying → ``tickets=[]``
    with reason ``no_tradable_contract``.  This is the cold-start /
    after-hours case in production."""
    from xyz.agents.schemas import ProposeInput

    _, Session = _make_db()
    db = Session()
    try:
        account_id = _seed_firm_and_account(db, firm_id=1, account_id=10)
        _, deployment_id = _seed_strategy_and_deployment(
            db, firm_id=1, account_id=account_id
        )
        _seed_long_shares(db, account_id)
        # No chain rows seeded.
    finally:
        db.close()

    now = datetime(2025, 1, 15, 14, 30, tzinfo=timezone.utc)
    agent = _make_agent(Session, now=now)
    artifact = agent.run(ProposeInput(firm_id=1, deployment_id=deployment_id))

    assert artifact.tickets == []
    assert artifact.reason == "no_tradable_contract"


# ===========================================================================
# 7. Trigger miss — unsupported template (v1 cut)
# ===========================================================================

def test_propose_with_unsupported_template_returns_empty():
    """v1 supports only ``covered_call``.  A cash-secured-put template
    returns ``tickets=[]`` with reason ``template_not_supported_in_v1``
    — same posture as Task 4.3 BACKTEST."""
    from xyz.agents.schemas import ProposeInput

    _, Session = _make_db()
    db = Session()
    try:
        account_id = _seed_firm_and_account(db, firm_id=1, account_id=10)
        # DSL with a non-covered_call template.
        bad_dsl = _covered_call_dsl(firm_id=1, author_user_id=1000)
        bad_dsl["template"] = "cash_secured_put"
        _, deployment_id = _seed_strategy_and_deployment(
            db, firm_id=1, account_id=account_id, dsl=bad_dsl
        )
        _seed_long_shares(db, account_id)
        now = datetime(2025, 1, 15, 14, 30, tzinfo=timezone.utc)
        _seed_chain_snapshot(db, asof=now)
    finally:
        db.close()

    agent = _make_agent(Session, now=now)
    artifact = agent.run(ProposeInput(firm_id=1, deployment_id=deployment_id))

    assert artifact.tickets == []
    assert artifact.reason == "template_not_supported_in_v1"


# ===========================================================================
# 8. Route auth — bearer required
# ===========================================================================

def test_propose_route_rejects_missing_bearer():
    """The /agents/propose endpoint requires a bearer token.

    Uses tests.helpers_app.get_test_client which mirrors the same
    bearer-auth dependency wiring as the real engine app.py.  This
    ensures the test catches a regression where any of the four
    agent endpoints loses its auth dependency (the helper mounts
    /agents/research, /agents/author, /agents/validate-dsl, and
    /agents/propose under the same _verify_bearer guard the
    production app uses)."""
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"KEY": "test-secret"}):
        from tests.helpers_app import get_test_client
        client = get_test_client()
        resp = client.post("/agents/propose", json={
            "firm_id": 1,
            "deployment_id": 1,
        })
    assert resp.status_code in (401, 403)


def test_propose_route_with_invalid_bearer_returns_401():
    """Mirrors the test_*_route_with_invalid_bearer pattern from
    tests for Tasks 4.1 + 4.2."""
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"KEY": "test-secret"}):
        from tests.helpers_app import get_test_client
        client = get_test_client()
        resp = client.post(
            "/agents/propose",
            json={"firm_id": 1, "deployment_id": 1},
            headers={"Authorization": "Bearer wrong-token"},
        )
    assert resp.status_code == 401


# ===========================================================================
# 9. Selection — nearest delta wins, with documented tiebreak
# ===========================================================================

def test_propose_picks_nearest_delta_with_tiebreak():
    """Three candidates: delta 0.20, 0.30, 0.40, all in DTE window.

    Target = 0.30 → distance(0.30) = 0 wins.  Distances for 0.20 and
    0.40 are equal (0.10), but per the tiebreak order (lowest strike
    first) the 0.20-delta row would lose to the 0.40-delta one IF the
    strikes were equal — which they aren't (deeper OTM = higher strike
    for calls).  This test mainly asserts the nearest-delta winner is
    picked deterministically.
    """
    from xyz.agents.schemas import ProposeInput

    _, Session = _make_db()
    db = Session()
    try:
        account_id = _seed_firm_and_account(db, firm_id=1, account_id=10)
        _, deployment_id = _seed_strategy_and_deployment(
            db, firm_id=1, account_id=account_id
        )
        _seed_long_shares(db, account_id)

        now = datetime(2025, 1, 15, 14, 30, tzinfo=timezone.utc)
        exp = now.date() + timedelta(days=35)
        contracts = [
            {
                "contract_ticker": "O:AAPL_K200",
                "expiry": exp,
                "strike": 200.0,
                "delta": 0.20,
                "bid": 1.40,
                "ask": 1.60,
                "mid": 1.50,
            },
            {
                "contract_ticker": "O:AAPL_K185",
                "expiry": exp,
                "strike": 185.0,
                "delta": 0.30,  # exact target
                "bid": 2.40,
                "ask": 2.60,
                "mid": 2.50,
            },
            {
                "contract_ticker": "O:AAPL_K170",
                "expiry": exp,
                "strike": 170.0,
                "delta": 0.40,
                "bid": 3.40,
                "ask": 3.60,
                "mid": 3.50,
            },
        ]
        _seed_chain_snapshot(db, asof=now, contracts=contracts)
    finally:
        db.close()

    agent = _make_agent(Session, now=now)
    artifact = agent.run(ProposeInput(firm_id=1, deployment_id=deployment_id))

    assert len(artifact.tickets) == 1
    t = artifact.tickets[0]
    assert t.order_ticket_json["contract_ticker"] == "O:AAPL_K185"
    assert t.order_ticket_json["delta"] == pytest.approx(0.30)


# ===========================================================================
# 10. Autonomy level — recorded but NOT gated by PROPOSE
# ===========================================================================

def test_propose_records_account_autonomy_level():
    """When the account's autonomy level for OPEN is LOWER than the
    strategy's required level, PROPOSE still emits the ticket (gating
    is COMPLIANCE's job per Task 4.5).  Both levels land in the ticket.
    """
    from xyz.agents.schemas import ProposeInput

    _, Session = _make_db()
    db = Session()
    try:
        account_id = _seed_firm_and_account(db, firm_id=1, account_id=10)
        # Strategy requires L2 for OPEN (default in _covered_call_dsl).
        _, deployment_id = _seed_strategy_and_deployment(
            db, firm_id=1, account_id=account_id
        )
        _seed_long_shares(db, account_id)
        # Account only allows L0 ("notify only") — but PROPOSE still emits.
        _seed_autonomy(db, account_id, family="OPEN", level="L0")
        now = datetime(2025, 1, 15, 14, 30, tzinfo=timezone.utc)
        _seed_chain_snapshot(db, asof=now)
    finally:
        db.close()

    agent = _make_agent(Session, now=now)
    artifact = agent.run(ProposeInput(firm_id=1, deployment_id=deployment_id))

    assert len(artifact.tickets) == 1
    t = artifact.tickets[0]
    assert t.autonomy_level_required == "L2"
    assert t.autonomy_level_account == "L0"
    # COMPLIANCE (Task 4.5) will block this at verdict time; PROPOSE
    # is intentionally permissive.
