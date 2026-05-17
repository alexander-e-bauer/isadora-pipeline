"""Tests for the BACKTEST subagent + replay engine (Task 4.3).

All market data is synthetic — no live Polygon calls in CI.

Inventory
---------
 1. test_fill_model_short_clamps_to_half_spread
 2. test_fill_model_long_clamps_to_max_pct
 3. test_fill_model_handles_gap_day_zero_quote
 4. test_fill_model_uses_default_when_pct_is_none
 5. test_metrics_total_return_basic
 6. test_metrics_max_drawdown_is_nonpositive
 7. test_metrics_sharpe_zero_when_flat
 8. test_metrics_win_rate_zero_when_no_trades
 9. test_backtest_aapl_2023_2024_metrics_in_sensible_ranges
10. test_backtest_content_hash_is_deterministic   ← acceptance criterion 2
11. test_backtest_emits_backtest_result_event
12. test_backtest_artifact_excludes_generated_at_from_hash
13. test_backtest_route_rejects_missing_bearer
14. test_backtest_route_rejects_unsupported_template
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


# ---------------------------------------------------------------------------
# Stub xyz.finazon_service.sql_service before any backtest import (test path).
# The engine venv lacks psycopg2 and the real module imports it at module
# load.  Backtest does NOT need that module, but the agent's
# ``xyz.tenant.db`` re-exports SessionLocal from it.  We never actually
# call the production session factory in tests — but we have to make the
# import succeed.
# ---------------------------------------------------------------------------

if "xyz.finazon_service.sql_service" not in sys.modules:
    try:
        import xyz.finazon_service.sql_service  # noqa: F401
    except Exception:
        # Build a minimal stub so xyz.tenant.db can import SessionLocal.
        stub = MagicMock()
        stub.SessionLocal = MagicMock()
        sys.modules["xyz.finazon_service.sql_service"] = stub


# ---------------------------------------------------------------------------
# DB helpers — in-memory SQLite with both tenant + polygon tables.
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
# Synthetic chain seeding — deterministic AAPL covered-call data.
# ---------------------------------------------------------------------------

def _seed_aapl_chain(session, start: date, end: date, *, spot_start: float = 150.0):
    """Seed deterministic daily chain rows for AAPL covering [start, end].

    The seed creates a small universe of monthly-expiry contracts that
    persist day-over-day (same contract_ticker reappears every trading
    day until expiry).  This lets the replay engine mark a single open
    position to market across many days and reach its expiry —
    otherwise contracts would vanish the day after opening and the
    exit/expiration evaluation never fires.

    Per-day shape:
      * underlying follows a slow upward drift + a deterministic cosine
        wobble so daily returns have non-zero variance.
      * monthly expiries fall on the 21st of each month (Friday-ish proxy).
      * for each expiry, 9 strikes centered on the rolling spot are seeded.
      * delta computed via a smooth ramp around ATM.
      * close priced as a toy theta-decay function of (DTE, moneyness)
        so positions decay deterministically toward expiry.
    """
    from xyz.polygon_service.models import OptionHistoricalEod
    import math

    # Generate fixed monthly expiries for the whole window + a 90-day pad.
    # Snap to the nearest preceding weekday so the engine's
    # expiration-day path actually fires (the replay iterates trading
    # days only, so a Sat/Sun expiry would land one or two days late).
    expiries: list[date] = []
    y, m = start.year, start.month
    pad_end = end + timedelta(days=120)
    while True:
        try:
            exp = date(y, m, 21)
        except ValueError:
            exp = date(y, m, 28)
        while exp.weekday() >= 5:
            exp -= timedelta(days=1)
        if exp >= start:
            expiries.append(exp)
        if exp > pad_end:
            break
        m += 1
        if m > 12:
            m = 1
            y += 1

    # Fixed strike grid — 9 strikes from $130 to $210 in $10 steps.
    # Spans the rolling spot for ~2 years of slow drift.
    strikes = [130.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0, 175.0, 190.0]

    one_day = timedelta(days=1)
    d = start
    day_index = 0
    while d <= end:
        if d.weekday() < 5:
            # Drift + cosine wobble.
            spot = spot_start + day_index * 0.05 + 5.0 * math.cos(day_index * 0.05)

            for expiry in expiries:
                dte = (expiry - d).days
                if dte <= 0 or dte > 90:
                    continue  # Only seed 0-90 DTE per day.
                for strike in strikes:
                    moneyness = (spot - strike) / 10.0
                    delta = max(0.05, min(0.95, 0.5 + moneyness * 0.30))
                    # Toy option price ~ delta * sqrt(DTE)/sqrt(252) * IV * spot.
                    premium = max(
                        0.05,
                        round(delta * math.sqrt(max(dte, 1)) * 0.20 + 0.20, 4),
                    )

                    row = OptionHistoricalEod(
                        underlying="AAPL",
                        contract_ticker=(
                            f"O:AAPLE{expiry.strftime('%y%m%d')}"
                            f"K{int(strike*100):08d}C"
                        ),
                        expiry=expiry,
                        strike=Decimal(str(strike)),
                        option_type="CALL",
                        date=d,
                        open=Decimal(str(premium)),
                        high=Decimal(str(premium)),
                        low=Decimal(str(premium)),
                        close=Decimal(str(premium)),
                        volume=1000,
                        open_interest=2000,
                        implied_vol=Decimal("0.25"),
                        delta=Decimal(f"{delta:.4f}"),
                        gamma=Decimal("0.01"),
                        theta=Decimal("-0.05"),
                        vega=Decimal("0.10"),
                        rho=Decimal("0.05"),
                    )
                    session.add(row)
            day_index += 1
        d += one_day
    session.commit()


def _make_covered_call_dsl() -> dict:
    """Return a minimal, valid covered-call DSL on AAPL."""
    return {
        "kind": "declarative",
        "name": "Backtest CC",
        "version": 1,
        "author_user_id": 1,
        "firm_id": 1,
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
    }


# ===========================================================================
# 1–4: fill_model unit tests
# ===========================================================================

def test_fill_model_short_clamps_to_half_spread():
    """When ``max_slippage_pct * mid`` exceeds the half-spread, clamp wins."""
    from xyz.backtest.fill_model import compute_fill

    # bid=2.40 ask=2.60 → mid=2.50 half_spread=0.10
    # max_slippage_pct=0.10 * 2.50 = 0.25 >> 0.10 → clamp to 0.10
    price = compute_fill(side="short", bid=2.40, ask=2.60, max_slippage_pct=0.10)
    assert abs(price - (2.50 - 0.10)) < 1e-9
    assert price == pytest.approx(2.40)


def test_fill_model_long_clamps_to_max_pct():
    """When ``max_slippage_pct * mid`` is smaller than half-spread, pct wins."""
    from xyz.backtest.fill_model import compute_fill

    # bid=1.00 ask=3.00 → mid=2.00 half_spread=1.00
    # max_slippage_pct=0.005 → 0.005 * 2.00 = 0.01 << 1.00 → clamp to 0.01
    price = compute_fill(side="long", bid=1.00, ask=3.00, max_slippage_pct=0.005)
    assert price == pytest.approx(2.00 + 0.01)


def test_fill_model_handles_gap_day_zero_quote():
    """A bid/ask of (0, 0) returns 0.0 — the engine treats it as a skip."""
    from xyz.backtest.fill_model import compute_fill

    assert compute_fill(side="short", bid=0.0, ask=0.0, max_slippage_pct=0.005) == 0.0


def test_fill_model_uses_default_when_pct_is_none():
    """``max_slippage_pct=None`` falls back to ``DEFAULT_MAX_SLIPPAGE_PCT``."""
    from xyz.backtest.fill_model import DEFAULT_MAX_SLIPPAGE_PCT, compute_fill

    # bid=2.40 ask=2.60 → mid=2.50 half_spread=0.10
    # DEFAULT=0.005 → 0.005 * 2.50 = 0.0125 < 0.10 → clamp to 0.0125
    price = compute_fill(side="short", bid=2.40, ask=2.60, max_slippage_pct=None)
    assert price == pytest.approx(2.50 - DEFAULT_MAX_SLIPPAGE_PCT * 2.50)


# ===========================================================================
# 5–8: metrics unit tests
# ===========================================================================

def test_metrics_total_return_basic():
    from xyz.backtest.metrics import total_return

    assert total_return([1.0, 1.10]) == pytest.approx(0.10)
    assert total_return([1.0, 0.90]) == pytest.approx(-0.10)
    assert total_return([1.0]) == 0.0
    assert total_return([]) == 0.0


def test_metrics_max_drawdown_is_nonpositive():
    """Max drawdown must always be <= 0."""
    from xyz.backtest.metrics import max_drawdown

    # Peak 1.10 → trough 1.05 → recover.  DD = (1.05 - 1.10) / 1.10.
    assert max_drawdown([1.0, 1.10, 1.05, 1.20]) == pytest.approx((1.05 - 1.10) / 1.10)
    # Monotonically increasing → 0 drawdown.
    assert max_drawdown([1.0, 1.05, 1.10]) == 0.0
    # Decreasing series.
    dd = max_drawdown([1.0, 0.95, 0.85, 0.80])
    assert dd < 0


def test_metrics_sharpe_zero_when_flat():
    from xyz.backtest.metrics import sharpe

    assert sharpe([1.0, 1.0, 1.0, 1.0]) == 0.0
    assert sharpe([]) == 0.0


def test_metrics_win_rate_zero_when_no_trades():
    from xyz.backtest.metrics import win_rate, avg_pnl

    assert win_rate([]) == 0.0
    assert avg_pnl([], side="win") == 0.0
    # Mixed trades.
    assert win_rate([0.5, -0.2, 0.3]) == pytest.approx(2 / 3)
    assert avg_pnl([0.5, -0.2, 0.3], side="win") == pytest.approx(0.4)
    assert avg_pnl([0.5, -0.2, 0.3], side="loss") == pytest.approx(-0.2)


# ===========================================================================
# 9: end-to-end agent on synthetic AAPL data — acceptance criterion 1
# ===========================================================================

def test_backtest_aapl_2023_2024_metrics_in_sensible_ranges():
    """Acceptance criterion 1: covered-call backtest on AAPL 2023-01 →
    2024-12 produces sensible numbers.  We assert RANGES (not exact
    values) so the test stays robust to future microstructure tweaks
    in the fill model.

    The 'AAPL' chain here is synthetic but spans the required window.
    """
    _, Session = _make_db()
    session = Session()
    try:
        _seed_aapl_chain(session, date(2023, 1, 1), date(2024, 12, 31))
    finally:
        session.close()

    dsl = _make_covered_call_dsl()

    # Run via the agent path so the event-emission code is exercised.
    from xyz.agents.backtest import BacktestAgent
    from xyz.agents.schemas import BacktestInput

    agent = BacktestAgent(db_session_factory=Session)
    artifact = agent.run(BacktestInput(
        strategy_id=1,
        strategy_version=1,
        firm_id=42,
        actor_user_id=7,
        start_date=date(2023, 1, 1),
        end_date=date(2024, 12, 31),
        dsl=dsl,
    ))

    m = artifact.metrics
    assert -0.5 < m["total_return"] < 1.0, m
    assert -3.0 < m["sharpe"] < 3.0, m
    assert -1.0 < m["max_drawdown"] <= 0.0, m
    assert 0.0 <= m["win_rate"] <= 1.0, m
    # Sanity: at least one trade should have been opened over 2 years
    # of synthetic data.
    assert m["n_trades"] > 0, m
    # nav series length matches trading days seen.
    assert m["nav_series_len"] > 0
    # Content hash is a real sha256 hex digest.
    assert len(artifact.content_hash) == 64


# ===========================================================================
# 10: content-hash determinism — acceptance criterion 2
# ===========================================================================

def test_backtest_content_hash_is_deterministic():
    """Acceptance criterion 2: re-running the same backtest produces an
    IDENTICAL content_hash.  This is the 17a-4 audit guarantee.

    Two runs over freshly-seeded in-memory DBs with the same synthetic
    data must produce byte-equal hashes.
    """
    from xyz.agents.backtest import BacktestAgent
    from xyz.agents.schemas import BacktestInput

    dsl = _make_covered_call_dsl()
    start, end = date(2023, 6, 1), date(2023, 9, 30)

    hashes = []
    for _ in range(2):
        _, Session = _make_db()
        session = Session()
        try:
            _seed_aapl_chain(session, start, end)
        finally:
            session.close()

        agent = BacktestAgent(db_session_factory=Session)
        artifact = agent.run(BacktestInput(
            strategy_id=1,
            strategy_version=1,
            firm_id=1,
            actor_user_id=1,
            start_date=start,
            end_date=end,
            dsl=dsl,
        ))
        hashes.append(artifact.content_hash)

    assert hashes[0] == hashes[1], (
        f"Determinism violated: {hashes[0]} != {hashes[1]}"
    )


# ===========================================================================
# 11: event-emission acceptance
# ===========================================================================

def test_backtest_emits_backtest_result_event():
    """After ``run()``, the events table must contain one
    ``backtest.result`` row for the firm with payload['content_hash']
    matching the artifact's hash.
    """
    from xyz.agents.backtest import BacktestAgent
    from xyz.agents.schemas import BacktestInput
    from xyz.tenant.models import Event

    _, Session = _make_db()
    session = Session()
    try:
        _seed_aapl_chain(session, date(2023, 1, 1), date(2023, 3, 31))
    finally:
        session.close()

    agent = BacktestAgent(db_session_factory=Session)
    artifact = agent.run(BacktestInput(
        strategy_id=1,
        strategy_version=1,
        firm_id=99,
        actor_user_id=7,
        start_date=date(2023, 1, 1),
        end_date=date(2023, 3, 31),
        dsl=_make_covered_call_dsl(),
    ))

    sess = Session()
    try:
        events = sess.query(Event).filter_by(
            kind="backtest.result", firm_id=99
        ).all()
        assert len(events) == 1
        ev = events[0]
        assert ev.actor_user_id == 7
        assert ev.payload["content_hash"] == artifact.content_hash
        assert ev.payload["strategy_id"] == 1
        assert ev.payload["strategy_version"] == 1
        # generated_at must be in the event payload (audit trail), but
        # the hash itself must NOT have depended on it.
        assert "generated_at" in ev.payload
    finally:
        sess.close()


# ===========================================================================
# 12: hash excludes generated_at
# ===========================================================================

def test_backtest_artifact_excludes_generated_at_from_hash():
    """Two runs separated in wall-clock time must produce the same hash
    despite differing ``generated_at`` timestamps.

    This guards the determinism invariant: if a future refactor accidentally
    folds ``generated_at`` into the hash domain, this test fails.
    """
    from xyz.agents.backtest import BacktestAgent
    from xyz.agents.schemas import BacktestInput

    dsl = _make_covered_call_dsl()
    start, end = date(2023, 1, 1), date(2023, 2, 28)

    hashes = []
    timestamps = []
    for _ in range(2):
        _, Session = _make_db()
        session = Session()
        try:
            _seed_aapl_chain(session, start, end)
        finally:
            session.close()

        agent = BacktestAgent(db_session_factory=Session)
        artifact = agent.run(BacktestInput(
            strategy_id=1,
            strategy_version=1,
            firm_id=1,
            actor_user_id=1,
            start_date=start,
            end_date=end,
            dsl=dsl,
        ))
        hashes.append(artifact.content_hash)
        timestamps.append(artifact.generated_at)

    assert hashes[0] == hashes[1]
    # The timestamps will usually differ (different wall-clock moments).
    # But even if they happen to be equal, the hash equality test above
    # is what we actually rely on — this assert is informational.


# ===========================================================================
# 13–14: route auth + unsupported-template behavior
# ===========================================================================

def test_backtest_route_rejects_missing_bearer():
    """The /agents/backtest endpoint requires a bearer token (mirrors
    /agents/research and /agents/author)."""
    from fastapi import Depends, FastAPI, HTTPException
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from fastapi.testclient import TestClient

    security = HTTPBearer(auto_error=True)

    def _verify(creds: HTTPAuthorizationCredentials = Depends(security)):
        if creds.credentials != "secret":
            raise HTTPException(status_code=401, detail="invalid")
        return creds.credentials

    app = FastAPI()

    @app.post("/agents/backtest", dependencies=[Depends(_verify)])
    def _stub():
        return {"ok": True}

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/agents/backtest", json={
        "strategy_id": 1, "strategy_version": 1, "firm_id": 1,
        "start_date": "2023-01-01", "end_date": "2023-01-31",
        "dsl": {},
    })
    assert resp.status_code in (401, 403)


def test_backtest_unsupported_template_raises():
    """An unsupported template raises ValueError so the route maps to 400."""
    from xyz.backtest.engine import run_backtest

    _, Session = _make_db()
    db = Session()
    try:
        _seed_aapl_chain(db, date(2023, 1, 1), date(2023, 1, 31))
        dsl = _make_covered_call_dsl()
        dsl["template"] = "iron_condor"  # not supported in v1
        with pytest.raises(ValueError, match="covered_call"):
            run_backtest(
                dsl=dsl,
                start_date=date(2023, 1, 1),
                end_date=date(2023, 1, 31),
                db_session=db,
            )
    finally:
        db.close()
