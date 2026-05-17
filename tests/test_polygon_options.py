"""Tests for the Polygon options data layer.

All Polygon HTTP calls are mocked — no live API calls in CI.
Uses respx to intercept httpx requests.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import respx
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "polygon"


def _load(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


# ---------------------------------------------------------------------------
# ORM / Base import guard — must happen before any polygon models are used
# ---------------------------------------------------------------------------

def _make_sqlite_session():
    """Return a (engine, Session) pair backed by an in-memory SQLite DB
    with all polygon_service tables created.

    Imports Base from xyz.finazon_service.base (the thin module) rather than
    xyz.finazon_service.sql_service so that psycopg2 and the live Postgres
    engine are never imported during tests.
    """
    from sqlalchemy import create_engine as _ce
    from sqlalchemy.orm import sessionmaker as _sm
    from sqlalchemy.pool import StaticPool as _SP

    # Import the thin Base first, then register polygon models on it.
    from xyz.finazon_service.base import Base as FBase
    import xyz.polygon_service.models  # noqa: register tables on Base

    eng = _ce(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=_SP,
    )
    FBase.metadata.create_all(eng)
    Session = _sm(bind=eng, autocommit=False, autoflush=False)
    return eng, Session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def polygon_db_session():
    """Fresh in-memory SQLite session per test — ensures test isolation."""
    eng, Session = _make_sqlite_session()
    session = Session()
    try:
        yield session
    finally:
        session.close()
        eng.dispose()


@pytest.fixture()
def rate_limiter_unlimited():
    """RateLimiter with very high capacity so tests don't wait."""
    from xyz.polygon_service.rate_limit import RateLimiter
    return RateLimiter(rate_per_min=100_000)


@pytest.fixture()
def chain_cache_fresh():
    from xyz.polygon_service.cache import ChainCache
    return ChainCache(ttl_seconds=300)


@pytest.fixture()
def client(rate_limiter_unlimited, chain_cache_fresh):
    from xyz.polygon_service.options_client import OptionsClient
    return OptionsClient(
        api_key="TEST_KEY",
        rate_limiter=rate_limiter_unlimited,
        cache=chain_cache_fresh,
    )


# ---------------------------------------------------------------------------
# Test 1 — get_chain parses Polygon v3 snapshot response
# ---------------------------------------------------------------------------

@respx.mock
def test_options_client_get_chain_parses_polygon_response(client):
    """Chain snapshot endpoint is parsed into a ChainSnapshot with correct contracts."""
    body = _load("chain_aapl.json")
    respx.get("https://api.polygon.io/v3/snapshot/options/AAPL").mock(
        return_value=httpx.Response(200, json=body)
    )

    snapshot = client.get_chain("AAPL")

    assert snapshot.underlying == "AAPL"
    assert snapshot.underlying_price == pytest.approx(185.0)
    assert len(snapshot.contracts) == 3

    call_185 = next(c for c in snapshot.contracts if c.strike == 185.0 and c.option_type == "CALL")
    assert call_185.contract_ticker == "O:AAPL250117C00185000"
    assert call_185.expiry == date(2025, 1, 17)
    assert call_185.implied_vol == pytest.approx(0.285)
    assert call_185.delta == pytest.approx(0.52)
    assert call_185.bid == pytest.approx(2.50)
    assert call_185.ask == pytest.approx(2.60)
    assert call_185.mid == pytest.approx(2.55)

    put_182 = next(c for c in snapshot.contracts if c.option_type == "PUT")
    assert put_182.option_type == "PUT"
    assert put_182.strike == pytest.approx(182.0)


# ---------------------------------------------------------------------------
# Test 2 — get_chain caches; second call does not hit HTTP
# ---------------------------------------------------------------------------

@respx.mock
def test_options_client_caches_chain_snapshot(client):
    """Second get_chain call within TTL window is served from cache."""
    body = _load("chain_aapl.json")
    route = respx.get("https://api.polygon.io/v3/snapshot/options/AAPL").mock(
        return_value=httpx.Response(200, json=body)
    )

    snapshot1 = client.get_chain("AAPL")
    snapshot2 = client.get_chain("AAPL")

    # HTTP should have been called exactly once.
    assert route.call_count == 1
    assert len(snapshot2.contracts) == len(snapshot1.contracts)


# ---------------------------------------------------------------------------
# Test 3 — RateLimiter blocks when bucket is empty
# ---------------------------------------------------------------------------

def test_options_client_rate_limiter_blocks_when_bucket_empty():
    """With rate_per_min=2 the third acquire() must sleep."""
    from xyz.polygon_service.rate_limit import RateLimiter

    slept: list[float] = []
    real_sleep = time.sleep

    def fake_sleep(secs: float) -> None:
        slept.append(secs)
        # Actually advance monotonic by advancing real time a little so refill
        # happens, but just record the intended sleep duration.
        real_sleep(0)  # yield without long wait

    rl = RateLimiter(rate_per_min=2)

    with patch("xyz.polygon_service.rate_limit.time.sleep", side_effect=fake_sleep):
        # First two acquires consume all tokens immediately (bucket starts full).
        rl.acquire()
        rl.acquire()
        # Manually drain remaining tokens to ensure bucket is empty.
        rl._tokens = 0.0

        # Third acquire must observe an empty bucket and call sleep.
        # We patch monotonic to avoid flakiness: make time appear not to advance
        # so refill doesn't happen before sleep is called.
        original_mono = time.monotonic
        call_count = [0]

        def frozen_mono():
            call_count[0] += 1
            # After the sleep fake is called, let time advance a lot so refill
            # completes on the next iteration.
            if slept:
                return original_mono() + 100.0
            return original_mono()

        with patch("xyz.polygon_service.rate_limit.time.monotonic", side_effect=frozen_mono):
            rl.acquire()

    assert len(slept) >= 1, "Expected at least one sleep() call when bucket was empty"


# ---------------------------------------------------------------------------
# Test 4 — get_last_quote round-trip
# ---------------------------------------------------------------------------

@respx.mock
def test_options_client_get_last_quote(client):
    """get_last_quote parses bid/ask/timestamp from Polygon v3 quotes endpoint."""
    body = _load("quote_aapl_call.json")
    contract = "O:AAPL250117C00185000"
    respx.get(f"https://api.polygon.io/v3/quotes/{contract}").mock(
        return_value=httpx.Response(200, json=body)
    )

    quote = client.get_last_quote(contract)

    assert quote.contract_ticker == contract
    assert quote.bid == pytest.approx(2.50)
    assert quote.ask == pytest.approx(2.60)
    assert quote.bid_size == 45
    assert quote.ask_size == 50
    assert isinstance(quote.timestamp, datetime)


# ---------------------------------------------------------------------------
# Test 5 — get_historical_aggs response parsing
# ---------------------------------------------------------------------------

@respx.mock
def test_options_client_get_historical_aggs(client):
    """Historical aggs are parsed into Aggregate objects with correct fields."""
    body = _load("historical_aapl_2024.json")
    contract = "O:AAPL250117C00185000"
    respx.get(
        f"https://api.polygon.io/v2/aggs/ticker/{contract}/range/1/day/2024-01-08/2024-01-12"
    ).mock(return_value=httpx.Response(200, json=body))

    from datetime import date as d
    aggs = client.get_historical_aggs(contract, from_date=d(2024, 1, 8), to_date=d(2024, 1, 12))

    assert len(aggs) == 5
    first = aggs[0]
    assert first.contract_ticker == contract
    assert isinstance(first.timestamp, datetime)
    assert first.open == pytest.approx(2.15)
    assert first.close == pytest.approx(2.20)
    assert first.volume == pytest.approx(3800)

    last = aggs[-1]
    assert last.close == pytest.approx(2.55)


# ---------------------------------------------------------------------------
# Test 6 — fetch_and_persist_historical_eod inserts rows
# ---------------------------------------------------------------------------

@respx.mock
def test_fetch_and_persist_historical_eod_inserts_rows(polygon_db_session, rate_limiter_unlimited, chain_cache_fresh):
    """fetch_and_persist_historical_eod inserts 5 rows for a single contract."""
    from xyz.polygon_service.historical import fetch_and_persist_historical_eod
    from xyz.polygon_service.options_client import OptionsClient, OptionContract
    from xyz.polygon_service.models import OptionHistoricalEod

    body = _load("historical_aapl_2024.json")
    contract_ticker = "O:AAPL250117C00185000"
    respx.get(
        f"https://api.polygon.io/v2/aggs/ticker/{contract_ticker}/range/1/day/2024-01-08/2024-01-12"
    ).mock(return_value=httpx.Response(200, json=body))

    c = OptionsClient(
        api_key="TEST_KEY",
        rate_limiter=rate_limiter_unlimited,
        cache=chain_cache_fresh,
    )
    contracts = [
        OptionContract(
            contract_ticker=contract_ticker,
            underlying="AAPL",
            expiry=date(2025, 1, 17),
            strike=185.0,
            option_type="CALL",
        )
    ]

    inserted = fetch_and_persist_historical_eod(
        client=c,
        db=polygon_db_session,
        underlying="AAPL",
        contracts=contracts,
        start_date=date(2024, 1, 8),
        end_date=date(2024, 1, 12),
    )
    polygon_db_session.commit()

    assert inserted == 5
    rows = polygon_db_session.query(OptionHistoricalEod).filter_by(contract_ticker=contract_ticker).all()
    assert len(rows) == 5
    assert rows[0].underlying == "AAPL"
    assert rows[0].option_type == "CALL"
    assert rows[0].strike is not None


# ---------------------------------------------------------------------------
# Test 7 — fetch_and_persist_historical_eod skips duplicates (idempotency)
# ---------------------------------------------------------------------------

@respx.mock
def test_fetch_and_persist_historical_eod_skips_duplicates(polygon_db_session, rate_limiter_unlimited, chain_cache_fresh):
    """Re-running fetch_and_persist_historical_eod does not insert duplicate rows."""
    from xyz.polygon_service.historical import fetch_and_persist_historical_eod
    from xyz.polygon_service.options_client import OptionsClient, OptionContract
    from xyz.polygon_service.models import OptionHistoricalEod

    body = _load("historical_aapl_2024.json")
    contract_ticker = "O:AAPL250117C00185000"
    # Allow two HTTP calls (first run + second run)
    respx.get(
        f"https://api.polygon.io/v2/aggs/ticker/{contract_ticker}/range/1/day/2024-01-08/2024-01-12"
    ).mock(return_value=httpx.Response(200, json=body))

    c = OptionsClient(
        api_key="TEST_KEY",
        rate_limiter=rate_limiter_unlimited,
        cache=chain_cache_fresh,
    )
    contracts = [
        OptionContract(
            contract_ticker=contract_ticker,
            underlying="AAPL",
            expiry=date(2025, 1, 17),
            strike=185.0,
            option_type="CALL",
        )
    ]

    kwargs = dict(
        client=c, db=polygon_db_session, underlying="AAPL",
        contracts=contracts, start_date=date(2024, 1, 8), end_date=date(2024, 1, 12)
    )

    # First run inserts 5 rows.
    inserted_first = fetch_and_persist_historical_eod(**kwargs)
    polygon_db_session.commit()

    # Second run should insert 0 (all duplicates).
    inserted_second = fetch_and_persist_historical_eod(**kwargs)
    polygon_db_session.commit()

    assert inserted_first == 5
    assert inserted_second == 0

    total = polygon_db_session.query(OptionHistoricalEod).filter_by(contract_ticker=contract_ticker).count()
    assert total == 5


# ---------------------------------------------------------------------------
# Test 8 — polygon models are registered on the finazon Base
# ---------------------------------------------------------------------------

def test_option_models_registered_on_finazon_base():
    """Importing xyz.polygon_service.models registers tables on the finazon Base.

    Uses the thin base module (no psycopg2 dependency) to confirm table
    registration; sql_service.py re-exports the same Base object.
    """
    from xyz.finazon_service.base import Base
    import xyz.polygon_service.models  # noqa: ensure registration

    table_names = set(Base.metadata.tables.keys())
    expected = {
        "option_historical_eod",
        "option_chains",
        "option_iv_surface",
        "option_quotes",
    }
    missing = expected - table_names
    assert not missing, f"Tables not registered on finazon Base: {missing}"
