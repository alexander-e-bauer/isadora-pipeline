"""Engine-side session helper for tenant tables.

The tenant tables (firms, users, clients, accounts, events, etc.) live in
the SAME Postgres database as engine's finazon tables (tickers,
historical_data, computed_metrics, etc.).  We REUSE engine's existing
SQLAlchemy engine and connection pool — no second pool is created.

Usage
-----
    from xyz.tenant.db import get_tenant_session

    with get_tenant_session() as session:
        firm = session.get(Firm, firm_id)

Or with manual lifecycle (e.g. in a background task):

    session = get_tenant_session()
    try:
        emit_event(db=session, ...)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy.orm import Session

# Reuse engine's existing SQLAlchemy engine + session factory.
# xyz.finazon_service.sql_service already creates the connection pool at
# import time (DATABASE_URI, pool_size=15, etc.) — we bind to that engine
# rather than creating a second one.
from xyz.finazon_service.sql_service import SessionLocal


def get_tenant_session() -> Session:
    """Return a new Session bound to engine's existing DB engine.

    The tenant tables share the same Postgres as engine's finazon tables.
    One engine → one pool → sessions for both schema namespaces.

    Callers own the session lifecycle (begin / commit / rollback / close).
    For a managed context use ``tenant_session()`` instead.
    """
    return SessionLocal()


@contextmanager
def tenant_session() -> Generator[Session, None, None]:
    """Context manager that yields a Session and commits on exit (rolls back on error).

    Example::

        from xyz.tenant.db import tenant_session
        from xyz.tenant.events import emit_event

        with tenant_session() as db:
            emit_event(db=db, kind="pipeline.run_started", firm_id=firm_id, ...)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
