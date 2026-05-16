"""Engine pytest fixtures — tenant tests.

Sets up:
  - sys.path patching so that server's app/ package is importable from
    the engine worktree (needed by the schema-parity test).  The server
    path is inserted BEFORE the engine root so that server's ``app/``
    package (a directory) wins over engine's ``app.py`` (a single file)
    when Python resolves the ``app`` name.
  - SQLite in-memory engine with the tenant Base (create_all) so that
    tests run without a live Postgres.
  - db_session fixture for ORM tests.
  - _fernet_key autouse fixture that injects a known Fernet key into
    xyz.tenant.encrypted_types so encrypted columns work in SQLite.
    The same key is also injected into server's encrypted_types module
    (if importable) so that cross-app schema-parity tests can read
    encrypted columns without a real ENCRYPTION_KEY env var.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Prepend server worktree to sys.path so tests that import server's app
# package (schema-parity, hash-equivalence tests) can do so without
# triggering the engine's top-level app.py.
# This must happen before any other local-package import.
# ---------------------------------------------------------------------------
_ENGINE_ROOT = Path(__file__).resolve().parents[1]
_SERVER_WT = _ENGINE_ROOT.parent / "server-fastapi-wt"
if str(_SERVER_WT) not in sys.path:
    sys.path.insert(0, str(_SERVER_WT))

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import the tenant Base so create_all registers all tenant tables.
from xyz.tenant.models import Base
import xyz.tenant.encrypted_types as encrypted_types


@pytest.fixture
def db_engine():
    """SQLite in-memory engine with tenant tables created."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """ORM Session against the in-memory SQLite engine."""
    Session = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(autouse=True)
def _fernet_key(monkeypatch):
    """Force a known Fernet key for encrypted_types in all tests.

    Patches both engine's xyz.tenant.encrypted_types and (when importable)
    server's app.models.encrypted_types so that tests which import server
    models can also decrypt/encrypt without a real ENCRYPTION_KEY env var.
    """
    key = Fernet.generate_key()
    fernet_instance = Fernet(key)
    monkeypatch.setattr(encrypted_types, "_fernet_cache", fernet_instance)

    # Also patch server's encrypted_types if it has been imported.
    try:
        import app.models.encrypted_types as server_et
        monkeypatch.setattr(server_et, "_fernet_cache", fernet_instance)
    except Exception:
        pass  # server package not yet imported — nothing to patch

    yield

    monkeypatch.setattr(encrypted_types, "_fernet_cache", None)
    try:
        import app.models.encrypted_types as server_et
        monkeypatch.setattr(server_et, "_fernet_cache", None)
    except Exception:
        pass
