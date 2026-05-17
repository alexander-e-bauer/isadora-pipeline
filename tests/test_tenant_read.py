"""Engine tenant tests — Task 2.1.

Test inventory
--------------
1. test_tenant_metadata_matches_server    — schema-parity guard between engine
                                            mirror and server's ORM.
2. test_engine_can_emit_event             — emit_event inserts a row with
                                            correct chain semantics.
3. test_engine_event_hash_matches_server  — engine and server hash functions
                                            produce identical output for the
                                            same inputs.
4. test_engine_emit_links_to_existing_chain — engine chains onto a
                                            pre-existing event (simulating a
                                            server-emitted row).

All tests run against in-memory SQLite (no live Postgres needed).

Cross-app import strategy
-------------------------
Tests that need server's code (schema parity, hash equivalence) use
importlib.util.spec_from_file_location to load server modules DIRECTLY
from their filesystem paths, bypassing normal package resolution.  This
avoids the collision between engine's top-level app.py and server's
app/ package that would otherwise occur when both paths are on sys.path,
and also avoids pulling in Pydantic Settings (app.config) which would
fail without unrelated env vars.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# File-based import helpers — load server modules without package resolution
# ---------------------------------------------------------------------------

_SERVER_ROOT = Path(__file__).resolve().parents[2] / "server-fastapi-wt"


def _load_module(path: Path, module_name: str) -> ModuleType:
    """Load a Python file as a module with the given name, bypassing package
    import machinery.  Dependencies that are already in sys.modules are
    reused; new dependencies are looked up normally.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"Cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _load_server_models() -> None:
    """Load server's SQLAlchemy models via direct file import.

    Loads only what we need for schema parity: declarative Base + each
    model file.  Does NOT load app.config (Pydantic Settings) and does
    NOT load the engine's app.py.

    Strategy:
    - Pre-register dummy ``app``, ``app.config``, ``app.models`` entries
      in sys.modules BEFORE any file is executed so that ``from app.config
      import get_settings`` inside encrypted_types.py finds our stub instead
      of resolving to the engine's app.py.
    - After registering stubs, patch _get_fernet in the encrypted_types
      module to use the test's Fernet key (the conftest _fernet_key fixture
      already injected one into xyz.tenant.encrypted_types._fernet_cache).
    """
    from types import ModuleType
    from cryptography.fernet import Fernet
    import xyz.tenant.encrypted_types as engine_et

    app_root = _SERVER_ROOT / "app"

    # ------------------------------------------------------------------
    # 1. Pre-register stub modules for any app.* that encrypted_types
    #    tries to import via ``from app.config import get_settings``.
    # ------------------------------------------------------------------
    if "app" not in sys.modules:
        sys.modules["app"] = ModuleType("app")

    # Stub app.config with a fake get_settings so encrypted_types doesn't
    # try to construct a real Settings object (needs pydantic_settings + env vars).
    if "app.config" not in sys.modules:
        config_stub = ModuleType("app.config")

        class _FakeSettings:
            encryption_key = "STUB"  # never actually used — _get_fernet patched below

        def get_settings():
            return _FakeSettings()

        config_stub.get_settings = get_settings  # type: ignore[attr-defined]
        sys.modules["app.config"] = config_stub

    # ------------------------------------------------------------------
    # 2. Load base.py (declarative Base).
    # ------------------------------------------------------------------
    base_mod = _load_module(app_root / "models" / "base.py", "_srv_models_base")

    # ------------------------------------------------------------------
    # 3. Load encrypted_types.py now that app.config is stubbed.
    # ------------------------------------------------------------------
    et_mod = _load_module(
        app_root / "models" / "encrypted_types.py", "_srv_models_encrypted_types"
    )
    # Replace _get_fernet so it uses the test's Fernet key.
    fernet_key = engine_et._fernet_cache
    def _patched_get_fernet() -> Fernet:
        return fernet_key  # type: ignore[return-value]
    et_mod._get_fernet = _patched_get_fernet
    et_mod._fernet_cache = fernet_key

    # ------------------------------------------------------------------
    # 4. Register stubs so model files can import each other cleanly.
    # ------------------------------------------------------------------
    sys.modules["app.models"] = ModuleType("app.models")
    sys.modules["app.models.base"] = base_mod
    sys.modules["app.models.encrypted_types"] = et_mod

    # ------------------------------------------------------------------
    # 5. Load each model file.
    # ------------------------------------------------------------------
    model_files = [
        ("firm", "app.models.firm"),
        ("user", "app.models.user"),
        ("user_access", "app.models.user_access"),
        ("event", "app.models.event"),
        ("client", "app.models.client"),
        ("account", "app.models.account"),
        ("autonomy", "app.models.autonomy"),
        ("broker_connection", "app.models.broker_connection"),
        ("subscription", "app.models.subscription"),
        ("payment_method", "app.models.payment_method"),
        ("processed_stripe_event", "app.models.processed_stripe_event"),
        ("strategy", "app.models.strategy"),
        ("deployment", "app.models.deployment"),
        ("position", "app.models.position"),
        ("trade", "app.models.trade"),
    ]
    for file_name, module_name in model_files:
        if module_name not in sys.modules:
            mod = _load_module(app_root / "models" / f"{file_name}.py", module_name)
            sys.modules[module_name] = mod


def _server_compute_hash_impl(
    *,
    kind: str,
    firm_id,
    actor_user_id,
    payload: dict,
    prev_event_hash,
    created_at_iso: str,
) -> str:
    """Inline copy of server's _compute_event_hash.

    This is intentionally a verbatim copy of the logic in
    server/app/events/emit.py — no import of server's package needed.
    Used by test_engine_event_hash_matches_server_implementation to
    verify that the engine and server produce identical hashes for the
    same inputs.
    """
    import hashlib
    import json

    canonical = json.dumps(
        {
            "kind": kind,
            "firm_id": firm_id,
            "actor_user_id": actor_user_id,
            "payload": payload,
            "prev_event_hash": prev_event_hash,
            "created_at": created_at_iso,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Engine-side imports
# ---------------------------------------------------------------------------
from xyz.tenant.models import Base, Event, Firm
from xyz.tenant.events import emit_event, _compute_event_hash as engine_compute_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_firm(db: Session, name: str = "Test Firm") -> Firm:
    firm = Firm(name=name)
    db.add(firm)
    db.flush()
    return firm


# ---------------------------------------------------------------------------
# 1. test_tenant_metadata_matches_server
# ---------------------------------------------------------------------------

def test_tenant_metadata_matches_server():
    """Column-level schema-parity check between engine mirror and server ORM.

    Loads server's model files directly via importlib (no package import)
    to avoid engine/server app-name collision and Pydantic Settings activation.
    Compares column names, nullable flags, and primary_key flags between the
    engine mirror and server's authoritative ORM definitions.
    """
    _load_server_models()

    # After _load_server_models() all server model files are in sys.modules
    # under "app.models.*" keys. The server's DeclarativeBase was loaded as
    # "_srv_models_base" — retrieve it.
    server_base_mod = sys.modules["_srv_models_base"]
    ServerBase = server_base_mod.Base  # type: ignore[attr-defined]

    engine_tables = Base.metadata.tables
    server_tables = ServerBase.metadata.tables

    # Every server table should have a corresponding engine table.
    missing_in_engine = set(server_tables) - set(engine_tables)
    assert not missing_in_engine, (
        f"Tables present in server but missing from engine mirror: {missing_in_engine}"
    )

    mismatches = []
    for table_name, server_table in server_tables.items():
        engine_table = engine_tables.get(table_name)
        if engine_table is None:
            continue  # already caught above

        server_cols = {c.name: c for c in server_table.columns}
        engine_cols = {c.name: c for c in engine_table.columns}

        for col_name, s_col in server_cols.items():
            if col_name not in engine_cols:
                mismatches.append(
                    f"{table_name}.{col_name}: present in server, "
                    f"missing in engine mirror"
                )
                continue
            e_col = engine_cols[col_name]
            if s_col.nullable != e_col.nullable:
                mismatches.append(
                    f"{table_name}.{col_name}: nullable mismatch "
                    f"server={s_col.nullable} engine={e_col.nullable}"
                )
            if s_col.primary_key != e_col.primary_key:
                mismatches.append(
                    f"{table_name}.{col_name}: primary_key mismatch "
                    f"server={s_col.primary_key} engine={e_col.primary_key}"
                )

    assert not mismatches, (
        "Schema parity violations between server and engine mirror:\n"
        + "\n".join(f"  {m}" for m in mismatches)
    )


# ---------------------------------------------------------------------------
# 2. test_engine_can_emit_event
# ---------------------------------------------------------------------------

def test_engine_can_emit_event(db_session):
    """emit_event inserts a row with the right kind, firm_id, and hash."""
    firm = _make_firm(db_session)

    event_id = emit_event(
        db=db_session,
        kind="pipeline.run_started",
        firm_id=firm.id,
        actor_user_id=None,
        payload={"ticker": "AAPL"},
    )
    db_session.commit()

    assert isinstance(event_id, str)
    assert len(event_id) == 32

    row = db_session.get(Event, event_id)
    assert row is not None
    assert row.kind == "pipeline.run_started"
    assert row.firm_id == firm.id
    assert row.payload == {"ticker": "AAPL"}
    assert row.event_hash is not None
    assert len(row.event_hash) == 64
    # First event in chain: no predecessor.
    assert row.prev_event_hash is None


# ---------------------------------------------------------------------------
# 3. test_engine_event_hash_matches_server_implementation
# ---------------------------------------------------------------------------

def _load_live_server_compute_hash():
    """Load the actual `_compute_event_hash` function from server/app/events/emit.py.

    This is the cross-app verification anchor: we compare engine's
    implementation against the *live* server function, not a copy of it.
    Any change to server's hash function that engine doesn't mirror will
    fail this test the moment it runs.

    Path: ../server-fastapi-wt/app/events/emit.py — the sibling worktree on
    the server side. If that path doesn't exist (e.g. the worktree was
    removed), fall back to the inline copy with a clear warning so the test
    is still meaningful in CI without the full multi-worktree layout.
    """
    server_emit_path = (
        Path(__file__).resolve().parents[2] / "server-fastapi-wt" / "app" / "events" / "emit.py"
    )
    if not server_emit_path.exists():
        import warnings
        warnings.warn(
            f"Server emit.py not found at {server_emit_path}; "
            "falling back to inline copy. Cross-app verification is weakened.",
            stacklevel=2,
        )
        return _server_compute_hash_impl

    mod = _load_module(server_emit_path, "_live_server_emit")
    return mod._compute_event_hash


def test_engine_event_hash_matches_server_implementation():
    """Engine and the LIVE server _compute_event_hash produce identical digests.

    Loads server/app/events/emit.py directly via importlib and compares
    its `_compute_event_hash` against engine's. If server's hash function
    diverges in any way (new field added, separator changed, key reordered),
    this test fails immediately — that's the chain-portability guarantee.
    """
    kwargs = dict(
        kind="user.registered",
        firm_id=42,
        actor_user_id=7,
        payload={"email_hash": "abcdef1234567890"},
        prev_event_hash="deadbeef" * 8,
        created_at_iso="2026-01-01T00:00:00+00:00",
    )

    live_server_hash = _load_live_server_compute_hash()
    engine_hash = engine_compute_hash(**kwargs)
    server_hash = live_server_hash(**kwargs)

    assert engine_hash == server_hash, (
        f"Hash mismatch between engine and LIVE server implementations:\n"
        f"  engine: {engine_hash}\n"
        f"  server: {server_hash}\n"
        "Engine's xyz/tenant/events.py is out of sync with "
        "server-fastapi-wt/app/events/emit.py — update engine's mirror."
    )
    assert len(engine_hash) == 64, "SHA-256 hex digest must be 64 chars"

    # Belt-and-suspenders: also assert the inline copy stays in sync. If this
    # ever fails, the inline copy is stale even though the live test still
    # works — fix the inline copy so the fallback path remains accurate.
    inline_hash = _server_compute_hash_impl(**kwargs)
    assert inline_hash == server_hash, (
        "Inline copy of _compute_event_hash has drifted from the live "
        "server function. Update _server_compute_hash_impl in this test "
        "to keep the fallback path accurate."
    )


# ---------------------------------------------------------------------------
# 4. test_engine_emit_links_to_existing_chain
# ---------------------------------------------------------------------------

def test_engine_emit_links_to_existing_chain(db_session):
    """Engine's emit_event correctly chains onto a pre-existing event.

    We pre-insert an event row directly (simulating a server-emitted event),
    then call engine's emit_event.  The new event's prev_event_hash must
    equal the pre-inserted event's event_hash — proving engine reads and
    chains forward from whatever is already in the table, regardless of
    which process wrote the prior row.
    """
    firm = _make_firm(db_session)

    # Simulate a server-emitted event by inserting directly into the DB.
    pre_existing_hash = "aabbcc" + "00" * 29  # 64-char hex string
    pre_event = Event(
        id="pre000000000000000000000000000000",
        firm_id=firm.id,
        actor_user_id=None,
        kind="user.registered",
        payload={"source": "server"},
        prev_event_hash=None,
        event_hash=pre_existing_hash,
    )
    db_session.add(pre_event)
    db_session.commit()

    # Now engine emits its first event for this firm.
    engine_event_id = emit_event(
        db=db_session,
        kind="pipeline.run_started",
        firm_id=firm.id,
        actor_user_id=None,
        payload={"run": 1},
    )
    db_session.commit()

    engine_row = db_session.get(Event, engine_event_id)
    assert engine_row is not None
    assert engine_row.prev_event_hash == pre_existing_hash, (
        f"Expected prev_event_hash={pre_existing_hash!r}, "
        f"got {engine_row.prev_event_hash!r}"
    )
