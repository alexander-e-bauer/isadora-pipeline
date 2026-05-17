"""Engine tenant position/trade tests — Task 2.3.

Test inventory
--------------
1. test_position_model_registers_on_tenant_base
   Assert 'positions' and 'trades' in xyz.tenant.models.Base.metadata.tables.

2. test_engine_position_schema_matches_server
   Load server's Position model via _load_server_models() and compare
   column metadata column-by-column.

3. test_insert_positions_bulk_creates_rows_and_emits_events
   Using SQLite in-memory tenant DB, call insert_positions_bulk with 3
   rows, assert 3 positions + 3 position.added events.

All tests run against in-memory SQLite (no live Postgres needed).
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
# File-based import helpers — reuse the same pattern as test_tenant_read.py
# ---------------------------------------------------------------------------

_SERVER_ROOT = Path(__file__).resolve().parents[2] / "server-fastapi-wt"


def _load_module(path: Path, module_name: str) -> ModuleType:
    """Load a Python file as a module with the given name."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"Cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _load_server_models() -> None:
    """Load server's SQLAlchemy models via direct file import.

    Mirrors the logic in test_tenant_read._load_server_models but also
    loads position.py and trade.py.
    """
    from cryptography.fernet import Fernet
    import xyz.tenant.encrypted_types as engine_et

    app_root = _SERVER_ROOT / "app"

    if "app" not in sys.modules:
        sys.modules["app"] = ModuleType("app")

    if "app.config" not in sys.modules:
        config_stub = ModuleType("app.config")

        class _FakeSettings:
            encryption_key = "STUB"

        def get_settings():
            return _FakeSettings()

        config_stub.get_settings = get_settings  # type: ignore[attr-defined]
        sys.modules["app.config"] = config_stub

    base_mod = _load_module(app_root / "models" / "base.py", "_srv_models_base")

    et_mod = _load_module(
        app_root / "models" / "encrypted_types.py", "_srv_models_encrypted_types"
    )
    fernet_key = engine_et._fernet_cache
    def _patched_get_fernet() -> Fernet:
        return fernet_key  # type: ignore[return-value]
    et_mod._get_fernet = _patched_get_fernet
    et_mod._fernet_cache = fernet_key

    sys.modules["app.models"] = ModuleType("app.models")
    sys.modules["app.models.base"] = base_mod
    sys.modules["app.models.encrypted_types"] = et_mod

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


# ---------------------------------------------------------------------------
# Engine-side imports
# ---------------------------------------------------------------------------
from xyz.tenant.models import Base, Event, Firm, Position, Trade


# ---------------------------------------------------------------------------
# 1. test_position_model_registers_on_tenant_base
# ---------------------------------------------------------------------------

def test_position_model_registers_on_tenant_base():
    """Assert 'positions' and 'trades' in Base.metadata.tables."""
    tables = Base.metadata.tables
    assert "positions" in tables, (
        f"'positions' not registered in tenant Base.metadata. "
        f"Tables found: {sorted(tables.keys())}"
    )
    assert "trades" in tables, (
        f"'trades' not registered in tenant Base.metadata. "
        f"Tables found: {sorted(tables.keys())}"
    )


# ---------------------------------------------------------------------------
# 2. test_engine_position_schema_matches_server
# ---------------------------------------------------------------------------

def test_engine_position_schema_matches_server():
    """Column-level parity check for positions and trades tables."""
    _load_server_models()

    server_base_mod = sys.modules["_srv_models_base"]
    ServerBase = server_base_mod.Base  # type: ignore[attr-defined]

    engine_tables = Base.metadata.tables
    server_tables = ServerBase.metadata.tables

    mismatches = []
    for table_name in ("positions", "trades"):
        server_table = server_tables.get(table_name)
        engine_table = engine_tables.get(table_name)

        assert server_table is not None, f"Server does not have table '{table_name}'"
        assert engine_table is not None, f"Engine mirror does not have table '{table_name}'"

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

        for col_name in engine_cols:
            if col_name not in server_cols:
                mismatches.append(
                    f"{table_name}.{col_name}: present in engine mirror, "
                    f"missing from server"
                )

    assert not mismatches, (
        "Schema parity violations for positions/trades tables:\n"
        + "\n".join(f"  {m}" for m in mismatches)
    )


# ---------------------------------------------------------------------------
# 3. test_insert_positions_bulk_creates_rows_and_emits_events
# ---------------------------------------------------------------------------

def test_insert_positions_bulk_creates_rows_and_emits_events(db_session):
    """Call insert_positions_bulk with 3 rows; assert 3 positions + 3 firm-scoped
    position.added events (firm_id resolved via the Account → Client chain)."""
    from xyz.positions.manual_entry import insert_positions_bulk
    from xyz.tenant.models import Account, Client, Firm

    # Build the full Firm → Client → Account chain so _resolve_firm_id can
    # find the firm and emit firm-scoped events.
    firm = Firm(name="Test Firm")
    db_session.add(firm)
    db_session.flush()

    client = Client(firm_id=firm.id, name="Test Client")
    db_session.add(client)
    db_session.flush()

    account = Account(
        client_id=client.id,
        nickname="Test Account",
        account_type="TAXABLE",
        base_currency="USD",
        is_active=True,
    )
    db_session.add(account)
    db_session.flush()

    rows = [
        {"symbol": "AAPL", "asset_class": "EQUITY", "qty": "100.000000", "cost_basis": "15000.000000"},
        {"symbol": "MSFT", "asset_class": "EQUITY", "qty": "50.000000", "cost_basis": "20000.000000"},
        {"symbol": "GOOG", "asset_class": "EQUITY", "qty": "10.000000", "cost_basis": "25000.000000"},
    ]

    ids = insert_positions_bulk(db_session, account.id, rows)
    db_session.commit()

    # 3 position ids returned
    assert len(ids) == 3
    assert all(isinstance(i, int) for i in ids)

    # 3 positions in the DB
    positions = db_session.scalars(
        select(Position).where(Position.account_id == account.id)
    ).all()
    assert len(positions) == 3

    symbols = {p.symbol for p in positions}
    assert symbols == {"AAPL", "MSFT", "GOOG"}

    # 3 position.added events emitted, ALL firm-scoped so /events queries
    # filtered by firm_id return them.
    events = db_session.scalars(
        select(Event).where(Event.kind == "position.added")
    ).all()
    assert len(events) == 3
    assert all(ev.firm_id == firm.id for ev in events), (
        "All emitted events must carry firm_id (resolved from the account chain), "
        "not None — otherwise firm-scoped /events queries miss them."
    )

    event_symbols = {e.payload["symbol"] for e in events}
    assert event_symbols == {"AAPL", "MSFT", "GOOG"}
