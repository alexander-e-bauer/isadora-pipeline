"""Schema parity checker — engine tenant mirror vs server ORM.

Exits 0 when engine's xyz.tenant.models.Base and server's
app.models.Base define identical tables (column names, nullable flags,
primary_key flags).  Exits 1 with a list of mismatches otherwise.

Run from the engine worktree root:

    python scripts/check_schema_parity.py

The script uses importlib to load server's model files directly from disk,
bypassing normal package resolution.  This avoids:
  - The collision between engine's top-level app.py and server's app/ package.
  - Pulling in Pydantic Settings (app.config), which needs extra env vars.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

ENGINE_ROOT = Path(__file__).resolve().parents[1]  # engine-tenant-wt/
SERVER_WT = ENGINE_ROOT.parent / "server-fastapi-wt"
SERVER_APP = SERVER_WT / "app"

if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))


# ---------------------------------------------------------------------------
# Engine tenant Base
# ---------------------------------------------------------------------------

def _get_engine_base():
    # Provide a dummy ENCRYPTION_KEY so encrypted_types initialises without error.
    from cryptography.fernet import Fernet
    import xyz.tenant.encrypted_types as et
    if et._fernet_cache is None:
        import os
        key = os.environ.get("ENCRYPTION_KEY")
        if key:
            et._fernet_cache = Fernet(key.encode())
        else:
            # Use a dummy key — we only need column metadata, not actual decryption.
            et._fernet_cache = Fernet(Fernet.generate_key())

    from xyz.tenant.models import Base as EngineBase
    return EngineBase


# ---------------------------------------------------------------------------
# Server Base — via direct file import (avoids app.py collision + pydantic)
# ---------------------------------------------------------------------------

def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"Cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _get_server_base():
    from cryptography.fernet import Fernet

    # Stub app and app.config so encrypted_types.py doesn't try to load
    # Pydantic Settings or the engine's app.py.
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

    base_mod = _load_module(SERVER_APP / "models" / "base.py", "_srv_models_base")

    # Patch encrypted_types to use a dummy Fernet (metadata only — no real decryption).
    et_mod = _load_module(
        SERVER_APP / "models" / "encrypted_types.py", "_srv_models_encrypted_types"
    )
    _dummy_fernet = Fernet(Fernet.generate_key())

    def _patched_get_fernet():
        return _dummy_fernet

    et_mod._get_fernet = _patched_get_fernet
    et_mod._fernet_cache = _dummy_fernet

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
    ]
    for file_name, module_name in model_files:
        if module_name not in sys.modules:
            mod = _load_module(SERVER_APP / "models" / f"{file_name}.py", module_name)
            sys.modules[module_name] = mod

    return base_mod.Base  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def _check_parity(EngineBase, ServerBase) -> list[str]:
    """Return a list of mismatch descriptions; empty list means parity OK."""
    engine_tables = EngineBase.metadata.tables
    server_tables = ServerBase.metadata.tables

    mismatches: list[str] = []

    for table_name in server_tables:
        if table_name not in engine_tables:
            mismatches.append(
                f"MISSING TABLE: '{table_name}' is in server but not in engine mirror"
            )

    for table_name in engine_tables:
        if table_name not in server_tables:
            mismatches.append(
                f"EXTRA TABLE: '{table_name}' is in engine mirror but not in server "
                "(engine should only mirror server-owned tables)"
            )

    for table_name in server_tables:
        if table_name not in engine_tables:
            continue

        server_table = server_tables[table_name]
        engine_table = engine_tables[table_name]

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
                    f"{table_name}.{col_name}: nullable mismatch — "
                    f"server={s_col.nullable}, engine={e_col.nullable}"
                )
            if s_col.primary_key != e_col.primary_key:
                mismatches.append(
                    f"{table_name}.{col_name}: primary_key mismatch — "
                    f"server={s_col.primary_key}, engine={e_col.primary_key}"
                )

        for col_name in engine_cols:
            if col_name not in server_cols:
                mismatches.append(
                    f"{table_name}.{col_name}: present in engine mirror, "
                    f"missing from server (engine mirror has a stale column)"
                )

    return mismatches


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    EngineBase = _get_engine_base()
    ServerBase = _get_server_base()

    mismatches = _check_parity(EngineBase, ServerBase)
    if mismatches:
        print("SCHEMA PARITY FAILURES:")
        for m in mismatches:
            print(f"  {m}")
        return 1

    engine_table_count = len(EngineBase.metadata.tables)
    server_table_count = len(ServerBase.metadata.tables)
    print(
        f"Schema parity OK — {engine_table_count} engine tables match "
        f"{server_table_count} server tables."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
