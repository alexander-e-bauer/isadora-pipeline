"""Dump engine's tenant SQLAlchemy schema mirror to deterministic JSON.

This script walks ``xyz.tenant.models.Base.metadata.tables`` (the
read-only mirror engine maintains for the tenant schema), filters to the
canonical tenant table set shared with server, and emits JSON suitable
for byte-for-byte comparison against server's dump.

The polygon / finazon tables (engine's writeable producer schema) live
in a DIFFERENT declarative Base (``xyz.finazon_service.sql_service``) and
are intentionally NOT reached from this module.  We only walk the tenant
mirror.

Usage
-----

    python scripts/dump_tenant_schema.py                    # stdout
    python scripts/dump_tenant_schema.py --output PATH      # to file
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_WORKTREE_ROOT = _THIS_DIR.parent
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

# Encrypted columns require a Fernet key at import time.  Metadata-only
# walks don't need a real key — generate one if not present.
if not os.environ.get("ENCRYPTION_KEY"):
    from cryptography.fernet import Fernet
    os.environ["ENCRYPTION_KEY"] = Fernet.generate_key().decode()

# Must match server/scripts/dump_tenant_schema.py CANONICAL_TENANT_TABLES
# byte-for-byte.  Keep these in sync — they are the contract between the
# two dump scripts.
CANONICAL_TENANT_TABLES: frozenset[str] = frozenset({
    "firms",
    "users",
    "user_client_access",
    "clients",
    "accounts",
    "account_autonomy",
    "broker_connections",
    "subscriptions",
    "payment_methods",
    "processed_stripe_events",
    "strategies",
    "deployments",
    "positions",
    "trades",
    "events",
    "backtest_results",
})


def _type_repr(col) -> str:
    """Render a SQLAlchemy column type to a stable string.

    Must mirror server's _type_repr.  See server/scripts/dump_tenant_schema.py
    for the rationale.
    """
    type_obj = col.type
    base = str(type_obj)
    decorator = type(type_obj).__name__
    if decorator.upper() != base.upper() and decorator not in {"NullType"}:
        return f"{decorator}({base})"
    return base


def _default_repr(col) -> str | None:
    if col.server_default is not None:
        arg = getattr(col.server_default, "arg", col.server_default)
        return str(arg)
    if col.default is not None:
        default = col.default
        if hasattr(default, "arg"):
            arg = default.arg
            if callable(arg):
                return f"<callable:{getattr(arg, '__name__', 'anonymous')}>"
            return repr(arg)
        return repr(default)
    return None


def _foreign_keys(col) -> list[str]:
    """Return sorted ``<table>.<col>`` FK target strings.

    Uses ``fk.target_fullname`` (the string form supplied to the
    ``ForeignKey(...)`` constructor) rather than ``fk.column`` because
    the latter requires the referenced table to be present in the same
    metadata — which is NOT the case on engine for FKs pointing at
    server-only tables such as ``backtest_results``.
    """
    return sorted(fk.target_fullname for fk in col.foreign_keys)


def _index_entries(table) -> list[dict[str, Any]]:
    entries = []
    for ix in table.indexes:
        entries.append({
            "name": ix.name,
            "columns": sorted(c.name for c in ix.columns),
            "unique": bool(ix.unique),
        })
    entries.sort(key=lambda e: (e["name"] or "", tuple(e["columns"])))
    return entries


def _unique_entries(table) -> list[dict[str, Any]]:
    from sqlalchemy import UniqueConstraint
    entries = []
    for constraint in table.constraints:
        if isinstance(constraint, UniqueConstraint):
            entries.append({
                "name": constraint.name,
                "columns": sorted(c.name for c in constraint.columns),
            })
    entries.sort(key=lambda e: (e["name"] or "", tuple(e["columns"])))
    return entries


def _dump_table(table) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for col in table.columns:
        columns[col.name] = {
            "type": _type_repr(col),
            "nullable": bool(col.nullable),
            "primary_key": bool(col.primary_key),
            "default": _default_repr(col),
            "foreign_keys": _foreign_keys(col),
        }
    return {
        "columns": columns,
        "indexes": _index_entries(table),
        "uniques": _unique_entries(table),
    }


def build_dump() -> dict[str, Any]:
    """Build the full schema dump dict (pre-serialisation)."""
    from xyz.tenant.models import Base  # lazy import after path/env setup

    tables_out: dict[str, Any] = {}
    metadata_tables = Base.metadata.tables
    for name in sorted(metadata_tables):
        if name not in CANONICAL_TENANT_TABLES:
            continue
        tables_out[name] = _dump_table(metadata_tables[name])

    for missing in sorted(CANONICAL_TENANT_TABLES - set(tables_out)):
        tables_out[missing] = {
            "__missing__": True,
            "columns": {},
            "indexes": [],
            "uniques": [],
        }

    return {
        "version": "1",
        "generator": "engine",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "tables": tables_out,
    }


def serialise(doc: dict[str, Any]) -> str:
    """Stable JSON encoding suitable for byte-for-byte comparison."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dump engine's tenant SQLAlchemy schema mirror to JSON.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write JSON to this path instead of stdout.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Pretty-print with this indent (for humans; not diff-safe).",
    )
    args = parser.parse_args(argv)

    doc = build_dump()

    if args.indent is not None:
        payload = json.dumps(doc, sort_keys=True, indent=args.indent, default=str)
    else:
        payload = serialise(doc)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
