"""Tests for engine's tenant-schema dump script (Task 5.3).

Symmetric counterpart to ``server-fastapi-wt/tests/test_dump_tenant_schema.py``.
Covers:
  - CLI runs without a DB connection and emits valid JSON.
  - Every canonical tenant table is present.
  - The dump excludes engine-only finazon / polygon tables.
  - ``generated_at`` is an ISO timestamp; both ``generated_at`` and
    ``generator`` are tracked but excluded from cross-app diff.
  - The dump is deterministic and round-trips cleanly.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).resolve().parent
_WORKTREE_ROOT = _THIS_DIR.parent
_SCRIPT = _WORKTREE_ROOT / "scripts" / "dump_tenant_schema.py"

# Engine-side canonical set must equal server-side canonical set.  This
# duplicates the constant inside dump_tenant_schema.py on purpose — the
# duplication is the contract.
_CANONICAL_TABLES = frozenset({
    "firms", "users", "user_client_access", "clients", "accounts",
    "account_autonomy", "broker_connections", "subscriptions",
    "payment_methods", "processed_stripe_events", "strategies",
    "deployments", "positions", "trades", "events", "backtest_results",
})

# Finazon / polygon tables that engine OWNS but should NEVER appear in the
# tenant dump.  This list is illustrative — the assertion is "intersection
# with the dump must be empty" so adding a new finazon table doesn't
# require touching this test.
_KNOWN_ENGINE_ONLY = frozenset({
    "historical_data",
    "computed_metrics",
    "market_emb_thirty_min",
    "market_emb_day",
    "market_emb_week",
})


def _run_dump(tmp_path: Path) -> dict:
    out_path = tmp_path / "engine-schema.json"
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--output", str(out_path)],
        capture_output=True,
        text=True,
        cwd=_WORKTREE_ROOT,
    )
    assert result.returncode == 0, (
        f"dump_tenant_schema.py failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert out_path.exists()
    return json.loads(out_path.read_text(encoding="utf-8"))


def test_dump_runs_without_db(tmp_path):
    doc = _run_dump(tmp_path)
    assert doc.get("version") == "1"
    assert doc.get("generator") == "engine"


def test_dump_has_generated_at_iso(tmp_path):
    doc = _run_dump(tmp_path)
    assert "generated_at" in doc
    datetime.fromisoformat(doc["generated_at"])


def test_dump_contains_all_canonical_tables(tmp_path):
    doc = _run_dump(tmp_path)
    tables = set(doc["tables"].keys())
    missing = _CANONICAL_TABLES - tables
    assert not missing, f"Canonical tables missing from dump: {sorted(missing)}"
    for name, body in doc["tables"].items():
        assert not body.get("__missing__"), (
            f"Table '{name}' rendered as __missing__ in the engine dump — "
            "this means engine's xyz.tenant.models is missing a canonical table."
        )


def test_dump_excludes_engine_only_tables(tmp_path):
    """The dump must EXCLUDE polygon / finazon producer-side tables.

    They live in a different declarative Base on the engine side, so the
    dump's metadata walk should never see them — but assert it explicitly
    in case a future refactor merges Bases.
    """
    doc = _run_dump(tmp_path)
    table_names = set(doc["tables"].keys())
    leaked = table_names & _KNOWN_ENGINE_ONLY
    assert not leaked, f"Engine-only tables leaked into tenant dump: {sorted(leaked)}"
    extra = table_names - _CANONICAL_TABLES
    assert not extra, f"Non-canonical tables in dump: {sorted(extra)}"


def test_dump_is_deterministic(tmp_path):
    doc1 = _run_dump(tmp_path / "first")
    doc2 = _run_dump(tmp_path / "second")
    for d in (doc1, doc2):
        d.pop("generated_at", None)
        d.pop("generator", None)
    j1 = json.dumps(doc1, sort_keys=True, separators=(",", ":"))
    j2 = json.dumps(doc2, sort_keys=True, separators=(",", ":"))
    assert j1 == j2


def test_dump_round_trip(tmp_path):
    doc = _run_dump(tmp_path)
    doc.pop("generated_at", None)
    doc.pop("generator", None)
    canonical = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    re_emitted = json.dumps(
        json.loads(canonical), sort_keys=True, separators=(",", ":")
    )
    assert canonical == re_emitted


def test_table_entries_have_expected_shape(tmp_path):
    doc = _run_dump(tmp_path)
    for name, body in doc["tables"].items():
        assert "columns" in body
        assert "indexes" in body
        assert "uniques" in body
        for col_name, col in body["columns"].items():
            for key in ("type", "nullable", "primary_key", "default", "foreign_keys"):
                assert key in col, f"{name}.{col_name} missing key: {key}"
            assert isinstance(col["nullable"], bool)
            assert isinstance(col["primary_key"], bool)
            assert isinstance(col["foreign_keys"], list)
