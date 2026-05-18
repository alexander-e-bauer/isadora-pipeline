"""Hash-chained event log — engine-side mirror of server's emit_event.

Public API
----------
emit_event(*, db, kind, firm_id, actor_user_id, payload) -> str
    Insert one immutable event row into the ``events`` table and return
    the event's id (UUID hex string).  The caller is responsible for the
    surrounding transaction; this function does NOT commit.

Chain guarantee
---------------
Every event carries:
  - ``event_hash``      — SHA-256 over the event's own canonical fields.
  - ``prev_event_hash`` — hash of the immediately preceding event in the
                          same firm's chain (None for the first event).

This implementation is byte-for-byte identical to server's
app/events/emit.py so that engine and server share the same hash chain.
Any event emitted by engine is indistinguishable (at the hash level)
from one emitted by server — the hash chain is portable across both
processes writing to the same Postgres table.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from xyz.observability.request_id import get_request_id
from xyz.tenant.models import Event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers  (identical to server/app/events/emit.py)
# ---------------------------------------------------------------------------

def _strip_observability_keys(payload: dict) -> dict:
    """Drop underscore-prefixed observability keys (``_request_id``, etc.)
    before hashing — see server's ``app/events/emit.py`` for the
    rationale.  Both apps MUST share this strip-rule or the cross-app
    hash chain diverges.
    """
    return {k: v for k, v in payload.items() if not k.startswith("_")}


def _compute_event_hash(
    *,
    kind: str,
    firm_id: int | None,
    actor_user_id: int | None,
    payload: dict,
    prev_event_hash: str | None,
    created_at_iso: str,
) -> str:
    """Return SHA-256 hex digest of the canonical JSON representation.

    The canonical form is compact, sorted-key JSON.  Must be identical to
    server's implementation — same field set, same separators, same key
    ordering, same JSON-native type enforcement.  Observability keys are
    stripped from the payload before hashing (see _strip_observability_keys).
    """
    canonical = json.dumps(
        {
            "kind": kind,
            "firm_id": firm_id,
            "actor_user_id": actor_user_id,
            "payload": _strip_observability_keys(payload),
            "prev_event_hash": prev_event_hash,
            "created_at": created_at_iso,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _latest_event_for_firm(db: Session, firm_id: int | None) -> Event | None:
    """Return the most recent event for ``firm_id`` with a FOR UPDATE lock.

    The lock serialises concurrent inserts within the same firm.
    On SQLite (used in tests) ``with_for_update()`` is silently ignored.
    """
    if firm_id is None:
        where_clause = Event.firm_id.is_(None)
    else:
        where_clause = Event.firm_id == firm_id

    stmt = (
        select(Event)
        .where(where_clause)
        .order_by(Event.created_at.desc(), Event.id.desc())
        .with_for_update()
        .limit(1)
    )
    return db.scalar(stmt)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def emit_event(
    *,
    db: Any,               # Session — typed as Any to avoid circular imports
    kind: str,
    firm_id: int | None,
    actor_user_id: int | None,
    payload: dict | None = None,
) -> str:
    """Append one immutable event to the hash-chained audit log.

    Parameters
    ----------
    db:
        Active SQLAlchemy ``Session``.  The caller owns the transaction;
        this function flushes but does NOT commit.
    kind:
        Event kind string.
    firm_id:
        Tenant firm id.  Pass ``None`` for system-wide events.
    actor_user_id:
        Id of the user who triggered the event, or ``None`` for system events.
    payload:
        Arbitrary JSON-serialisable dict with event-specific data.

    Returns
    -------
    str
        The new event's id (32-char UUID hex string).
    """
    payload = dict(payload or {})  # copy: don't mutate caller's dict

    # 0. Stamp the active request id into the payload (Task 5.2 — mirror
    # of server's app/events/emit.py).  When emitted outside a request
    # scope (background tasks, scripts) the ContextVar is None and we
    # omit the key — explicit absence is cleaner than literal None for
    # downstream Cloud Logging filters.
    if "_request_id" not in payload:
        active_request_id = get_request_id()
        if active_request_id is not None:
            payload["_request_id"] = active_request_id

    # 1. Lock the latest event in this firm's chain to prevent forks.
    prev_event = _latest_event_for_firm(db, firm_id)
    prev_event_hash: str | None = prev_event.event_hash if prev_event else None

    # 2. Choose the timestamp NOW so it is included in the hash.
    created_at = datetime.now(timezone.utc)
    created_at_iso = created_at.isoformat()

    # 3. Compute the hash over canonical fields.
    event_hash = _compute_event_hash(
        kind=kind,
        firm_id=firm_id,
        actor_user_id=actor_user_id,
        payload=payload,
        prev_event_hash=prev_event_hash,
        created_at_iso=created_at_iso,
    )

    # 4. Build and stage the event row (do NOT commit — caller owns the tx).
    event_id = uuid.uuid4().hex
    event = Event(
        id=event_id,
        created_at=created_at,
        firm_id=firm_id,
        actor_user_id=actor_user_id,
        kind=kind,
        payload=payload,
        prev_event_hash=prev_event_hash,
        event_hash=event_hash,
    )
    db.add(event)
    db.flush()

    logger.info(
        "event kind=%s firm_id=%s actor=%s id=%s prev_hash=%.8s… hash=%.8s…",
        kind,
        firm_id,
        actor_user_id,
        event_id,
        prev_event_hash or "None",
        event_hash,
    )

    return event_id
