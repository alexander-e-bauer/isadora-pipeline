"""Manual position entry helper — bulk-insert positions for an account.

Used for CSV import and ad-hoc seeding outside the HTTP layer.

v1 exports one function: insert_positions_bulk.

Caller is responsible for the surrounding transaction; this helper does
NOT commit (matches emit_event semantics).
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from xyz.tenant.events import emit_event
from xyz.tenant.models import Account, Client, Position


def _resolve_firm_id(db: Session, account_id: int) -> int:
    """Look up the firm_id for an account via the Account → Client → Firm chain.

    Raises ValueError if the account does not exist. We need firm_id so the
    emitted position.added events land on the correct firm's chain — without
    it, /events firm-scoping queries return nothing for these rows.
    """
    firm_id = db.scalar(
        select(Client.firm_id)
        .join(Account, Account.client_id == Client.id)
        .where(Account.id == account_id)
    )
    if firm_id is None:
        raise ValueError(f"Account {account_id} does not exist (or has no client)")
    return firm_id


def insert_positions_bulk(
    db: Session,
    account_id: int,
    rows: list[dict],
    *,
    actor_user_id: int | None = None,
) -> list[int]:
    """Bulk-insert position rows for an account.

    Emits one ``position.added`` event per row (chained on the account's
    firm).  Returns the new position ids in insertion order.

    Parameters
    ----------
    db:
        Active SQLAlchemy Session.  The caller owns the transaction;
        this helper does NOT commit.
    account_id:
        Target account id.  Firm is resolved automatically via the
        Account → Client → Firm chain so emitted events are firm-scoped.
        Raises ValueError if the account doesn't exist.
    rows:
        List of dicts matching Position column names.  Required keys:
        ``symbol``, ``asset_class``, ``qty``, ``cost_basis``.
        Optional keys: ``lot_method``, ``option_type``, ``strike``,
        ``expiry``, ``multiplier``.
    actor_user_id:
        Optional user id to record on each emitted event (e.g., the
        advisor running a CSV import).  Pass None for system-driven
        imports.

    Returns
    -------
    list[int]
        The new position ids in the same order as ``rows``.
    """
    firm_id = _resolve_firm_id(db, account_id)
    ids: list[int] = []

    for row in rows:
        position = Position(
            account_id=account_id,
            symbol=row["symbol"],
            asset_class=row["asset_class"],
            qty=row["qty"],
            cost_basis=row["cost_basis"],
            lot_method=row.get("lot_method", "FIFO"),
            option_type=row.get("option_type"),
            strike=row.get("strike"),
            expiry=row.get("expiry"),
            multiplier=row.get("multiplier", 100),
        )
        db.add(position)
        db.flush()  # get the auto-generated id

        emit_event(
            db=db,
            kind="position.added",
            firm_id=firm_id,
            actor_user_id=actor_user_id,
            payload={
                "position_id": position.id,
                "account_id": account_id,
                "symbol": position.symbol,
                "asset_class": str(position.asset_class.value if hasattr(position.asset_class, "value") else position.asset_class),
                "qty": str(position.qty),
                "cost_basis": str(position.cost_basis),
                "source": "manual_entry",
            },
        )

        ids.append(position.id)

    return ids
