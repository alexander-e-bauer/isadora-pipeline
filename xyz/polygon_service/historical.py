"""Helpers to fetch Polygon historical EOD bars and persist them.

``fetch_and_persist_historical_eod`` is the main entry point consumed by
the AUTHOR/BACKTEST subagents in Chunk 4.  It fetches daily bars for each
supplied contract and upserts rows into ``option_historical_eod``, skipping
(contract_ticker, date) pairs that already exist.

Dialect note:
  - PostgreSQL: uses ``INSERT … ON CONFLICT DO NOTHING`` natively.
  - SQLite (tests): uses ``INSERT OR IGNORE`` via SQLAlchemy's
    ``sqlite_on_conflict_do_nothing`` parameter where supported, falling
    back to a manual existence check otherwise.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import inspect
from sqlalchemy.orm import Session

from xyz.polygon_service.models import OptionHistoricalEod
from xyz.polygon_service.options_client import Aggregate, OptionContract, OptionsClient


def _dialect_name(session: Session) -> str:
    return session.bind.dialect.name  # type: ignore[union-attr]


def _upsert_rows(session: Session, rows: list[dict]) -> int:
    """Insert rows, skipping duplicates.  Returns the number inserted."""
    if not rows:
        return 0

    dialect = _dialect_name(session)
    inserted = 0

    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = pg_insert(OptionHistoricalEod).values(rows)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["contract_ticker", "date"]
        )
        result = session.execute(stmt)
        inserted = result.rowcount if result.rowcount >= 0 else len(rows)
    else:
        # SQLite (and anything else): check existence before inserting.
        for row in rows:
            exists = (
                session.query(OptionHistoricalEod)
                .filter_by(contract_ticker=row["contract_ticker"], date=row["date"])
                .first()
            )
            if exists is None:
                session.add(OptionHistoricalEod(**row))
                inserted += 1

    session.flush()
    return inserted


def fetch_and_persist_historical_eod(
    client: OptionsClient,
    db: Session,
    underlying: str,
    contracts: list[OptionContract],
    start_date: date,
    end_date: date,
) -> int:
    """Fetch historical EOD bars for each contract and upsert into option_historical_eod.

    Args:
        client:     Initialised OptionsClient.
        db:         Open SQLAlchemy Session (caller manages commit/rollback).
        underlying: The equity ticker symbol, e.g. "AAPL".
        contracts:  Filtered list of OptionContract objects to process.
        start_date: Inclusive start of the historical range.
        end_date:   Inclusive end of the historical range.

    Returns:
        Total number of new rows inserted (duplicates are skipped).
    """
    total_inserted = 0

    for contract in contracts:
        aggs: list[Aggregate] = client.get_historical_aggs(
            contract.contract_ticker,
            from_date=start_date,
            to_date=end_date,
            timespan="day",
        )

        rows = []
        for agg in aggs:
            bar_date = agg.timestamp.date()
            rows.append(
                {
                    "underlying": underlying.upper(),
                    "contract_ticker": contract.contract_ticker,
                    "expiry": contract.expiry,
                    "strike": contract.strike,
                    "option_type": contract.option_type,
                    "date": bar_date,
                    "open": agg.open,
                    "high": agg.high,
                    "low": agg.low,
                    "close": agg.close,
                    "volume": int(agg.volume) if agg.volume is not None else None,
                    "open_interest": None,
                    "implied_vol": None,
                    "delta": None,
                    "gamma": None,
                    "theta": None,
                    "vega": None,
                    "rho": None,
                    "fetched_at": datetime.utcnow(),
                }
            )

        total_inserted += _upsert_rows(db, rows)

    return total_inserted
