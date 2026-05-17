"""ORM models for the Polygon Options data layer.

All models are registered on the shared engine Base from
xyz.finazon_service.sql_service so that init_db()'s create_all() creates
these tables alongside the existing financial-data tables.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Column,
    Date,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)

# SQLite does not support BigInteger as a primary-key autoincrement column
# (SQLite requires the INTEGER affinity for its ROWID alias).  Use Integer
# with a Postgres variant so that production gets BIGINT and tests (SQLite)
# get the INTEGER needed for autoincrement.
_BigPK = Integer().with_variant(BigInteger(), "postgresql")

from xyz.finazon_service.base import Base


class OptionHistoricalEod(Base):
    """Daily OHLCV bars per options contract — primary backtest table."""

    __tablename__ = "option_historical_eod"

    id = Column(_BigPK, primary_key=True, autoincrement=True)
    underlying = Column(String(20), nullable=False)
    contract_ticker = Column(String(50), nullable=False)  # e.g. O:AAPL250117C00185000
    expiry = Column(Date, nullable=False)
    strike = Column(Numeric(10, 4), nullable=False)
    option_type = Column(String(4), nullable=False)  # 'CALL' or 'PUT'
    date = Column(Date, nullable=False)
    open = Column(Numeric(10, 4))
    high = Column(Numeric(10, 4))
    low = Column(Numeric(10, 4))
    close = Column(Numeric(10, 4))
    volume = Column(BigInteger)
    open_interest = Column(BigInteger, nullable=True)
    implied_vol = Column(Numeric(10, 6), nullable=True)
    delta = Column(Numeric(10, 6), nullable=True)
    gamma = Column(Numeric(10, 6), nullable=True)
    theta = Column(Numeric(10, 6), nullable=True)
    vega = Column(Numeric(10, 6), nullable=True)
    rho = Column(Numeric(10, 6), nullable=True)
    fetched_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("contract_ticker", "date", name="uq_option_hist_eod_contract_date"),
        Index("ix_option_hist_eod_underlying_date", "underlying", "date"),
        Index("ix_option_hist_eod_underlying_expiry_strike", "underlying", "expiry", "strike"),
    )


class OptionChains(Base):
    """Snapshot of a full option chain at a given moment."""

    __tablename__ = "option_chains"

    id = Column(_BigPK, primary_key=True, autoincrement=True)
    underlying = Column(String(20), nullable=False)
    asof_at = Column(DateTime(timezone=True), nullable=False)
    contract_ticker = Column(String(50), nullable=False)
    expiry = Column(Date, nullable=False)
    strike = Column(Numeric(10, 4), nullable=False)
    option_type = Column(String(4), nullable=False)
    bid = Column(Numeric(10, 4))
    ask = Column(Numeric(10, 4))
    last = Column(Numeric(10, 4))
    mid = Column(Numeric(10, 4))
    volume = Column(BigInteger)
    open_interest = Column(BigInteger)
    implied_vol = Column(Numeric(10, 6), nullable=True)
    delta = Column(Numeric(10, 6), nullable=True)
    gamma = Column(Numeric(10, 6), nullable=True)
    theta = Column(Numeric(10, 6), nullable=True)
    vega = Column(Numeric(10, 6), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "underlying", "asof_at", "contract_ticker",
            name="uq_option_chains_underlying_asof_contract",
        ),
        Index("ix_option_chains_underlying_asof_at", "underlying", "asof_at"),
    )


class OptionIvSurface(Base):
    """Downsampled IV-by-expiry-and-strike for regime analysis."""

    __tablename__ = "option_iv_surface"

    id = Column(_BigPK, primary_key=True, autoincrement=True)
    underlying = Column(String(20), nullable=False)
    asof_date = Column(Date, nullable=False)
    expiry = Column(Date, nullable=False)
    strike = Column(Numeric(10, 4), nullable=False)
    option_type = Column(String(4), nullable=False)
    implied_vol = Column(Numeric(10, 6), nullable=False)
    delta = Column(Numeric(10, 6))
    fetched_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "underlying", "asof_date", "expiry", "strike", "option_type",
            name="uq_option_iv_surface",
        ),
        Index("ix_option_iv_surface_underlying_asof_date", "underlying", "asof_date"),
    )


class OptionQuotes(Base):
    """Sparse on-demand single-contract quote cache."""

    __tablename__ = "option_quotes"

    id = Column(_BigPK, primary_key=True, autoincrement=True)
    contract_ticker = Column(String(50), nullable=False)
    underlying = Column(String(20), nullable=False)
    fetched_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    bid = Column(Numeric(10, 4))
    ask = Column(Numeric(10, 4))
    last = Column(Numeric(10, 4))
    mid = Column(Numeric(10, 4))
    volume = Column(BigInteger)
    open_interest = Column(BigInteger)
    implied_vol = Column(Numeric(10, 6), nullable=True)
    delta = Column(Numeric(10, 6), nullable=True)
    gamma = Column(Numeric(10, 6), nullable=True)
    theta = Column(Numeric(10, 6), nullable=True)
    vega = Column(Numeric(10, 6), nullable=True)

    __table_args__ = (
        Index("ix_option_quotes_contract_fetched_at", "contract_ticker", "fetched_at"),
    )
