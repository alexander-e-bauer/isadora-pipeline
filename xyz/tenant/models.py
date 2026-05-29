"""Tenant ORM model mirror — engine-side read-only copy of server's tenant schema.

These model definitions are column-for-column identical to server's
app/models/*.py so that engine can query tenant tables (firms, users,
clients, accounts, strategies, deployments, positions, trades) and write
to the events table.

IMPORTANT: Engine MUST NOT call Base.metadata.create_all() for these
models in production — server's Alembic migrations own all DDL for the
tenant schema.  Engine reads (and writes events only).

A separate declarative Base is used here so engine's finazon-schema
models (in xyz/finazon_service/sql_service.py) and the tenant models
live in different metadata registries.  This prevents any accidental
create_all call from affecting the wrong set of tables.
"""
from __future__ import annotations

import enum
import hashlib
from datetime import date, datetime  # noqa: F401 — date used by Position.expiry / BacktestResult.start_date
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from xyz.tenant.encrypted_types import EncryptedString, EncryptedText


# Separate Base from engine's finazon_service Base to keep metadata registries
# isolated.
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Payload column type — JSONB on Postgres, JSON on SQLite (tests)
# ---------------------------------------------------------------------------

PayloadType = JSON().with_variant(JSONB(), "postgresql")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class UserRole(enum.Enum):
    ADVISOR = "ADVISOR"
    COMPLIANCE_OFFICER = "COMPLIANCE_OFFICER"
    FIRM_ADMIN = "FIRM_ADMIN"


class ActionFamily(str, enum.Enum):
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    ROLL = "ROLL"
    HEDGE = "HEDGE"
    PORTFOLIO = "PORTFOLIO"
    EVENT = "EVENT"
    STATE = "STATE"


class AutonomyLevel(str, enum.Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"


class BrokerTier(str, enum.Enum):
    MANUAL = "MANUAL"
    READ_ONLY = "READ_ONLY"
    TRADING_AUTH = "TRADING_AUTH"


class StrategyKind(str, enum.Enum):
    DECLARATIVE = "declarative"
    SCRIPTED = "scripted"


class StrategyState(str, enum.Enum):
    DRAFT = "DRAFT"
    BACKTESTING = "BACKTESTING"
    READY = "READY"
    APPROVED = "APPROVED"
    PUBLISHED = "PUBLISHED"
    ARCHIVED = "ARCHIVED"


class DeploymentState(str, enum.Enum):
    PROPOSED = "PROPOSED"
    ADVISOR_APPROVED = "ADVISOR_APPROVED"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    WINDING_DOWN = "WINDING_DOWN"
    CLOSED = "CLOSED"


class SubscriptionStatus(str, enum.Enum):
    TRIALING = "TRIALING"
    ACTIVE = "ACTIVE"
    PAST_DUE = "PAST_DUE"
    CANCELED = "CANCELED"
    INCOMPLETE = "INCOMPLETE"
    UNPAID = "UNPAID"


class AssetClass(str, enum.Enum):
    EQUITY = "EQUITY"
    OPTION = "OPTION"


class LotMethod(str, enum.Enum):
    FIFO = "FIFO"
    LIFO = "LIFO"
    HIFO = "HIFO"
    SPECIFIC = "SPECIFIC"


class OptionType(str, enum.Enum):
    CALL = "CALL"
    PUT = "PUT"


class TradeState(str, enum.Enum):
    PROPOSED = "PROPOSED"
    ADVISOR_APPROVED = "ADVISOR_APPROVED"
    EXECUTED = "EXECUTED"
    REJECTED = "REJECTED"
    CANCELED = "CANCELED"


class RiskClass(str, enum.Enum):
    RISK_INCREASING = "RISK_INCREASING"
    RISK_NEUTRAL = "RISK_NEUTRAL"
    RISK_REDUCING = "RISK_REDUCING"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Firm(Base):
    __tablename__ = "firms"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    allow_l5: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), nullable=False, default=False
    )
    max_autonomy_cap: Mapped[str | None] = mapped_column(String(4), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Firm id={self.id} name={self.name!r}>"


def hash_email(email: str) -> str:
    """Return lowercase-stripped SHA-256 hex digest of email."""
    normalised = email.strip().lower()
    return hashlib.sha256(normalised.encode()).hexdigest()


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    firm_id: Mapped[int] = mapped_column(
        ForeignKey("firms.id"), nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(EncryptedText, nullable=False)
    email_hash: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    full_name: Mapped[str | None] = mapped_column(EncryptedString(100), nullable=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean, server_default=text("true"), nullable=False, default=True
    )
    stripe_customer_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True, unique=True
    )
    trial_ends_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    firm: Mapped["Firm"] = relationship("Firm", lazy="joined")

    @staticmethod
    def hash_email(email: str) -> str:
        return hash_email(email)

    def __repr__(self) -> str:
        return f"<User id={self.id} role={self.role}>"


class UserClientAccess(Base):
    __tablename__ = "user_client_access"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=False, index=True
    )
    client_id: Mapped[int] = mapped_column(
        ForeignKey("clients.id"), nullable=False, index=True
    )
    access_level: Mapped[str] = mapped_column(
        String(50), server_default="read", nullable=False, default="read"
    )
    granted_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    granted_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )

    def __repr__(self) -> str:
        return (
            f"<UserClientAccess id={self.id} user_id={self.user_id} "
            f"client_id={self.client_id}>"
        )


class Client(Base):
    __tablename__ = "clients"

    id: Mapped[int] = mapped_column(primary_key=True)
    firm_id: Mapped[int] = mapped_column(
        ForeignKey("firms.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(EncryptedString(255), nullable=False)
    external_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    household_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Client id={self.id} firm_id={self.firm_id}>"


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(primary_key=True)
    client_id: Mapped[int] = mapped_column(
        ForeignKey("clients.id"), nullable=False, index=True
    )
    nickname: Mapped[str] = mapped_column(String(100), nullable=False)
    account_type: Mapped[str] = mapped_column(String(32), nullable=False)
    base_currency: Mapped[str] = mapped_column(
        String(3), nullable=False, server_default="USD"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Account id={self.id} client_id={self.client_id} nickname={self.nickname!r}>"


class AccountAutonomy(Base):
    __tablename__ = "account_autonomy"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id"), nullable=False, index=True
    )
    action_family: Mapped[ActionFamily] = mapped_column(
        Enum(ActionFamily), nullable=False
    )
    level: Mapped[AutonomyLevel] = mapped_column(
        Enum(AutonomyLevel), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "account_id", "action_family", name="uq_account_autonomy_family"
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<AccountAutonomy id={self.id} account_id={self.account_id} "
            f"family={self.action_family} level={self.level}>"
        )


class BrokerConnection(Base):
    __tablename__ = "broker_connections"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id"),
        nullable=False,
        index=True,
        unique=True,
    )
    tier: Mapped[BrokerTier] = mapped_column(
        Enum(BrokerTier),
        nullable=False,
        server_default=BrokerTier.MANUAL.value,
    )
    broker_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    external_account_id: Mapped[str | None] = mapped_column(
        EncryptedString(128), nullable=True
    )
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<BrokerConnection id={self.id} account_id={self.account_id} "
            f"tier={self.tier}>"
        )


class Strategy(Base):
    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(primary_key=True)
    firm_id: Mapped[int] = mapped_column(ForeignKey("firms.id"), nullable=False, index=True)
    author_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    kind: Mapped[StrategyKind] = mapped_column(
        Enum(StrategyKind, name="strategykind"),
        nullable=False,
        server_default=StrategyKind.DECLARATIVE.value,
    )
    dsl_json: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
        server_default="{}",
    )
    state: Mapped[StrategyState] = mapped_column(
        Enum(StrategyState, name="strategystate"),
        nullable=False,
        server_default=StrategyState.DRAFT.value,
    )
    dsl_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # FK + index mirror server's app.models.strategy.  Engine does NOT define
    # a BacktestResult mirror (server owns backtest_results); the FK is still
    # valid in SQLAlchemy metadata because the referenced table need only be
    # resolvable at DDL/query time, which engine never performs against this
    # column.  Keeping the FK declaration here is what schema-parity demands.
    backtest_result_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("backtest_results.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_strategies_firm_name_version", "firm_id", "name", "version", unique=True),
    )

    def __repr__(self) -> str:
        return (
            f"<Strategy id={self.id} firm_id={self.firm_id} "
            f"name={self.name!r} version={self.version} state={self.state}>"
        )


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_id: Mapped[int] = mapped_column(
        ForeignKey("strategies.id"), nullable=False, index=True
    )
    strategy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id"), nullable=False, index=True
    )
    state: Mapped[DeploymentState] = mapped_column(
        Enum(DeploymentState, name="deploymentstate"),
        nullable=False,
        server_default=DeploymentState.PROPOSED.value,
    )
    autonomy_snapshot_json: Mapped[dict | None] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"), nullable=True
    )
    advisor_approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    paused_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    winding_down_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<Deployment id={self.id} strategy_id={self.strategy_id} "
            f"account_id={self.account_id} state={self.state}>"
        )


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(primary_key=True)
    firm_id: Mapped[int] = mapped_column(
        ForeignKey("firms.id"), nullable=False, unique=True, index=True
    )
    stripe_customer_id: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True
    )
    stripe_subscription_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True, unique=True
    )
    status: Mapped[SubscriptionStatus] = mapped_column(
        Enum(SubscriptionStatus),
        nullable=False,
        server_default=SubscriptionStatus.TRIALING.value,
    )
    seats: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    current_period_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    trial_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    accounts_included: Mapped[int | None] = mapped_column(Integer, nullable=True)
    overage_per_account_cents: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<Subscription id={self.id} firm_id={self.firm_id} "
            f"status={self.status}>"
        )


class PaymentMethod(Base):
    __tablename__ = "payment_methods"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"), nullable=False, index=True
    )
    stripe_payment_method_id: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True
    )
    brand: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last4: Mapped[str | None] = mapped_column(String(4), nullable=True)
    exp_month: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exp_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_default: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<PaymentMethod id={self.id} user_id={self.user_id} "
            f"brand={self.brand} last4={self.last4}>"
        )


class ProcessedStripeEvent(Base):
    __tablename__ = "processed_stripe_events"

    stripe_event_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class Event(Base):
    """Immutable audit-log event row — engine writes here via emit_event."""

    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    firm_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("firms.id"), nullable=True, index=True
    )
    actor_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True, index=True
    )
    kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    payload: Mapped[dict] = mapped_column(PayloadType, nullable=False, server_default="{}")
    prev_event_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    event_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    __table_args__ = (
        Index("ix_events_firm_id_created_at", "firm_id", "created_at"),
        Index("ix_events_kind_created_at", "kind", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<Event id={self.id} kind={self.kind!r} firm_id={self.firm_id}>"
        )


class Position(Base):
    """Read-only mirror of server's positions table."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id"), nullable=False, index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    asset_class: Mapped[AssetClass] = mapped_column(Enum(AssetClass), nullable=False)
    qty: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    cost_basis: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    lot_method: Mapped[LotMethod] = mapped_column(
        Enum(LotMethod), nullable=False, server_default=LotMethod.FIFO.value
    )
    # Option-only fields — required when asset_class=OPTION, NULL when asset_class=EQUITY.
    option_type: Mapped[OptionType | None] = mapped_column(Enum(OptionType), nullable=True)
    strike: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    expiry: Mapped[date | None] = mapped_column(Date, nullable=True)
    multiplier: Mapped[int | None] = mapped_column(
        Integer, nullable=True, server_default="100"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_positions_account_symbol", "account_id", "symbol"),
    )

    def __repr__(self) -> str:
        return (
            f"<Position id={self.id} account_id={self.account_id} "
            f"symbol={self.symbol!r} asset_class={self.asset_class} qty={self.qty}>"
        )


class BacktestResult(Base):
    """Read-only mirror of server's backtest_results table.

    Engine NEVER reads or writes this table — server owns backtest result
    artifacts.  The mirror exists only so that Strategy.backtest_result_id's
    ForeignKey resolves against in-metadata tables (otherwise SQLite
    create_all in tests fails with NoReferencedTableError).
    """

    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    strategy_id: Mapped[int] = mapped_column(
        ForeignKey("strategies.id"), nullable=False, index=True
    )
    strategy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    metrics_json: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
        server_default="{}",
    )
    content_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index(
            "ix_backtest_results_strategy_id_version",
            "strategy_id",
            "strategy_version",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<BacktestResult id={self.id} strategy_id={self.strategy_id} "
            f"v{self.strategy_version} hash={self.content_hash[:8]}...>"
        )


class ForecastResult(Base):
    """Read-only mirror of server's forecast_results table.

    Engine NEVER reads or writes this table — server owns forecast result
    artifacts. The mirror exists only for cross-app schema-parity checking
    (test_tenant_metadata_matches_server will verify engine and server
    definitions match column-by-column when the server-side model lands).
    """

    __tablename__ = "forecast_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    firm_id: Mapped[int] = mapped_column(ForeignKey("firms.id"), nullable=False, index=True)
    strategy_id: Mapped[int] = mapped_column(ForeignKey("strategies.id"), nullable=False, index=True)
    strategy_version: Mapped[int] = mapped_column(Integer, nullable=False)
    research_artifact_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    t0: Mapped[date] = mapped_column(Date, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    n_paths: Mapped[int] = mapped_column(Integer, nullable=False)
    forecast_seed: Mapped[int] = mapped_column(BigInteger, nullable=False)
    calibration_source: Mapped[str] = mapped_column(String(32), nullable=False)
    calibrated_params_json: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
        server_default="{}",
    )
    t0_market_context_json: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
        server_default="{}",
    )
    results_json: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
        server_default="{}",
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "firm_id", "strategy_id", "strategy_version", "content_hash",
            name="uq_forecast_results_firm_strategy_version_hash"
        ),
        Index(
            "forecast_results_strategy_idx",
            "firm_id", "strategy_id", "strategy_version", "t0",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ForecastResult id={self.id} strategy_id={self.strategy_id} "
            f"v{self.strategy_version} hash={self.content_hash[:8]}...>"
        )


class Trade(Base):
    """Read-only mirror of server's trades table."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id"), nullable=False, index=True
    )
    # deployment_id nullable — non-strategy trades exist (e.g. manual adjustments).
    deployment_id: Mapped[int | None] = mapped_column(
        ForeignKey("deployments.id"), nullable=True, index=True
    )
    state: Mapped[TradeState] = mapped_column(
        Enum(TradeState), nullable=False, server_default=TradeState.PROPOSED.value
    )
    action_family: Mapped[ActionFamily] = mapped_column(
        Enum(ActionFamily), nullable=False
    )
    leaf_action: Mapped[str] = mapped_column(String(64), nullable=False)
    risk_class: Mapped[RiskClass] = mapped_column(Enum(RiskClass), nullable=False)
    order_ticket_json: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=False,
        server_default="{}",
    )
    compliance_verdict_json: Mapped[dict | None] = mapped_column(
        JSON().with_variant(JSONB(), "postgresql"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    advisor_approved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    executed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    rejected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    canceled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return (
            f"<Trade id={self.id} account_id={self.account_id} "
            f"state={self.state} action_family={self.action_family}>"
        )
