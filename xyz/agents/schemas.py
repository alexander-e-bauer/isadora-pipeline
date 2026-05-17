"""Pydantic models shared by all engine-side subagents."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field, model_validator

_BACKTEST_MIN_DAYS = 30


class ResearchInput(BaseModel):
    firm_id: int
    account_id: int | None = None
    symbol: str | None = None
    brief: str | None = None
    actor_user_id: int | None = None

    def primary_target(self) -> str:
        """Short human-readable label for what this research is about."""
        return self.symbol or (self.brief or "")[:80]


class ResearchSection(BaseModel):
    title: str
    body: str
    citations: list[dict] = Field(default_factory=list)


class ResearchArtifact(BaseModel):
    """Structured research output — what the agent emits + persists."""

    symbol: str | None
    summary: str
    iv_regime: ResearchSection | None        # IV rank, IV vs RV, term structure
    earnings_calendar: ResearchSection | None
    news_headlines: ResearchSection | None   # last 5 items with citations
    peer_comparison: ResearchSection | None  # optional, may be None
    generated_at: datetime
    content_hash: str  # sha256 of the canonical JSON form (excl. generated_at + hash)


# ---------------------------------------------------------------------------
# AUTHOR subagent
# ---------------------------------------------------------------------------

class AuthorInput(BaseModel):
    """Inputs to the AUTHOR subagent.

    The advisor supplies a free-form ``brief`` describing what they want
    (e.g. "income on AAPL, low assignment risk, monthlies, IRA suitable").
    The agent uses ``firm_id`` and ``actor_user_id`` to stamp the resulting
    DSL and the ``strategy.draft`` audit event.

    ``target_account_ids`` is informational — it allows the agent to tune
    suitability constraints (e.g. IRA accounts get a covered-call default).
    A full account-level inspection is deferred to PROPOSE.
    """

    firm_id: int
    brief: str = Field(..., min_length=1)
    actor_user_id: int | None = None
    target_account_ids: list[int] = Field(default_factory=list)


class AuthorArtifact(BaseModel):
    """Structured AUTHOR output — a draft strategy DSL + Reg-BI rationale.

    ``dsl`` is the canonical DSL document (JSON-Schema validated upstream).
    ``rationale`` is a 1–2 paragraph plain-English justification used as a
    Reg-BI stub the advisor will review and edit before approving.
    ``template`` echoes the family the agent selected (covered_call,
    cash_secured_put, collar) for convenience — it is also embedded in
    ``dsl["template"]``.
    """

    template: Literal["covered_call", "cash_secured_put", "collar"]
    dsl: dict
    rationale: str
    generated_at: datetime
    content_hash: str  # sha256 of dsl + rationale (excl. generated_at + hash)


# ---------------------------------------------------------------------------
# BACKTEST subagent  (Task 4.3)
# ---------------------------------------------------------------------------

class BacktestInput(BaseModel):
    """Inputs to the BACKTEST subagent.

    The caller supplies the DSL document directly (not a strategy_id
    lookup) because the engine has no read-write contract with server's
    strategies table — server is the source-of-truth for the DSL.  The
    ``strategy_id`` + ``strategy_version`` fields are echoed back in the
    output artifact for the server to bind the result to the right row.

    ``actor_user_id`` is the advisor who kicked off the backtest; it is
    used to stamp the ``backtest.result`` audit event.
    """

    strategy_id: int
    strategy_version: int
    firm_id: int
    actor_user_id: int | None = None
    start_date: date
    end_date: date
    dsl: dict

    @model_validator(mode="after")
    def _validate_window(self) -> "BacktestInput":
        # Refuse impossibly short windows.  CAGR is annualised via
        # (1 + r)^(252/n) — a 3-day window with 1% return annualises to
        # ~2,900%, which would be displayed to advisors as a real CAGR
        # and flagged as a data error in compliance review.  30 calendar
        # days is the smallest interval where CAGR is meaningful for
        # monthly covered-call strategies.
        if self.end_date < self.start_date + timedelta(days=_BACKTEST_MIN_DAYS):
            raise ValueError(
                f"end_date must be at least {_BACKTEST_MIN_DAYS} days after start_date "
                "(annualised metrics are not meaningful on shorter windows)"
            )
        return self


class BacktestArtifact(BaseModel):
    """Structured BACKTEST output — the immutable, hashed audit artifact.

    The full NAV series is intentionally omitted from this payload (the
    plan calls it "too big for a JSON column").  The ``metrics`` dict is
    what the server persists in ``backtest_results.metrics_json`` and is
    what callers compare against in the spec acceptance criteria.

    ``content_hash`` excludes ``generated_at`` so re-running the same
    backtest over the same data yields a byte-identical digest — the
    determinism guarantee that 17a-4 audit relies on.
    """

    strategy_id: int
    strategy_version: int
    firm_id: int
    start_date: date
    end_date: date
    metrics: dict
    n_trades: int
    content_hash: str  # sha256 of dsl + dates + metrics (excl. generated_at)
    generated_at: datetime


# ---------------------------------------------------------------------------
# PROPOSE subagent  (Task 4.4)
# ---------------------------------------------------------------------------

class ProposeInput(BaseModel):
    """Inputs to the PROPOSE subagent.

    The caller identifies the deployment to evaluate.  ``firm_id`` is
    required so the agent's audit event is correctly scoped — the
    deployment look-up itself is performed against the tenant DB and is
    cross-checked against ``firm_id`` before any further work runs.

    ``actor_user_id`` is the advisor whose dashboard triggered the run
    (or ``None`` for a system-tick proposal that's not in v1 — v1
    triggers are advisor-initiated).
    """

    firm_id: int
    deployment_id: int
    actor_user_id: int | None = None


class TradeTicket(BaseModel):
    """A single proposed-trade ticket emitted by PROPOSE.

    Each ticket carries everything COMPLIANCE (Task 4.5) needs to render
    a verdict and everything the dashboard needs to surface for advisor
    approval.  v1 only generates one ticket per PROPOSE call (one
    deployment, one underlying, one entry), but the artifact is a list
    so the v1.5 multi-leg/multi-symbol expansion does not break the
    response shape.

    Required spec fields:
    - leaf_action          (spec §6 leaf taxonomy; v1 column only)
    - action_family        (spec §6 family taxonomy)
    - risk_class           (spec §6 risk taxonomy)
    - account_id           (which account this proposes to trade in)
    - deployment_id        (which strategy deployment this came from)
    - order_ticket_json    (broker-facing detail — contract, side, limit, qty)
    - autonomy_level_required / autonomy_level_account  (spec §7 matrix)
    - reg_bi_rationale     (spec §10 advisor-reviewable stub)
    - generated_at         (audit timestamp; NOT in content_hash)
    """

    account_id: int
    deployment_id: int
    leaf_action: Literal[
        "OPEN_NEW",
        "CLOSE_WIN",
        "CLOSE_LOSS",
        "ROLL_OUT_ONLY",
    ]
    action_family: Literal["OPEN", "CLOSE", "ROLL", "HEDGE", "PORTFOLIO", "EVENT", "STATE"]
    risk_class: Literal["RISK_INCREASING", "RISK_NEUTRAL", "RISK_REDUCING"]
    order_ticket_json: dict
    autonomy_level_required: str  # L0..L5
    autonomy_level_account: str | None  # L0..L5 or None when account has no row
    reg_bi_rationale: str
    generated_at: datetime


class ProposeArtifact(BaseModel):
    """Wrapper around the list of tickets PROPOSE produced (zero or more).

    A trigger-miss returns ``tickets=[]`` with a populated ``reason``
    string — NOT an error (acceptance criterion 1 from Task 4.4).
    """

    deployment_id: int
    firm_id: int
    tickets: list[TradeTicket]
    reason: str | None = None  # populated when tickets=[] to explain the miss
    generated_at: datetime
