"""Pydantic models shared by all engine-side subagents."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


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
