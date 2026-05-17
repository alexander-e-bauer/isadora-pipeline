"""Pydantic models shared by all engine-side subagents."""
from __future__ import annotations

from datetime import datetime
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
