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
