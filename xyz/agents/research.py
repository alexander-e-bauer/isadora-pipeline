"""RESEARCH subagent.

Orchestrates context gathering → Claude LLM call → structured artifact emission.

Flow
----
1. Gather raw context from Polygon (options chain / IV) and the engine DB
   (Document / MarketEmb* rows tied to the ticker).
2. Build a prompt with a cached system block (role + schema + citation rules).
3. Call Claude; parse the JSON response into a ResearchArtifact.
4. Compute a content hash and emit a ``research.artifact`` event to the
   shared event log.
5. Return the ResearchArtifact to the caller.

Design constraints (v1)
-----------------------
- No tool-use — context is pre-gathered and handed to Claude as JSON.
- No streaming — one-shot request/response.
- No new ORM tables — the artifact lives in the events table as payload.
- No hallucinated citations — if a fact has no source, it is omitted.
"""
from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Generator

from sqlalchemy.orm import Session

from xyz.agents.lib.anthropic_client import AnthropicClient, CachedSystemBlock
from xyz.agents.lib.citation import Citation
from xyz.agents.schemas import ResearchArtifact, ResearchInput, ResearchSection
from xyz.tenant.events import emit_event

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (cached)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a financial-research analyst supporting a registered investment advisor (RIA).

## Your role
Produce a grounded, factual research summary on a ticker, account, or brief supplied by
the advisor. You surface context only — you do NOT generate price targets, trade
recommendations, or hedge ratios. That is the AUTHOR agent's job.

## Output format
Respond with a single JSON object that exactly matches this schema (no markdown fence):

{
  "symbol": <string or null>,
  "summary": <string — 2-4 sentence executive summary>,
  "iv_regime": {
    "title": <string>,
    "body": <string>,
    "citations": [{"kind": "db_row"|"polygon"|"url", "source": <string>, "excerpt": <string or null>}]
  } or null,
  "earnings_calendar": {
    "title": <string>,
    "body": <string>,
    "citations": [...]
  } or null,
  "news_headlines": {
    "title": <string>,
    "body": <string>,
    "citations": [{"kind": "db_row"|"polygon"|"url", "source": <string>, "excerpt": <string or null>}]
  } or null,
  "peer_comparison": {
    "title": <string>,
    "body": <string>,
    "citations": [...]
  } or null
}

## Citation discipline
- Every factual claim MUST cite a source from the provided context using the citations array.
- Use kind="db_row" for rows from the engine database (e.g. "documents:42").
- Use kind="polygon" for Polygon snapshot data (e.g. "snapshot:AAPL:atm_iv").
- Use kind="url" only if a URL is explicitly provided in the context.
- If you cannot cite a claim, omit it entirely. Never fabricate sources.

## Constraints
- Do not generate price targets, trade recommendations, or hedge ratios.
- Do not speculate about future earnings dates unless a date is provided in the context.
- Do not include content outside the JSON object in your response.
- If a section has no available context, set it to null in the JSON.
"""

_SYSTEM_BLOCKS = [CachedSystemBlock(text=_SYSTEM_PROMPT)]


# ---------------------------------------------------------------------------
# Hash helper
# ---------------------------------------------------------------------------

def _hash_artifact(artifact_dict: dict) -> str:
    """SHA-256 of the canonical JSON of artifact_dict, excluding generated_at and content_hash.

    Stripping those two fields ensures the hash is determined by content only
    and is reproducible across calls.
    """
    stable = {k: v for k, v in artifact_dict.items() if k not in ("generated_at", "content_hash")}
    canonical = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ResearchAgent:
    """RESEARCH subagent — gathers context and calls Claude to produce a ResearchArtifact."""

    def __init__(
        self,
        *,
        anthropic_client: AnthropicClient,
        options_client: Any,   # OptionsClient — typed as Any to avoid hard dep in tests
        db_session_factory: Callable[[], Session],
    ) -> None:
        self._llm = anthropic_client
        self._options = options_client
        self._db_factory = db_session_factory

    @contextmanager
    def _db(self) -> Generator[Session, None, None]:
        """Yield a DB session, committing on success and rolling back on error."""
        session = self._db_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, input: ResearchInput) -> ResearchArtifact:
        """Gather context, call Claude, emit event, return artifact.

        Parameters
        ----------
        input:
            ResearchInput validated at the HTTP layer.

        Returns
        -------
        ResearchArtifact
            The fully populated, hashed research artifact.

        Raises
        ------
        ValueError
            If neither symbol nor brief is provided (should be caught at route).
        json.JSONDecodeError
            If Claude returns malformed JSON (do not attempt to repair).
        """
        if not input.symbol and not input.brief:
            raise ValueError("ResearchInput must have at least one of: symbol, brief")

        logger.info("ResearchAgent.run target=%s firm_id=%s", input.primary_target(), input.firm_id)

        # 1. Gather context
        ctx = self._gather_context(input)

        # 2. Call Claude
        artifact_dict = self._call_claude(input, ctx)

        # 3. Build artifact + hash
        content_hash = _hash_artifact(artifact_dict)
        artifact = ResearchArtifact(
            **artifact_dict,
            generated_at=datetime.now(timezone.utc),
            content_hash=content_hash,
        )

        # 4. Emit event
        payload = artifact.model_dump(mode="json")
        with self._db() as db:
            emit_event(
                db=db,
                kind="research.artifact",
                firm_id=input.firm_id,
                actor_user_id=input.actor_user_id,
                payload=payload,
            )

        logger.info(
            "research.artifact emitted target=%s hash=%.8s…",
            input.primary_target(),
            content_hash,
        )
        return artifact

    # ------------------------------------------------------------------
    # Context gathering
    # ------------------------------------------------------------------

    def _gather_context(self, input: ResearchInput) -> dict:
        """Collect raw data from Polygon and the engine DB.

        Returns a dict that is serialised and handed to Claude as the user
        message content.  All data must be citeable — each piece of context
        is labelled with its source so Claude can produce proper citations.
        """
        ctx: dict[str, Any] = {
            "symbol": input.symbol,
            "brief": input.brief,
            "account_id": input.account_id,
        }

        if input.symbol:
            ctx["iv_context"] = self._gather_iv_context(input.symbol)
            ctx["earnings_context"] = {
                "note": "Earnings calendar unavailable in v1.",
                "citation": None,
            }
            ctx["news_context"] = self._gather_news_context(input.symbol)

        return ctx

    def _gather_iv_from_surface(self, symbol: str) -> dict | None:
        """Query the engine DB's ``option_iv_surface`` for the most recent
        IV samples for ``symbol``.

        Returns a context dict on hit, ``None`` on miss/error so the caller
        can fall back to the Polygon snapshot path.  Errors (DB unavailable,
        table missing in non-prod) are swallowed and logged — this is a best-
        effort primary lookup, never a fatal one.
        """
        try:
            from xyz.finazon_service.sql_service import SessionLocal
            from xyz.polygon_service.models import OptionIvSurface

            session = SessionLocal()
            try:
                latest_date = session.query(OptionIvSurface.asof_date).filter(
                    OptionIvSurface.underlying == symbol
                ).order_by(OptionIvSurface.asof_date.desc()).first()
                if latest_date is None:
                    return None
                asof_date = latest_date[0]

                rows = (
                    session.query(OptionIvSurface)
                    .filter(
                        OptionIvSurface.underlying == symbol,
                        OptionIvSurface.asof_date == asof_date,
                    )
                    .all()
                )
                if not rows:
                    return None

                ivs = [float(r.implied_vol) for r in rows if r.implied_vol is not None]
                avg_iv = sum(ivs) / len(ivs) if ivs else None

                citation = Citation(
                    kind="db_row",
                    source=f"option_iv_surface:{symbol}:{asof_date.isoformat()}",
                    excerpt=f"rows={len(rows)} avg_iv={avg_iv}",
                )
                return {
                    "underlying_price": None,
                    "atm_iv_estimate": avg_iv,
                    "contract_count": len(rows),
                    "asof": asof_date.isoformat(),
                    "citation": citation.to_dict(),
                    "note": (
                        f"IV regime from option_iv_surface ({len(rows)} rows "
                        f"as of {asof_date.isoformat()})."
                    ),
                }
            finally:
                session.close()
        except Exception as exc:
            logger.warning("option_iv_surface lookup unavailable for %s: %s", symbol, exc)
            return None

    def _gather_iv_context(self, symbol: str) -> dict:
        """Build the IV regime context.

        Primary source: the ``option_iv_surface`` table in the engine DB —
        cheaper, quota-free, and the spec'd source-of-truth for IV history.
        Fallback: a live Polygon chain snapshot when no surface rows exist
        for the symbol yet (cold start / unscheduled ticker).
        On any error (network, rate-limit, DB unreachable) returns a stub
        noting the unavailability — the agent must not hallucinate IV data.
        """
        # Primary source: option_iv_surface (engine DB).
        db_iv = self._gather_iv_from_surface(symbol)
        if db_iv is not None:
            return db_iv

        # Fallback: live Polygon snapshot.
        try:
            snapshot = self._options.get_chain(symbol)
            underlying_price = snapshot.underlying_price or 0.0

            # Find ATM contracts (strike within 2% of underlying)
            atm_ivs: list[float] = []
            for c in snapshot.contracts:
                if underlying_price and abs(c.strike - underlying_price) / underlying_price < 0.02:
                    if c.implied_vol is not None:
                        atm_ivs.append(c.implied_vol)

            if not atm_ivs:
                # Fallback: use all available IVs
                atm_ivs = [c.implied_vol for c in snapshot.contracts if c.implied_vol is not None]

            avg_iv = sum(atm_ivs) / len(atm_ivs) if atm_ivs else None

            citation = Citation(
                kind="polygon",
                source=f"snapshot:{symbol}:{snapshot.asof_at.isoformat()}",
                excerpt=f"underlying_price={underlying_price}, atm_iv_sample_count={len(atm_ivs)}",
            )
            return {
                "underlying_price": underlying_price,
                "atm_iv_estimate": avg_iv,
                "contract_count": len(snapshot.contracts),
                "asof": snapshot.asof_at.isoformat(),
                "citation": citation.to_dict(),
                "note": (
                    "IV rank vs 252-day range unavailable in v1 "
                    "(historical IV surface not yet populated)."
                ),
            }
        except Exception as exc:
            logger.warning("IV context unavailable for %s: %s", symbol, exc)
            return {
                "underlying_price": None,
                "atm_iv_estimate": None,
                "contract_count": 0,
                "asof": None,
                "citation": None,
                "note": f"IV context unavailable: {exc}",
            }

    def _gather_news_context(self, symbol: str) -> list[dict]:
        """Query the engine DB for the 5 most recent documents linked to a ticker.

        Returns a list of dicts with title, source citation, and row id.
        If the table is empty or missing, returns an empty list — Claude will
        set news_headlines to null.
        """
        try:
            # We import here to avoid importing sql_service at module load (tests mock this)
            from xyz.finazon_service.sql_service import SessionLocal, Ticker, Document

            session = SessionLocal()
            try:
                ticker_row = session.query(Ticker).filter(Ticker.symbol == symbol).first()
                if not ticker_row:
                    return []

                docs = (
                    session.query(Document)
                    .filter(Document.tickers.any(id=ticker_row.id))
                    .order_by(Document.id.desc())
                    .limit(5)
                    .all()
                )
                return [
                    {
                        "title": getattr(doc, "title", str(doc.id)),
                        "citation": Citation(
                            kind="db_row",
                            source=f"documents:{doc.id}",
                            excerpt=getattr(doc, "title", None),
                        ).to_dict(),
                    }
                    for doc in docs
                ]
            finally:
                session.close()
        except Exception as exc:
            logger.warning("News context unavailable for %s: %s", symbol, exc)
            return []

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_claude(self, input: ResearchInput, ctx: dict) -> dict:
        """Build prompt, call Claude, parse and return the artifact dict.

        Raises
        ------
        json.JSONDecodeError
            If Claude's response is not valid JSON.
        KeyError / ValueError
            If parsed JSON is missing required fields.
        """
        user_message = json.dumps(
            {
                "request": {
                    "symbol": input.symbol,
                    "brief": input.brief,
                    "account_id": input.account_id,
                },
                "context": ctx,
                "instructions": (
                    "Produce a research artifact for the request above using the "
                    "provided context. Cite every factual claim. Return only the "
                    "JSON object described in the system prompt — no surrounding text."
                ),
            },
            indent=2,
            default=str,
        )

        response = self._llm.complete(
            system=_SYSTEM_BLOCKS,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=4096,
            temperature=0.2,
        )

        raw_text = response.content[0].text
        artifact_dict = json.loads(raw_text)

        # Normalise optional fields that Claude might omit entirely
        for field in ("iv_regime", "earnings_calendar", "news_headlines", "peer_comparison"):
            artifact_dict.setdefault(field, None)

        return artifact_dict
