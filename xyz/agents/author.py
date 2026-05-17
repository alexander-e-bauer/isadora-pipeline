"""AUTHOR subagent — drafts a declarative Strategy DSL from an advisor brief.

Flow
----
1. Receive an advisor brief (free-form natural language) + firm context.
2. Build a prompt with a cached system block (role + DSL schema + template
   guidance) and a user message containing the brief.
3. Call Claude; parse the JSON response into an AuthorArtifact.
4. Validate the produced DSL against the JSON-Schema; abort if invalid.
5. Compute a content hash, emit a ``strategy.draft`` event, return artifact.

Design constraints (v1)
-----------------------
- Only three templates: covered_call (CC), cash_secured_put (CSP), collar.
- No tool-use; Claude returns a single JSON object.
- The validator is the authoritative gate — if Claude produces an invalid
  DSL the agent raises ``ValueError`` and refuses to emit the event so the
  audit log never contains a malformed draft.

The system prompt deliberately inlines the DSL JSON-Schema (compact form)
so Claude can self-validate before responding.  The schema is large but
prompt-cached via ``CachedSystemBlock`` — the cost is paid once per cold
cache, then amortised across every brief that lands during the 5-minute
TTL window.
"""
from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable, Generator

from sqlalchemy.orm import Session

from xyz.agents.lib.anthropic_client import AnthropicClient, CachedSystemBlock
from xyz.agents.schemas import AuthorArtifact, AuthorInput
from xyz.dsl.schema import DSL_SCHEMA
from xyz.dsl.validate import validate_dsl
from xyz.tenant.events import emit_event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt (cached)
# ---------------------------------------------------------------------------

_DSL_SCHEMA_JSON = json.dumps(DSL_SCHEMA, separators=(",", ":"))

_SYSTEM_PROMPT = f"""\
You are AUTHOR, a strategy-drafting subagent for an options-overlay platform
serving registered investment advisors (RIAs).

## Your role
Convert an advisor brief (plain English) into a *declarative* Strategy DSL
JSON document conforming to the schema below.  You also produce a short
Reg-BI rationale stub the advisor will review before approving.

## v1 template universe
You may draft one of exactly three templates:
- "covered_call"      — short OTM call against a long underlying.
- "cash_secured_put"  — short OTM put fully cash-collateralized.
- "collar"            — long underlying + short OTM call + long OTM put.

If the brief does not clearly map to one of these, choose the closest and
flag the assumption in `rationale`.  Do NOT invent new templates.

## Sensible defaults
Apply these defaults unless the brief explicitly overrides them:
- 30Δ short strike target (delta_short ≈ 0.30).
- 30–45 DTE expiration window.
- Close at 50% profit (max_profit_pct_close = 0.50).
- Earnings blackout: 7 calendar days each side (risk_box.time_windows.earnings_blackout_days = 7).
- IRA-suitable accounts: prefer covered_call or cash_secured_put.  Collars
  are permitted (defensive) but flag the lower yield in rationale.

## Output format
Respond with a SINGLE JSON object (no markdown fence) with these keys:
{{
  "template": "covered_call" | "cash_secured_put" | "collar",
  "dsl":     <a JSON document conforming to the DSL schema below>,
  "rationale": <string — 1-2 paragraph Reg-BI rationale stub>
}}

The `dsl` object MUST satisfy the following JSON-Schema:
```json
{_DSL_SCHEMA_JSON}
```

Required top-level DSL keys (do not omit any):
- kind: must be the string "declarative".  Never emit "scripted".
- name, version (integer, start at 1), author_user_id, firm_id.
- selection, trigger, action, exit  — these are free-form sub-documents but
  MUST be present (even as an empty {{}} when nothing applies).
- risk_box  — populate the relevant nested limits.  Always include
  `time_windows.earnings_blackout_days` when relevant.
- autonomy_requirement  — map each action family (OPEN, CLOSE, ROLL, HEDGE,
  PORTFOLIO, EVENT, STATE) to an autonomy level "L0".."L5".  Default to "L2"
  for OPEN/HEDGE/PORTFOLIO, "L3" for CLOSE/ROLL, "L4" for EVENT/STATE.

## Constraints
- Always set `kind` to "declarative".
- Always include all five sections (selection, trigger, action, exit, risk_box).
- For IRA suitability hints, never propose strategies that can produce
  short uncovered call exposure.
- Cite none of your sources — AUTHOR does not need citations (that's
  RESEARCH's job).  Your output is a *proposal*, not an analysis.
- Do not include any text outside the JSON object in your response.
"""

_SYSTEM_BLOCKS = [CachedSystemBlock(text=_SYSTEM_PROMPT)]


# ---------------------------------------------------------------------------
# Hash helper
# ---------------------------------------------------------------------------

def _hash_artifact(artifact_dict: dict) -> str:
    """SHA-256 of the canonical JSON of the artifact dict.

    ``generated_at`` and ``content_hash`` are stripped from the hashed view
    so the digest is purely content-driven.  Matches the pattern used by
    RESEARCH for cross-agent symmetry.
    """
    stable = {k: v for k, v in artifact_dict.items() if k not in ("generated_at", "content_hash")}
    canonical = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AuthorAgent:
    """AUTHOR subagent — drafts a declarative DSL from an advisor brief."""

    def __init__(
        self,
        *,
        anthropic_client: AnthropicClient,
        db_session_factory: Callable[[], Session],
    ) -> None:
        self._llm = anthropic_client
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

    def run(self, input: AuthorInput) -> AuthorArtifact:
        """Call Claude, validate the produced DSL, emit event, return artifact.

        Raises
        ------
        json.JSONDecodeError
            If Claude returns non-JSON content.
        KeyError
            If the parsed response is missing ``template`` / ``dsl`` / ``rationale``.
        ValueError
            If the produced DSL fails JSON-Schema validation.  No event is
            emitted in this case so the audit log stays clean of malformed
            drafts.
        """
        logger.info(
            "AuthorAgent.run firm_id=%s brief=%.60s…", input.firm_id, input.brief
        )

        # 1. Call Claude.
        response_dict = self._call_claude(input)

        # 2. Stamp the DSL with firm/author identity even if Claude omitted
        #    them — these are the authoritative values, not model-generated.
        dsl = response_dict["dsl"]
        dsl.setdefault("kind", "declarative")
        dsl["author_user_id"] = input.actor_user_id if input.actor_user_id is not None else dsl.get("author_user_id", 0)
        dsl["firm_id"] = input.firm_id
        dsl.setdefault("version", 1)

        # 3. Validate the DSL.  We treat validation failure as a hard error
        #    so a malformed draft never lands in the event log.
        valid, errors = validate_dsl(dsl)
        if not valid:
            raise ValueError(f"AUTHOR produced invalid DSL: {errors}")

        # 4. Build the artifact + content hash.
        artifact_dict = {
            "template": response_dict["template"],
            "dsl": dsl,
            "rationale": response_dict["rationale"],
        }
        content_hash = _hash_artifact(artifact_dict)
        artifact = AuthorArtifact(
            **artifact_dict,
            generated_at=datetime.now(timezone.utc),
            content_hash=content_hash,
        )

        # 5. Emit event.
        payload = artifact.model_dump(mode="json")
        with self._db() as db:
            emit_event(
                db=db,
                kind="strategy.draft",
                firm_id=input.firm_id,
                actor_user_id=input.actor_user_id,
                payload=payload,
            )

        logger.info(
            "strategy.draft emitted template=%s firm_id=%s hash=%.8s…",
            artifact.template,
            input.firm_id,
            content_hash,
        )
        return artifact

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_claude(self, input: AuthorInput) -> dict:
        """Build the user message, call Claude, parse the response.

        Raises
        ------
        json.JSONDecodeError
            If the response is not valid JSON.
        KeyError
            If the response is missing one of ``template`` / ``dsl`` / ``rationale``.
        """
        user_message = json.dumps(
            {
                "brief": input.brief,
                "firm_id": input.firm_id,
                "actor_user_id": input.actor_user_id,
                "target_account_ids": input.target_account_ids,
                "instructions": (
                    "Draft a covered_call, cash_secured_put, or collar template "
                    "for this brief.  Return only the JSON object described in "
                    "the system prompt — no surrounding text, no markdown fence."
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
        parsed = json.loads(raw_text)

        # Surface a clear KeyError before the route maps it to a 502 — the
        # caller will get the missing-field name in the error detail.
        for required in ("template", "dsl", "rationale"):
            if required not in parsed:
                raise KeyError(required)

        return parsed
