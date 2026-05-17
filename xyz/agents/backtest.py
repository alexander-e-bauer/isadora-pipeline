"""BACKTEST subagent — orchestrates the day-by-day replay engine (Task 4.3).

This is the third engine-side subagent (after RESEARCH and AUTHOR).
Unlike RESEARCH and AUTHOR, BACKTEST does **not** call Claude.  The
"agent" naming exists only for spec/coordinator consistency — the
implementation is pure computation around ``xyz.backtest``.  Do not add
Claude calls here; do not import the Anthropic SDK.

Flow
----
1. Open a DB session via ``db_session_factory``.
2. Call ``xyz.backtest.run_backtest`` with the supplied DSL + date range.
3. Build a ``BacktestArtifact`` (metrics + hash + bookkeeping fields).
4. Emit a ``backtest.result`` audit event with the artifact as payload.
5. Return the artifact.

Transport contract
------------------
The engine returns the artifact in the response body — the caller (the
dashboard or a future orchestrator) is responsible for POSTing the
result into the server's ``backtest_results`` table via ``POST
/backtests``.  This keeps the engine read-only against the tenant DB
(matches the pattern from Tasks 4.1 / 4.2), and means the server stays
the source-of-truth for what backtest rows exist.

Determinism
-----------
``xyz.backtest.engine`` guarantees that two runs over the same inputs +
the same market data produce the same ``content_hash``.  The agent
itself does NOT add any non-deterministic data to the payload — the
``generated_at`` timestamp is stamped after the hash is computed and is
explicitly excluded from the hash's domain.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable, Generator

from sqlalchemy.orm import Session

from xyz.agents.schemas import BacktestArtifact, BacktestInput
from xyz.backtest.engine import run_backtest
from xyz.tenant.events import emit_event

logger = logging.getLogger(__name__)


class BacktestAgent:
    """Orchestrator around ``xyz.backtest.run_backtest``.

    No Claude calls.  No LLM dependency.  This is a deterministic
    function-with-side-effects (event emission + DB read).
    """

    def __init__(
        self,
        *,
        db_session_factory: Callable[[], Session],
    ) -> None:
        self._db_factory = db_session_factory

    @contextmanager
    def _db(self) -> Generator[Session, None, None]:
        """Yield a session; commit on success, rollback on error."""
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

    def run(self, input: BacktestInput) -> BacktestArtifact:
        """Run the replay, emit ``backtest.result``, return artifact.

        Raises
        ------
        ValueError
            If the DSL is not a supported template, dates are inverted,
            or no chain data exists for the underlying in the window.
        """
        logger.info(
            "BacktestAgent.run firm_id=%s strategy_id=%s v%s [%s, %s]",
            input.firm_id,
            input.strategy_id,
            input.strategy_version,
            input.start_date,
            input.end_date,
        )

        # 1. Run the deterministic engine.  The DB session here is used
        #    only for reading market data — no rows are written by the
        #    engine layer itself.  We open one session per call so the
        #    engine can lazily load chain rows.
        session = self._db_factory()
        try:
            result = run_backtest(
                dsl=input.dsl,
                start_date=input.start_date,
                end_date=input.end_date,
                db_session=session,
            )
        finally:
            session.close()

        # 2. Build the artifact.  ``generated_at`` is set AFTER the hash
        #    so it cannot influence the digest.
        artifact = BacktestArtifact(
            strategy_id=input.strategy_id,
            strategy_version=input.strategy_version,
            firm_id=input.firm_id,
            start_date=result.start_date,
            end_date=result.end_date,
            metrics=result.metrics,
            n_trades=len(result.trades),
            content_hash=result.content_hash,
            generated_at=datetime.now(timezone.utc),
        )

        # 3. Emit the audit event.  The event payload is the artifact
        #    minus the timestamp — same canonicalisation principle as
        #    AUTHOR / RESEARCH.  Pydantic's mode="json" handles date /
        #    datetime serialisation into JSON-native strings.
        payload = artifact.model_dump(mode="json")
        with self._db() as db:
            emit_event(
                db=db,
                kind="backtest.result",
                firm_id=input.firm_id,
                actor_user_id=input.actor_user_id,
                payload=payload,
            )

        logger.info(
            "backtest.result emitted strategy_id=%s v%s hash=%.8s…",
            input.strategy_id,
            input.strategy_version,
            artifact.content_hash,
        )
        return artifact
