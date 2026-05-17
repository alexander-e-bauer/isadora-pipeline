"""PROPOSE subagent — evaluates a deployment's triggers and emits trade tickets.

This is the fourth engine-side subagent (after RESEARCH, AUTHOR, BACKTEST).
Like BACKTEST, PROPOSE does **not** call Claude — it is deterministic
computation over the tenant DB (deployment, strategy, positions,
autonomy matrix) and the engine's options market data
(``option_chains``).  The "agent" naming is for spec/coordinator
consistency only.

Flow (spec §5)
--------------
1. Look up the deployment.  If not ``ACTIVE`` → return an empty
   ``ProposeArtifact`` with ``reason="deployment_not_active"`` and emit
   no event.  Spec §9 is explicit that ONLY ACTIVE deployments propose.
2. Fetch the deployment's strategy and DSL at the frozen
   ``strategy_version``.  If the template is unsupported in v1 (anything
   other than ``covered_call`` per Task 4.3's v1 cut) → return empty.
3. Evaluate the covered-call trigger:
   - Account holds ≥ 100 long shares of the underlying.
   - No currently-open short call against the same underlying in this
     deployment (avoid stacking inadvertent shorts).
   - DSL ``regime_gates`` (if any) pass — v1 ignores them (always pass).
   - The ``option_chains`` snapshot contains at least one tradable
     candidate (positive bid).
4. Pick the target contract (same selection method as BACKTEST: nearest
   delta to ``delta_short_target``, ties broken by lowest strike, then
   nearest expiry, then ticker lex order, all within the DTE window).
5. Compute the limit price (short side: ``mid - max_slippage_pct * mid``).
6. Read the account's autonomy level for ``OPEN`` and the strategy's
   required level.  PROPOSE does NOT gate on this — COMPLIANCE (Task
   4.5) does.  We record both on the ticket for audit / dashboard.
7. Emit one ``ticket.proposed`` audit event per ticket.

Transport contract
------------------
The agent returns the ticket(s) in the response body.  The caller (the
dashboard or a future orchestrator) is responsible for POSTing each
ticket into the server's ``trades`` table via ``POST /trades``.  v1
does NOT wire an engine→server HTTP write — this avoids needing a
service-token bearer-auth on the server side and keeps the propose-write
surface aligned with existing JWT-authenticated routes.  Deferred to v1.5.

Determinism
-----------
For a given (deployment, strategy_version, account positions,
``option_chains`` snapshot), the agent produces the same ticket.  The
``generated_at`` timestamp is set after ticket construction and is NOT
hashed (matches BACKTEST's pattern).
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Generator

from sqlalchemy import select
from sqlalchemy.orm import Session

from xyz.agents.schemas import ProposeArtifact, ProposeInput, TradeTicket
from xyz.backtest.fill_model import DEFAULT_MAX_SLIPPAGE_PCT
from xyz.tenant.events import emit_event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers — small, isolated, easy to unit-test.
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float | None:
    """Coerce a Decimal / int / float / None to a plain float, preserving None."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _max_slippage_pct(dsl: dict) -> float:
    """Return the per-strategy slippage tolerance, defaulting to 0.005 (5bps)."""
    rb = dsl.get("risk_box") or {}
    em = rb.get("execution_microstructure") or {}
    val = em.get("max_slippage_tolerance_pct")
    if val is None:
        return DEFAULT_MAX_SLIPPAGE_PCT
    return float(val)


def _required_autonomy_level(dsl: dict, family: str) -> str:
    """Pull the required autonomy level for ``family`` from the DSL.

    Falls back to ``L2`` (the AUTHOR default for OPEN/HEDGE/PORTFOLIO) when
    the DSL omits ``autonomy_requirement`` entirely — this matches the
    AUTHOR system prompt's documented default and keeps PROPOSE working
    on DSLs drafted before ``autonomy_requirement`` was required.
    """
    requirement = (dsl.get("autonomy_requirement") or {}).get(family)
    if isinstance(requirement, str) and requirement.startswith("L"):
        return requirement
    return "L2"


def _account_autonomy_level(db: Session, account_id: int, family: str) -> str | None:
    """Read the account's autonomy level for ``family``, or None if unset."""
    from xyz.tenant.models import AccountAutonomy, ActionFamily

    row = db.scalar(
        select(AccountAutonomy).where(
            AccountAutonomy.account_id == account_id,
            AccountAutonomy.action_family == ActionFamily(family),
        )
    )
    if row is None:
        return None
    # row.level is the AutonomyLevel enum; its .value is "L0".."L5".
    return row.level.value if hasattr(row.level, "value") else str(row.level)


def _account_has_long_shares(db: Session, account_id: int, symbol: str, min_qty: int = 100) -> bool:
    """Return True if the account holds >= ``min_qty`` long shares of ``symbol``."""
    from xyz.tenant.models import AssetClass, Position

    total = Decimal("0")
    rows = db.scalars(
        select(Position).where(
            Position.account_id == account_id,
            Position.symbol == symbol,
            Position.asset_class == AssetClass.EQUITY,
        )
    ).all()
    for r in rows:
        total += r.qty
    return total >= Decimal(min_qty)


def _has_open_short_call(db: Session, account_id: int, symbol: str, today: datetime) -> bool:
    """Return True if there's an open short call against ``symbol`` not yet expired.

    Scoping policy: this check is **account-wide for the symbol**, not
    per-deployment.  Two deployments on the same underlying within the
    same account will not stack short calls — the second PROPOSE call
    will see the first deployment's short and block.  This is a
    deliberate v1 simplification (avoid accidental concentrated short
    exposure when an advisor activates a second strategy on the same
    ticker).  v1.5 may scope to ``deployment_id`` once stacking policy
    is formalised.

    Expiry boundary: ``r.expiry > today_date`` — a contract whose
    expiry is today is treated as expired (worthless after close), so
    the account is free to open a new call on the same day.  This
    avoids false "existing_open_short_call" blocks on monthly expiry
    Fridays.
    """
    from xyz.tenant.models import AssetClass, OptionType, Position

    today_date = today.date()
    rows = db.scalars(
        select(Position).where(
            Position.account_id == account_id,
            Position.symbol == symbol,
            Position.asset_class == AssetClass.OPTION,
            Position.option_type == OptionType.CALL,
        )
    ).all()
    for r in rows:
        if r.qty is None:
            continue
        if r.qty >= 0:
            continue
        # Same-day expiry: treat as already expired so a fresh open is
        # not blocked on monthly expiration Fridays.
        if r.expiry is None or r.expiry <= today_date:
            continue
        return True
    return False


def _extract_symbol(dsl: dict) -> str | None:
    """Return the first ticker in ``selection.universe`` if present."""
    universe = (dsl.get("selection") or {}).get("universe") or []
    if not universe:
        return None
    return str(universe[0])


def _pick_target_contract(
    db: Session,
    symbol: str,
    *,
    delta_target: float,
    dte_min: int,
    dte_max: int,
    asof: datetime,
) -> Any | None:
    """Return the best ``option_chains`` row for ``symbol`` at ``asof``.

    Selection method (matches BACKTEST's tiebreak order):
      1. CALL only.
      2. Delta present, in (0, 1) — discard zero / ITM legs.
      3. DTE in [dte_min, dte_max].
      4. Positive bid (tradable).
      5. Sort by (|delta - target|, strike, expiry, contract_ticker).

    The query uses the most recent ``asof_at`` for the underlying when
    multiple snapshots exist (cheap "latest snapshot" semantics).  If
    no chain rows exist for the symbol, returns None.
    """
    from xyz.polygon_service.models import OptionChains

    # Latest snapshot timestamp for this underlying.
    latest_asof = db.scalar(
        select(OptionChains.asof_at)
        .where(OptionChains.underlying == symbol)
        .order_by(OptionChains.asof_at.desc())
        .limit(1)
    )
    if latest_asof is None:
        return None

    rows = db.scalars(
        select(OptionChains).where(
            OptionChains.underlying == symbol,
            OptionChains.asof_at == latest_asof,
            OptionChains.option_type == "CALL",
        )
    ).all()

    candidates: list[tuple[float, Any]] = []  # (delta_distance, row)
    asof_date = asof.date()
    for r in rows:
        delta = _safe_float(r.delta)
        if delta is None or delta <= 0 or delta >= 1.0:
            continue
        if r.expiry is None:
            continue
        dte = (r.expiry - asof_date).days
        if dte < dte_min or dte > dte_max:
            continue
        bid = _safe_float(r.bid) or 0.0
        if bid <= 0:
            continue
        distance = abs(delta - delta_target)
        candidates.append((distance, r))

    if not candidates:
        return None

    candidates.sort(
        key=lambda kv: (
            round(kv[0], 6),
            _safe_float(kv[1].strike) or 0.0,
            kv[1].expiry,
            kv[1].contract_ticker,
        )
    )
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ProposeAgent:
    """Orchestrator that evaluates a deployment and emits ticket(s).

    No Claude calls.  No LLM dependency.  Pure DB-driven computation +
    one audit event per ticket emitted.
    """

    def __init__(
        self,
        *,
        db_session_factory: Callable[[], Session],
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._db_factory = db_session_factory
        # Injectable clock so tests can pin "now" for deterministic DTE math.
        self._now = now or (lambda: datetime.now(timezone.utc))

    @contextmanager
    def _db(self) -> Generator[Session, None, None]:
        """Yield a session, committing on success, rolling back on error."""
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

    def run(self, input: ProposeInput) -> ProposeArtifact:
        """Evaluate the deployment, build ticket(s), emit events, return artifact.

        Returns ``ProposeArtifact(tickets=[], reason="...")`` on any
        precondition miss — never raises in that case.  ``ValueError`` is
        only raised on hard data-shape errors that indicate a bug
        upstream (e.g. deployment exists but its strategy_id is dangling).
        """
        logger.info(
            "ProposeAgent.run firm_id=%s deployment_id=%s",
            input.firm_id,
            input.deployment_id,
        )

        now = self._now()

        # All reads happen inside one session.  Writes (event emission)
        # happen inside the SAME session so the event chain includes the
        # ticket payload in its hash.
        with self._db() as db:
            artifact = self._evaluate(db, input, now)
        return artifact

    # ------------------------------------------------------------------
    # Internal evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        db: Session,
        input: ProposeInput,
        now: datetime,
    ) -> ProposeArtifact:
        """Internal: does the work inside a single open session."""
        # Lazy-import to keep module load light.
        from xyz.tenant.models import (
            Account,
            Client,
            Deployment,
            DeploymentState,
            Strategy,
        )

        # ------------------------------------------------------------------
        # 1. Look up the deployment and tenant-scope-check via account → client.
        # ------------------------------------------------------------------
        deployment = db.scalar(
            select(Deployment)
            .join(Account, Deployment.account_id == Account.id)
            .join(Client, Account.client_id == Client.id)
            .where(
                Deployment.id == input.deployment_id,
                Client.firm_id == input.firm_id,
            )
        )
        if deployment is None:
            return self._empty(
                input,
                now=now,
                reason="deployment_not_found",
            )

        if deployment.state != DeploymentState.ACTIVE:
            return self._empty(
                input,
                now=now,
                reason="deployment_not_active",
            )

        # ------------------------------------------------------------------
        # 2. Look up the strategy + DSL.
        # ------------------------------------------------------------------
        strategy = db.scalar(
            select(Strategy).where(
                Strategy.id == deployment.strategy_id,
                Strategy.firm_id == input.firm_id,
            )
        )
        if strategy is None:
            return self._empty(
                input,
                now=now,
                reason="strategy_not_found",
            )

        dsl = strategy.dsl_json or {}
        template = dsl.get("template")
        if template != "covered_call":
            return self._empty(
                input,
                now=now,
                reason="template_not_supported_in_v1",
            )

        symbol = _extract_symbol(dsl)
        if symbol is None:
            return self._empty(
                input,
                now=now,
                reason="dsl_missing_underlying_symbol",
            )

        # ------------------------------------------------------------------
        # 3. Trigger preconditions on account state.
        # ------------------------------------------------------------------
        if not _account_has_long_shares(db, deployment.account_id, symbol):
            return self._empty(
                input,
                now=now,
                reason="insufficient_long_shares",
            )

        if _has_open_short_call(db, deployment.account_id, symbol, now):
            return self._empty(
                input,
                now=now,
                reason="existing_open_short_call",
            )

        # ------------------------------------------------------------------
        # 4. Pull target contract from option_chains.
        # ------------------------------------------------------------------
        action = dsl.get("action") or {}
        delta_target = float(action.get("delta_short_target", 0.30))
        dte_min = int(action.get("dte_min", 30))
        dte_max = int(action.get("dte_max", 45))

        contract = _pick_target_contract(
            db,
            symbol,
            delta_target=delta_target,
            dte_min=dte_min,
            dte_max=dte_max,
            asof=now,
        )
        if contract is None:
            return self._empty(
                input,
                now=now,
                reason="no_tradable_contract",
            )

        # ------------------------------------------------------------------
        # 5. Limit price (short side: shave epsilon below mid).
        # ------------------------------------------------------------------
        bid = _safe_float(contract.bid) or 0.0
        ask = _safe_float(contract.ask) or 0.0
        mid = _safe_float(contract.mid)
        if mid is None or mid <= 0:
            # Recover mid from bid/ask when the snapshot only stored
            # quotes.  If both are zero the candidate filter above would
            # have skipped this row, so this path is rare.
            mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else max(bid, ask)
        if mid <= 0:
            return self._empty(
                input,
                now=now,
                reason="no_tradable_contract",  # degenerate quote
            )

        slippage_pct = _max_slippage_pct(dsl)
        epsilon = slippage_pct * mid
        limit_price = round(mid - epsilon, 4)
        # Floor at zero — defensive, even though epsilon < mid by construction.
        if limit_price < 0:
            limit_price = 0.0

        # ------------------------------------------------------------------
        # 6. Autonomy levels.  PROPOSE does NOT gate — COMPLIANCE does.
        # ------------------------------------------------------------------
        required_level = _required_autonomy_level(dsl, "OPEN")
        account_level = _account_autonomy_level(
            db, deployment.account_id, "OPEN"
        )

        # ------------------------------------------------------------------
        # 7. Build the ticket.
        # ------------------------------------------------------------------
        strike = _safe_float(contract.strike)
        delta = _safe_float(contract.delta)
        expiry_str = contract.expiry.isoformat() if contract.expiry else None
        strategy_name = strategy.name
        rationale = (
            f"Covered call against existing long position in {symbol}. "
            f"Strategy '{strategy_name}' version {deployment.strategy_version}. "
            f"Target delta {delta_target:.2f}, expiry {expiry_str}, "
            f"strike {strike}."
        )

        order_ticket_json: dict = {
            "symbol": symbol,
            "contract_ticker": contract.contract_ticker,
            "strike": strike,
            "expiry": expiry_str,
            "option_type": "CALL",
            "side": "sell_to_open",
            "qty": 1,  # one contract = 100-share cover; v1 sells one lot
            "delta": delta,
            "delta_target": delta_target,
            "dte_window": [dte_min, dte_max],
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "limit_price": limit_price,
            "limit_side": "sell",
            "max_slippage_tolerance_pct": slippage_pct,
            "strategy_id": strategy.id,
            "strategy_version": deployment.strategy_version,
            "autonomy_level_required": required_level,
            "autonomy_level_account": account_level,
            "reg_bi_rationale": rationale,
        }

        ticket = TradeTicket(
            account_id=deployment.account_id,
            deployment_id=deployment.id,
            leaf_action="OPEN_NEW",
            action_family="OPEN",
            # Spec §6: OPEN_NEW adds delta-down exposure (short call) →
            # RISK_INCREASING for the strategic action taxonomy v1 cell.
            risk_class="RISK_INCREASING",
            order_ticket_json=order_ticket_json,
            autonomy_level_required=required_level,
            autonomy_level_account=account_level,
            reg_bi_rationale=rationale,
            generated_at=now,
        )

        # ------------------------------------------------------------------
        # 8. Emit the audit event.  Payload mirrors the ticket but is
        #    serialised to JSON-native via Pydantic mode="json".
        # ------------------------------------------------------------------
        emit_event(
            db=db,
            kind="ticket.proposed",
            firm_id=input.firm_id,
            actor_user_id=input.actor_user_id,
            payload=ticket.model_dump(mode="json"),
        )

        logger.info(
            "ticket.proposed emitted deployment_id=%s account_id=%s "
            "symbol=%s strike=%s expiry=%s",
            input.deployment_id,
            deployment.account_id,
            symbol,
            strike,
            expiry_str,
        )

        return ProposeArtifact(
            deployment_id=input.deployment_id,
            firm_id=input.firm_id,
            tickets=[ticket],
            reason=None,
            generated_at=now,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty(
        self,
        input: ProposeInput,
        *,
        now: datetime,
        reason: str,
    ) -> ProposeArtifact:
        """Return an empty artifact with the given miss reason.

        Per acceptance criterion 1, a trigger miss is NOT an error — the
        caller gets a 200 response with an explanation, and NO event is
        emitted on the engine chain.  This keeps the audit log clean of
        "no-op" rows that would dominate the chain at scale.
        """
        logger.info(
            "ProposeAgent miss firm_id=%s deployment_id=%s reason=%s",
            input.firm_id,
            input.deployment_id,
            reason,
        )
        return ProposeArtifact(
            deployment_id=input.deployment_id,
            firm_id=input.firm_id,
            tickets=[],
            reason=reason,
            generated_at=now,
        )
