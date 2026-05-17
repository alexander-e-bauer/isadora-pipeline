"""Day-by-day replay loop for the v1 BACKTEST subagent.

Scope (per plan §495)
---------------------
* Only the ``covered_call`` template.
* Daily-bar fills only.  No intraday microstructure.
* No Greek recomputation.  No IV-surface modeling.
* No tax-lot accounting.  Every trade is flat P&L.
* Assignment proxy: a contract that ends EOD ITM on its expiration date.

Inputs
------
The engine reads from one and only one tenant-DB table:
    ``option_historical_eod`` — daily OHLCV + Greeks + IV per contract,
    queried per underlying + date range.

The engine NEVER reads positions / trades / firm tables.  It is a pure
hypothetical replay over a DSL + market data.  Resulting events emitted
upstream by ``BacktestAgent`` are firm-scoped audit rows, but the math
here is firm-agnostic.

Replay loop
-----------
For each trading day ``d`` in ``[start_date, end_date]`` sorted ascending:
    1. Mark open positions to market (EOD close on the underlying for
       shares; EOD chain close for short calls).
    2. Evaluate ``dsl.exit`` rules — close in-the-money positions whose
       ``max_profit_pct_close`` threshold is hit, or close when the
       expiration date arrives.
    3. Skip adjustment / roll logic (deferred to v1.5).
    4. Evaluate the OPEN_NEW trigger — pick a new contract per the
       ``selection`` filters + ``action.delta_short_target`` /
       ``dte_min`` / ``dte_max``.
    5. Mark NAV (cash + share value − short-call liability).

NAV is normalised so ``NAV[0] = 1.0`` — the metrics layer is unit-agnostic.

Determinism
-----------
- ``[start_date, end_date]`` iteration is the sorted unique set of dates
  present in the underlying chain rows.
- When multiple chain rows qualify for an open, the row with the lowest
  (strike, expiry, contract_ticker) tuple wins — no random tie-break.
- All floats are serialised at 6-decimal precision before hashing.
- ``content_hash`` is sha256 over a canonical JSON view of
  ``{dsl, start_date, end_date, metrics}`` — timestamps are NOT included.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Sequence

from sqlalchemy import and_
from sqlalchemy.orm import Session

from xyz.backtest.fill_model import DEFAULT_MAX_SLIPPAGE_PCT, compute_fill
from xyz.backtest.metrics import (
    avg_pnl,
    cagr,
    max_drawdown,
    sharpe,
    total_return,
    win_rate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """One closed (or auto-expired) short-call trade.

    Fields are all native types so the dataclass serialises cleanly
    into the artifact JSON.
    """

    contract_ticker: str
    open_date: date
    close_date: date
    strike: float
    expiry: date
    opened_at_price: float
    closed_at_price: float
    pnl: float
    close_reason: str  # "max_profit", "expiration_otm", "expiration_itm"


@dataclass
class BacktestResult:
    """Output of ``run_backtest``.

    ``metrics`` holds the aggregate numbers persisted into
    ``backtest_results.metrics_json``.  ``content_hash`` is the audit
    digest that proves determinism — two runs over the same inputs MUST
    produce the same hash.
    """

    start_date: date
    end_date: date
    metrics: dict
    trades: list[TradeRecord]
    nav_series: list[float]
    content_hash: str
    dsl_snapshot: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float:
    """Coerce a Decimal / int / float / None to a plain float (None → 0.0)."""
    if value is None:
        return 0.0
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _quantize_floats(obj: Any) -> Any:
    """Recursively walk ``obj`` rendering every float as a 6-decimal string.

    This is the canonicalisation step the content hash relies on.  All
    floats are pre-rendered to strings so the hash is byte-stable
    independent of the host's float-print rounding mode.
    """
    if isinstance(obj, float):
        return f"{obj:.6f}"
    if isinstance(obj, dict):
        return {k: _quantize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_quantize_floats(v) for v in obj]
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return obj


def _content_hash(*, dsl: dict, start: date, end: date, metrics: dict) -> str:
    """Stable sha256 over inputs+metrics.  Excludes any timestamps.

    The DSL is quantized alongside ``metrics`` because callers may submit
    float-valued DSL keys (``delta_short_target: 0.30`` etc.).  Without
    quantization those values would be serialized by ``json.dumps`` using
    Python's float repr, which can differ across platforms/builds for
    edge cases.  Pre-rendering to 6-decimal strings keeps the hash
    byte-stable for the same logical input.
    """
    canonical_view = {
        "dsl": _quantize_floats(dsl),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "metrics": _quantize_floats(metrics),
    }
    canonical = json.dumps(
        canonical_view,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _extract_symbol(dsl: dict) -> str:
    """Return the first ticker in ``selection.universe`` (covered_call has 1)."""
    universe = (dsl.get("selection") or {}).get("universe") or []
    if not universe:
        raise ValueError("DSL.selection.universe must contain at least one ticker")
    return universe[0]


def _max_slippage_pct(dsl: dict) -> float:
    """Return the per-strategy slippage tolerance, defaulting to 0.005."""
    rb = dsl.get("risk_box") or {}
    em = rb.get("execution_microstructure") or {}
    val = em.get("max_slippage_tolerance_pct")
    if val is None:
        return DEFAULT_MAX_SLIPPAGE_PCT
    return float(val)


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

@dataclass
class _OpenShortCall:
    """An open short-call position being tracked through the replay loop."""

    contract_ticker: str
    open_date: date
    expiry: date
    strike: float
    opened_at_price: float  # premium received (positive number)


# ---------------------------------------------------------------------------
# Market-data accessor — query helper isolated for testability.
# ---------------------------------------------------------------------------

def _load_chain_for_symbol(
    db: Session,
    symbol: str,
    start: date,
    end: date,
) -> list[Any]:
    """Return all ``OptionHistoricalEod`` rows for ``symbol`` in [start, end]."""
    from xyz.polygon_service.models import OptionHistoricalEod

    rows = (
        db.query(OptionHistoricalEod)
        .filter(
            and_(
                OptionHistoricalEod.underlying == symbol,
                OptionHistoricalEod.date >= start,
                OptionHistoricalEod.date <= end,
            )
        )
        .all()
    )
    return rows


def _load_underlying_close_by_date(
    db: Session,
    symbol: str,
    start: date,
    end: date,
) -> dict[date, float]:
    """Return per-day implied underlying close from option chain rows.

    v1 simplification: we do not have a dedicated equity-price table in the
    engine schema (``historical_data`` is keyed on 30m intervals and may
    not have full daily coverage for backtest dates).  Instead, we derive a
    per-day underlying close from the chain itself — strikes near ATM with
    near-zero delta-times-spot drift are good enough for a covered-call
    backtest where the absolute level only feeds the assignment proxy.

    Heuristic: for each date, take the median strike of all chain rows
    weighted toward delta ~= 0.5 calls.  If no call with |delta - 0.5| <
    0.15 is present, fall back to the median of all call strikes for the
    day.  Synthetic but deterministic.
    """
    from xyz.polygon_service.models import OptionHistoricalEod

    rows = (
        db.query(OptionHistoricalEod)
        .filter(
            and_(
                OptionHistoricalEod.underlying == symbol,
                OptionHistoricalEod.option_type == "CALL",
                OptionHistoricalEod.date >= start,
                OptionHistoricalEod.date <= end,
            )
        )
        .all()
    )

    by_date: dict[date, list[tuple[float, float | None]]] = {}
    for r in rows:
        by_date.setdefault(r.date, []).append(
            (_safe_float(r.strike), _safe_float(r.delta) if r.delta is not None else None)
        )

    closes: dict[date, float] = {}
    for d, samples in by_date.items():
        # Prefer near-ATM (delta ~ 0.5) call as a proxy for spot.
        atm = [s for (s, dlt) in samples if dlt is not None and abs(dlt - 0.5) < 0.15]
        if atm:
            atm_sorted = sorted(atm)
            closes[d] = atm_sorted[len(atm_sorted) // 2]
            continue
        # Fallback: median strike.
        strikes = sorted(s for (s, _) in samples)
        if strikes:
            closes[d] = strikes[len(strikes) // 2]
    return closes


# ---------------------------------------------------------------------------
# Replay loop
# ---------------------------------------------------------------------------

def run_backtest(
    *,
    dsl: dict,
    start_date: date,
    end_date: date,
    db_session: Session,
) -> BacktestResult:
    """Run the day-by-day replay for ``dsl`` over ``[start_date, end_date]``.

    Returns
    -------
    BacktestResult
        Aggregate metrics + per-trade records + a deterministic content hash.

    Raises
    ------
    ValueError
        If the DSL is not a supported template, or if start_date > end_date.
    """
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    template = dsl.get("template") or "covered_call"
    if template != "covered_call":
        raise ValueError(
            f"v1 BACKTEST supports only the 'covered_call' template, got {template!r}"
        )

    symbol = _extract_symbol(dsl)
    max_slippage = _max_slippage_pct(dsl)

    action = dsl.get("action") or {}
    delta_target = float(action.get("delta_short_target", 0.30))
    dte_min = int(action.get("dte_min", 30))
    dte_max = int(action.get("dte_max", 45))

    exit_rules = dsl.get("exit") or {}
    max_profit_pct = float(exit_rules.get("max_profit_pct_close", 0.50))
    close_when_dte_lt = exit_rules.get("close_when_dte_lt")
    if close_when_dte_lt is not None:
        close_when_dte_lt = int(close_when_dte_lt)

    # ------------------------------------------------------------------
    # 1. Load the symbol's chain rows.
    # ------------------------------------------------------------------
    chain_rows = _load_chain_for_symbol(db_session, symbol, start_date, end_date)
    if not chain_rows:
        raise ValueError(
            f"No option_historical_eod rows found for {symbol} in [{start_date}, {end_date}]"
        )

    # Underlying spot (per day) approximated from the chain itself; keeps
    # the engine self-contained against a single table.
    spot_by_date = _load_underlying_close_by_date(
        db_session, symbol, start_date, end_date
    )

    # Index rows: by date for daily iteration, by (contract_ticker, date)
    # for marking open positions.
    by_date: dict[date, list[Any]] = {}
    by_contract_date: dict[tuple[str, date], Any] = {}
    for r in chain_rows:
        by_date.setdefault(r.date, []).append(r)
        by_contract_date[(r.contract_ticker, r.date)] = r

    trading_days = sorted(by_date.keys())
    if not trading_days:
        raise ValueError(f"No trading days found for {symbol} in date range")

    # ------------------------------------------------------------------
    # 2. State.
    # ------------------------------------------------------------------
    # cash, shares, short_call: a covered-call portfolio.
    # We bootstrap with 100 shares + cash = 0; each open short call adds
    # premium to cash.
    INITIAL_SHARES = 100  # one covered-call lot
    spot_day0 = spot_by_date.get(trading_days[0])
    if spot_day0 is None:
        spot_day0 = _safe_float(chain_rows[0].close)
    cash = 0.0
    shares = INITIAL_SHARES
    open_short_call: _OpenShortCall | None = None
    closed_trades: list[TradeRecord] = []

    # NAV[0] is the starting capital — share value only, no short call yet.
    nav_series: list[float] = []
    initial_nav = shares * spot_day0
    if initial_nav <= 0:
        # Defensive: avoid division by zero downstream.
        initial_nav = 1.0

    # ------------------------------------------------------------------
    # 3. Replay loop.
    # ------------------------------------------------------------------
    for d in trading_days:
        spot = spot_by_date.get(d, spot_day0)

        # --- 3a. Mark open short-call position to market.
        short_call_liability = 0.0
        if open_short_call is not None:
            row = by_contract_date.get((open_short_call.contract_ticker, d))
            mtm_close = _safe_float(row.close) if row else open_short_call.opened_at_price
            short_call_liability = mtm_close

        # --- 3b. Exit evaluation.
        if open_short_call is not None:
            row = by_contract_date.get((open_short_call.contract_ticker, d))
            if row is not None:
                bid = _safe_float(row.bid) if hasattr(row, "bid") and getattr(row, "bid", None) is not None else _safe_float(row.close)
                ask = _safe_float(row.ask) if hasattr(row, "ask") and getattr(row, "ask", None) is not None else _safe_float(row.close)
                # OptionHistoricalEod doesn't store bid/ask — use close as
                # the EOD print on both sides (zero spread).  Tests can
                # subclass / monkeypatch in additional fields if needed.
                bid = _safe_float(row.close)
                ask = _safe_float(row.close)
                close_price = compute_fill(
                    side="long",  # buying-to-close a short
                    bid=bid,
                    ask=ask,
                    max_slippage_pct=max_slippage,
                )

                # Max-profit close: short premium decayed >= threshold.
                premium_in = open_short_call.opened_at_price
                if premium_in > 0:
                    profit_pct = (premium_in - close_price) / premium_in
                else:
                    profit_pct = 0.0

                close_now = False
                close_reason = ""
                if profit_pct >= max_profit_pct:
                    close_now = True
                    close_reason = "max_profit"

                # Expiration close: at expiry, settle ITM/OTM.
                if d >= open_short_call.expiry:
                    close_now = True
                    close_reason = (
                        "expiration_itm"
                        if spot > open_short_call.strike
                        else "expiration_otm"
                    )
                    # At expiry, the contract settles to max(spot - strike, 0)
                    # for a call.  This becomes the "buy-back" price.
                    close_price = max(spot - open_short_call.strike, 0.0)

                # close_when_dte_lt — bail out before expiration window.
                if not close_now and close_when_dte_lt is not None:
                    dte = (open_short_call.expiry - d).days
                    if dte < close_when_dte_lt:
                        close_now = True
                        close_reason = "close_when_dte_lt"

                if close_now:
                    pnl = open_short_call.opened_at_price - close_price
                    cash += pnl  # short premium net P&L
                    closed_trades.append(
                        TradeRecord(
                            contract_ticker=open_short_call.contract_ticker,
                            open_date=open_short_call.open_date,
                            close_date=d,
                            strike=open_short_call.strike,
                            expiry=open_short_call.expiry,
                            opened_at_price=open_short_call.opened_at_price,
                            closed_at_price=close_price,
                            pnl=pnl,
                            close_reason=close_reason,
                        )
                    )
                    open_short_call = None
                    short_call_liability = 0.0

        # --- 3c. Open evaluation — only if no open call.
        if open_short_call is None:
            # Find candidate contracts: today's chain rows for calls with
            # DTE in [dte_min, dte_max] and delta nearest to target.
            candidates: list[tuple[float, Any]] = []  # (delta_distance, row)
            for r in by_date[d]:
                if r.option_type != "CALL":
                    continue
                if r.delta is None:
                    continue
                rd = _safe_float(r.delta)
                # Short call: skip ATM/ITM (delta close to 1) and zero-delta wings.
                if rd <= 0 or rd >= 1.0:
                    continue
                dte = (r.expiry - d).days
                if dte < dte_min or dte > dte_max:
                    continue
                if r.close is None or _safe_float(r.close) <= 0:
                    continue
                # Strike must be OTM for a covered call.
                strike = _safe_float(r.strike)
                if strike <= spot:
                    continue
                distance = abs(rd - delta_target)
                candidates.append((distance, r))

            if candidates:
                # Tiebreaker: nearest-to-target delta, then lowest strike,
                # then lowest expiry, then contract ticker lex order.
                candidates.sort(
                    key=lambda kv: (
                        round(kv[0], 6),
                        _safe_float(kv[1].strike),
                        kv[1].expiry,
                        kv[1].contract_ticker,
                    )
                )
                best = candidates[0][1]
                fill_price = compute_fill(
                    side="short",
                    bid=_safe_float(best.close),
                    ask=_safe_float(best.close),
                    max_slippage_pct=max_slippage,
                )
                if fill_price > 0:
                    open_short_call = _OpenShortCall(
                        contract_ticker=best.contract_ticker,
                        open_date=d,
                        expiry=best.expiry,
                        strike=_safe_float(best.strike),
                        opened_at_price=fill_price,
                    )
                    cash += fill_price
                    short_call_liability = _safe_float(best.close)

        # --- 3d. End-of-day NAV.
        nav = cash + shares * spot - short_call_liability
        # Normalise so NAV[0] starts at 1.0 — metrics layer is unitless.
        nav_series.append(nav / initial_nav)

    # ------------------------------------------------------------------
    # 4. Aggregate metrics.
    # ------------------------------------------------------------------
    closed_pnls = [t.pnl for t in closed_trades]
    assignment_count = sum(1 for t in closed_trades if t.close_reason == "expiration_itm")

    metrics = {
        "total_return": total_return(nav_series),
        "cagr": cagr(nav_series),
        "max_drawdown": max_drawdown(nav_series),
        "sharpe": sharpe(nav_series),
        "win_rate": win_rate(closed_pnls),
        "avg_winner_pnl": avg_pnl(closed_pnls, side="win"),
        "avg_loser_pnl": avg_pnl(closed_pnls, side="loss"),
        "assignment_count": assignment_count,
        "n_trades": len(closed_trades),
        "nav_series_len": len(nav_series),
    }

    # ------------------------------------------------------------------
    # 5. Hash + return.
    # ------------------------------------------------------------------
    digest = _content_hash(
        dsl=dsl,
        start=start_date,
        end=end_date,
        metrics=metrics,
    )
    logger.info(
        "backtest done symbol=%s days=%d trades=%d hash=%.8s…",
        symbol,
        len(nav_series),
        len(closed_trades),
        digest,
    )

    return BacktestResult(
        start_date=start_date,
        end_date=end_date,
        metrics=metrics,
        trades=closed_trades,
        nav_series=nav_series,
        content_hash=digest,
        dsl_snapshot=dsl,
    )
