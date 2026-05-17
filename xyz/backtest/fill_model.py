"""Daily-bar fill model with slippage clamp (v1 — intentionally dumb).

Plan §495 is explicit: keep the v1 fill model deterministic and simple.
This module contains exactly one public function, ``compute_fill``, which
returns the price at which a daily-bar trade is assumed to fill given:
    * ``bid`` / ``ask`` — the EOD chain row's quotes
    * ``side``          — ``"short"`` (selling) or ``"long"`` (buying)
    * ``max_slippage_pct`` — the per-strategy tolerance clamp pulled from
      ``dsl.risk_box.execution_microstructure.max_slippage_tolerance_pct``.
      Defaults to ``0.005`` (5 bps) if ``None`` per the plan's contract.

Slippage rule
-------------
    half_spread = (ask - bid) / 2
    slippage    = min(half_spread, max_slippage_pct * mid)
    short fill  = mid - slippage   (you receive less)
    long  fill  = mid + slippage   (you pay more)

This deliberately ignores microstructure (depth, taker queue, etc.) — that
modeling belongs in v1.5+.  Keeping the v1 fill model dumb is what makes
the day-by-day result trivially reproducible across runs.
"""
from __future__ import annotations

from typing import Literal

DEFAULT_MAX_SLIPPAGE_PCT = 0.005


Side = Literal["short", "long"]


def compute_fill(
    *,
    side: Side,
    bid: float,
    ask: float,
    max_slippage_pct: float | None,
) -> float:
    """Return the fill price for a daily-bar trade.

    Parameters
    ----------
    side:
        ``"short"`` for a sell (receive less) or ``"long"`` for a buy (pay more).
    bid, ask:
        EOD chain quote for the contract.  Must satisfy ``ask >= bid >= 0``.
        If both are zero (gap day quote) the function returns ``0.0``.
    max_slippage_pct:
        Strategy-level tolerance clamp.  ``None`` means "use the v1 default".

    Returns
    -------
    float
        The price at which the trade is assumed to fill.

    Raises
    ------
    ValueError
        If ``side`` is not ``"short"`` or ``"long"``.
    """
    if side not in ("short", "long"):
        raise ValueError(f"side must be 'short' or 'long', got {side!r}")

    if max_slippage_pct is None:
        max_slippage_pct = DEFAULT_MAX_SLIPPAGE_PCT

    bid = float(bid or 0.0)
    ask = float(ask or 0.0)

    # Degenerate / gap-day quote: bail with a non-actionable zero.  The
    # backtest engine treats a zero quote as "no liquidity" and skips the
    # trade — better than fabricating a synthetic price.
    if bid <= 0 and ask <= 0:
        return 0.0

    # Normalise inverted quotes (data-quality guard).  ask < bid is a vendor
    # bug, not a model decision; we clamp so downstream math doesn't go
    # negative.
    if ask < bid:
        ask, bid = bid, ask

    mid = (bid + ask) / 2.0
    half_spread = (ask - bid) / 2.0
    slippage = min(half_spread, max_slippage_pct * mid)

    if side == "short":
        return mid - slippage
    return mid + slippage
