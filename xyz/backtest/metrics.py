"""Backtest performance metrics — pure functions over a NAV series.

All functions take a list of floats (NAV per trading day, in order) and
return a single float.  No numpy dependency — keeps the engine container
slim and the math auditable line-by-line.

Determinism note
----------------
Every return value is a Python float.  The hashing layer in
``engine.py`` serialises results at fixed precision (6 decimals) before
hashing, so floating-point representation drift in the LSBs cannot
change the content hash.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

TRADING_DAYS_PER_YEAR = 252


def total_return(nav: Sequence[float]) -> float:
    """Cumulative return from NAV[0] to NAV[-1].

    Returns 0.0 for an empty or single-point series (no return possible).
    """
    if not nav or len(nav) < 2:
        return 0.0
    start = float(nav[0])
    if start <= 0:
        return 0.0
    return float(nav[-1]) / start - 1.0


def cagr(nav: Sequence[float]) -> float:
    """Compound annual growth rate, annualised by 252 trading days.

    Uses ``len(nav)`` as the trading-day count; for a backtest the NAV
    list contains one entry per trading day, so this is exact.
    Returns 0.0 for series too short to annualise.
    """
    if not nav or len(nav) < 2:
        return 0.0
    start = float(nav[0])
    end = float(nav[-1])
    if start <= 0 or end <= 0:
        return 0.0
    n_days = len(nav)
    return (end / start) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0


def max_drawdown(nav: Sequence[float]) -> float:
    """Largest peak-to-trough drawdown over the series (always <= 0).

    Computed as the minimum of ``(nav[i] - cummax[i]) / cummax[i]``.
    """
    if not nav:
        return 0.0
    peak = float(nav[0])
    worst = 0.0
    for v in nav:
        v = float(v)
        if v > peak:
            peak = v
        if peak <= 0:
            continue
        dd = (v - peak) / peak
        if dd < worst:
            worst = dd
    return worst


def _daily_returns(nav: Sequence[float]) -> list[float]:
    """Return the day-over-day arithmetic returns implied by ``nav``."""
    out: list[float] = []
    for i in range(1, len(nav)):
        prev = float(nav[i - 1])
        cur = float(nav[i])
        if prev <= 0:
            out.append(0.0)
        else:
            out.append(cur / prev - 1.0)
    return out


def sharpe(nav: Sequence[float]) -> float:
    """Annualised Sharpe ratio computed from daily-return mean / std.

    Assumes a 0% risk-free rate (v1 simplification).  Returns 0.0 when
    fewer than 2 data points or when std is zero.
    """
    if not nav or len(nav) < 2:
        return 0.0
    rets = _daily_returns(nav)
    if not rets:
        return 0.0
    mean = sum(rets) / len(rets)
    # Sample standard deviation (ddof=1) — same convention as numpy default.
    if len(rets) < 2:
        return 0.0
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(TRADING_DAYS_PER_YEAR)


def win_rate(closed_pnls: Iterable[float]) -> float:
    """Fraction of closed trades with P&L > 0.

    Returns 0.0 when there are no closed trades.
    """
    pnls = list(closed_pnls)
    if not pnls:
        return 0.0
    wins = sum(1 for p in pnls if p > 0)
    return wins / len(pnls)


def avg_pnl(closed_pnls: Iterable[float], side: str) -> float:
    """Mean P&L of either winning (``side="win"``) or losing (``side="loss"``) trades.

    Returns 0.0 if there are no trades on the requested side.
    """
    if side not in ("win", "loss"):
        raise ValueError("side must be 'win' or 'loss'")
    pnls = list(closed_pnls)
    if side == "win":
        relevant = [p for p in pnls if p > 0]
    else:
        relevant = [p for p in pnls if p <= 0]
    if not relevant:
        return 0.0
    return sum(relevant) / len(relevant)
