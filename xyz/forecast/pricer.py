"""Black-Scholes pricer — vectorized over the path dimension.

Used by SimulatorCore to mark short-call positions to market at each
simulated day and to pick contracts by target delta on path-open days.

All functions accept numpy arrays for ``spot`` and ``iv`` (one element
per path); ``strike``, ``dte_days``, ``r``, and ``target_delta`` are
scalars (the same contract is being priced across all paths at a
given simulated day).

No randomness. Pure math. Fully deterministic.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr   # vectorized normal CDF


def bs_call_price(
    spot: np.ndarray,
    strike: float,
    dte_days: int,
    iv: np.ndarray,
    r: float = 0.045,
) -> np.ndarray:
    """Black-Scholes European call price, vectorized over paths."""
    T = max(dte_days, 1) / 365.0
    iv_safe = np.maximum(iv, 1e-8)               # avoid div-by-zero
    sqrt_T = np.sqrt(T)
    d1 = (np.log(spot / strike) + (r + 0.5 * iv_safe**2) * T) / (iv_safe * sqrt_T)
    d2 = d1 - iv_safe * sqrt_T
    return spot * ndtr(d1) - strike * np.exp(-r * T) * ndtr(d2)


def bs_call_delta(
    spot: np.ndarray,
    strike: float,
    dte_days: int,
    iv: np.ndarray,
    r: float = 0.045,
) -> np.ndarray:
    """Black-Scholes call delta = N(d1)."""
    T = max(dte_days, 1) / 365.0
    iv_safe = np.maximum(iv, 1e-8)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(spot / strike) + (r + 0.5 * iv_safe**2) * T) / (iv_safe * sqrt_T)
    return ndtr(d1)


def strike_from_delta(
    spot: np.ndarray,
    dte_days: int,
    iv: np.ndarray,
    target_delta: float,
    r: float = 0.045,
) -> np.ndarray:
    """Inverse: K = S · exp((r + σ²/2)·T − σ·√T · Φ⁻¹(δ))."""
    T = max(dte_days, 1) / 365.0
    iv_safe = np.maximum(iv, 1e-8)
    sqrt_T = np.sqrt(T)
    from scipy.special import ndtri
    z = ndtri(target_delta)
    return spot * np.exp((r + 0.5 * iv_safe**2) * T - iv_safe * sqrt_T * z)
