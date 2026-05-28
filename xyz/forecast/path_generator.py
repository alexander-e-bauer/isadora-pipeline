"""ForwardGBMTwoFactor — generates correlated (spot, ATM-IV) paths.

Spot:  d log S  = (μ - σ²/2) dt + σ √dt · ε₁
IV:    d log IV = κ (log θ - log IV) dt + σ_v √dt · ε₂

ε₁ and ε₂ are correlated bivariate normals with correlation ρ.

All RNG flows from np.random.default_rng(seed) → deterministic given seed.
Vectorized over n_paths; loops only over time steps.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from xyz.forecast.schemas import CalibratedParams


class PathGenerator(Protocol):
    def generate(
        self, *, n_paths: int, n_days: int, seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (spot_paths, iv_paths), each shape (n_paths, n_days)."""
        ...


class ForwardGBMTwoFactor:
    """Forward Monte Carlo path generator — GBM on log-spot, OU on log-IV."""

    def __init__(self, *, spot0: float, params: CalibratedParams):
        self.spot0 = float(spot0)
        self.params = params

    def generate(
        self, *, n_paths: int, n_days: int, seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        p = self.params
        dt = 1.0 / 252.0
        sqrt_dt = np.sqrt(dt)
        rng = np.random.default_rng(seed)

        # Correlated bivariate normals via Cholesky on [[1, ρ], [ρ, 1]]
        eps = rng.standard_normal((2, n_paths, n_days))
        L = np.linalg.cholesky(np.array([[1.0, p.iv_rho], [p.iv_rho, 1.0]]))
        eps_corr = np.einsum("ij,jpt->ipt", L, eps)
        eps_S = eps_corr[0]
        eps_IV = eps_corr[1]

        log_s = np.empty((n_paths, n_days))
        log_iv = np.empty((n_paths, n_days))
        log_s[:, 0] = np.log(self.spot0)
        log_iv[:, 0] = np.log(p.iv_atm_t0)
        log_theta = np.log(p.iv_theta)

        for t in range(1, n_days):
            log_s[:, t] = log_s[:, t - 1] + (
                (p.mu_spot - 0.5 * p.sigma_spot**2) * dt
                + p.sigma_spot * sqrt_dt * eps_S[:, t]
            )
            log_iv[:, t] = log_iv[:, t - 1] + (
                p.iv_kappa * (log_theta - log_iv[:, t - 1]) * dt
                + p.iv_sigma_v * sqrt_dt * eps_IV[:, t]
            )

        return np.exp(log_s), np.exp(log_iv)
