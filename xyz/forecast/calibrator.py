"""Calibrator — fits GBM + OU-on-log-IV params from historical data.

Reads only via the abstract ``db`` argument's helper methods so tests
can substitute a mock. In production those methods are implemented on
a ``CalibratorDb`` wrapper around a SQLAlchemy session.

Calibration steps (per spec §4.1):

  1. regime_finder.find(symbol, t0, k, days) → list of concrete windows
  2. db.get_daily_log_returns(symbol, windows) → returns array
  3. μ, σ = mean × 252, std × √252
  4. db.get_atm_iv_series(symbol, windows) → daily ATM-IV array
  5. Fit OU on log-IV via AR(1) OLS → κ, θ, σ_v
  6. ρ = corr(d_log_returns, d_log_IV)
  7. Snapshot IV surface at t0 + ATM IV at t0
  8. Apply DSL overrides
  9. Warn if calibrated σ diverges from realized vol
"""
from __future__ import annotations

import logging
from dataclasses import replace
from datetime import date

import numpy as np

from xyz.forecast.regime_finder import SimilarRegimeFinder
from xyz.forecast.schemas import CalibratedParams, ForecastOverrides

logger = logging.getLogger(__name__)

MIN_HISTORY_DAYS = 60
DEFAULT_LOOKBACK_DAYS = 252
DEFAULT_K_WINDOWS = 3


class Calibrator:
    def __init__(self, *, regime_finder: SimilarRegimeFinder, db) -> None:
        self._regime_finder = regime_finder
        self._db = db

    def calibrate(
        self, *,
        symbol: str,
        t0: date,
        overrides: ForecastOverrides,
    ) -> CalibratedParams:
        # 1. Pick calibration windows
        days = overrides.lookback_days or DEFAULT_LOOKBACK_DAYS
        k    = overrides.k_similar_windows or DEFAULT_K_WINDOWS
        # If user requested recent_window via overrides, swap the finder
        finder = (
            self._regime_finder
            if overrides.calibration_source == "embedding"
            else _RecentWindowFinderLite()
        )
        windows = finder.find(symbol=symbol, t0=t0, k=k, days=days, granularity="D")

        # 2. Pull returns + IV series for the concatenated windows
        returns = self._db.get_daily_log_returns(symbol=symbol, windows=windows)
        iv_atm_series = self._db.get_atm_iv_series(symbol=symbol, windows=windows)

        if len(returns) < MIN_HISTORY_DAYS:
            raise ValueError(
                f"INSUFFICIENT_HISTORY: need ≥{MIN_HISTORY_DAYS} days of returns "
                f"for {symbol} in the calibration windows, got {len(returns)}"
            )
        if len(iv_atm_series) < MIN_HISTORY_DAYS:
            raise ValueError(
                f"INSUFFICIENT_HISTORY: need ≥{MIN_HISTORY_DAYS} days of ATM-IV "
                f"for {symbol}, got {len(iv_atm_series)}"
            )

        # 3. μ, σ annualized
        mu = float(np.mean(returns)) * 252.0
        sigma = float(np.std(returns, ddof=1)) * np.sqrt(252.0)

        # 4-5. OU on log-IV via AR(1) OLS
        log_iv = np.log(np.maximum(iv_atm_series, 1e-8))
        dt = 1.0 / 252.0
        d_log_iv = np.diff(log_iv)
        x = log_iv[:-1]
        # Δlog IV = α + β · log IV · dt + noise; map to κ, θ
        # AR(1): log_iv[t+1] = a + b * log_iv[t] + ε
        # Discretize OU: a = κ θ dt, b = 1 - κ dt → κ = (1-b)/dt, θ = a/(κ dt)
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, log_iv[1:], rcond=None)
        a, b = float(coef[0]), float(coef[1])
        kappa = max((1.0 - b) / dt, 1e-3)
        theta_log = a / max(kappa * dt, 1e-8)
        theta = float(np.exp(theta_log))
        residuals = log_iv[1:] - (a + b * x)
        sigma_v = float(np.std(residuals, ddof=1)) / np.sqrt(dt)

        # 6. ρ — Pearson corr of returns vs Δ log IV (matched lengths)
        n = min(len(returns) - 1, len(d_log_iv))
        if n > 1:
            rho = float(np.corrcoef(returns[1:n + 1], d_log_iv[:n])[0, 1])
        else:
            rho = 0.0

        # 7. IV surface + ATM at t0
        iv_surface_t0 = self._db.get_t0_iv_surface(symbol=symbol, t0=t0)
        iv_atm_t0 = float(self._db.get_t0_iv_atm(symbol=symbol, t0=t0))

        # 8. Apply DSL overrides
        sigma = sigma * overrides.sigma_spot_mult
        sigma_v = sigma_v * overrides.sigma_iv_mult
        if overrides.drift_override is not None:
            mu = overrides.drift_override
        if overrides.rho_override is not None:
            rho = overrides.rho_override
        # Clamp ρ into [-0.99, 0.99] (Cholesky needs strictly less than 1)
        rho = max(min(rho, 0.99), -0.99)

        # 9. Calibration sanity warning
        realized_vol = float(self._db.get_t0_realized_volatility(symbol=symbol, t0=t0))
        if not (0.5 * realized_vol <= sigma <= 2.0 * realized_vol):
            logger.warning(
                "Calibrated sigma_spot=%.4f diverges from realized_vol=%.4f for %s; "
                "verify lookback windows",
                sigma, realized_vol, symbol,
            )

        # Embedding query period (the daily MarketEmbDay row keyed to t0) is
        # provided by the regime finder if it was used. RecentWindowFinder
        # leaves it None.
        embedding_query_period = getattr(finder, "last_query_period", None)

        return CalibratedParams(
            mu_spot=mu,
            sigma_spot=sigma,
            iv_surface_t0=iv_surface_t0,
            iv_atm_t0=iv_atm_t0,
            iv_kappa=kappa,
            iv_theta=theta,
            iv_sigma_v=sigma_v,
            iv_rho=rho,
            lookback_windows=windows,
            calibration_source=overrides.calibration_source,
            embedding_query_period=embedding_query_period,
        )


class _RecentWindowFinderLite:
    """Tiny inline fallback so Calibrator can swap finders based on override."""
    from datetime import timedelta as _td

    def find(self, *, symbol, t0, k, days, granularity="D"):
        from datetime import timedelta
        return [(t0 - timedelta(days=days - 1), t0)]
