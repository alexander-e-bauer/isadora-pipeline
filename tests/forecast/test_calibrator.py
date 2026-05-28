"""Calibrator tests — recover known params from synthetic data."""
from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np

from xyz.forecast.calibrator import Calibrator
from xyz.forecast.regime_finder import RecentWindowFinder
from xyz.forecast.schemas import ForecastOverrides


# ---------------------------------------------------------------------------
# Synthetic-data fixtures — wrap a fake DB session that returns generated
# log-returns and IV time series matching known params.
# ---------------------------------------------------------------------------

def _make_synthetic_db(
    *,
    mu_true: float,
    sigma_true: float,
    iv_atm_path: np.ndarray,           # one ATM IV per day
    n_days: int,
    seed: int = 1,
):
    """Return a mock db session that yields synthetic returns + IV series."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    daily_log_returns = (mu_true - 0.5 * sigma_true**2) * dt + sigma_true * np.sqrt(dt) * rng.standard_normal(n_days)

    db = MagicMock()
    # The Calibrator calls db helpers we stub out:
    db.get_daily_log_returns = MagicMock(return_value=daily_log_returns)
    db.get_atm_iv_series      = MagicMock(return_value=iv_atm_path)
    db.get_t0_iv_surface      = MagicMock(return_value={(100.0, 30): 0.20, (105.0, 30): 0.21})
    db.get_t0_iv_atm          = MagicMock(return_value=float(iv_atm_path[-1]))
    db.get_t0_realized_volatility = MagicMock(return_value=sigma_true)
    return db


def _make_calibrator(db):
    return Calibrator(regime_finder=RecentWindowFinder(), db=db)


def test_recovers_known_mu_sigma_from_synthetic_gbm():
    n_days = 252 * 5     # 5 years of synthetic returns
    db = _make_synthetic_db(
        mu_true=0.08, sigma_true=0.25,
        iv_atm_path=np.full(n_days, 0.20), n_days=n_days,
    )
    cal = _make_calibrator(db)
    params = cal.calibrate(symbol="AAPL", t0=date(2025, 12, 31),
                           overrides=ForecastOverrides())
    assert abs(params.mu_spot - 0.08) < 0.05
    assert abs(params.sigma_spot - 0.25) < 0.02


def test_recovers_negative_rho_when_anticorrelated():
    """Construct synthetic ATM-IV that moves opposite to spot. ρ should < 0."""
    n_days = 252 * 3
    rng = np.random.default_rng(11)
    # Common shock; spot follows shock, IV follows -shock
    eps = rng.standard_normal(n_days)
    spot_returns = 0.20 * np.sqrt(1/252) * eps
    iv_log_diffs = -0.50 * np.sqrt(1/252) * eps
    iv_path = np.exp(np.cumsum(iv_log_diffs)) * 0.20

    db = MagicMock()
    db.get_daily_log_returns = MagicMock(return_value=spot_returns)
    db.get_atm_iv_series      = MagicMock(return_value=iv_path)
    db.get_t0_iv_surface      = MagicMock(return_value={(100.0, 30): 0.20})
    db.get_t0_iv_atm          = MagicMock(return_value=float(iv_path[-1]))
    db.get_t0_realized_volatility = MagicMock(return_value=0.20)

    params = _make_calibrator(db).calibrate(
        symbol="AAPL", t0=date(2025, 12, 31), overrides=ForecastOverrides(),
    )
    assert params.iv_rho < -0.5


def test_overrides_apply_multiplicatively():
    db = _make_synthetic_db(
        mu_true=0.05, sigma_true=0.20,
        iv_atm_path=np.full(252, 0.18), n_days=252,
    )
    params = _make_calibrator(db).calibrate(
        symbol="AAPL", t0=date(2025, 12, 31),
        overrides=ForecastOverrides(sigma_spot_mult=2.0, sigma_iv_mult=0.5),
    )
    # Calibrated σ is ~0.20; after 2× → ~0.40
    assert 0.35 < params.sigma_spot < 0.45
    # σ_v halved
    assert params.iv_sigma_v < 0.5    # rough sanity, exact value depends on synthetic noise


def test_absolute_overrides_replace_calibrated_values():
    db = _make_synthetic_db(
        mu_true=0.05, sigma_true=0.20,
        iv_atm_path=np.full(252, 0.18), n_days=252,
    )
    params = _make_calibrator(db).calibrate(
        symbol="AAPL", t0=date(2025, 12, 31),
        overrides=ForecastOverrides(drift_override=0.15, rho_override=-0.9),
    )
    assert params.mu_spot == 0.15
    assert params.iv_rho == -0.9


def test_lookback_windows_recorded_as_concrete_dates():
    db = _make_synthetic_db(
        mu_true=0.05, sigma_true=0.20,
        iv_atm_path=np.full(252, 0.18), n_days=252,
    )
    params = _make_calibrator(db).calibrate(
        symbol="AAPL", t0=date(2025, 12, 31), overrides=ForecastOverrides(),
    )
    assert len(params.lookback_windows) >= 1
    assert all(isinstance(w[0], date) and isinstance(w[1], date)
               for w in params.lookback_windows)


def test_raises_on_insufficient_data():
    db = _make_synthetic_db(
        mu_true=0.05, sigma_true=0.20,
        iv_atm_path=np.full(30, 0.18), n_days=30,   # < 60 days
    )
    import pytest
    with pytest.raises(ValueError, match="INSUFFICIENT_HISTORY|insufficient"):
        _make_calibrator(db).calibrate(
            symbol="AAPL", t0=date(2025, 12, 31), overrides=ForecastOverrides(),
        )
