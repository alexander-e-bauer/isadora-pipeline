"""ForwardGBMTwoFactor tests — determinism + statistical recovery."""
from __future__ import annotations

import numpy as np

from xyz.forecast.path_generator import ForwardGBMTwoFactor
from xyz.forecast.schemas import CalibratedParams


def _params(*, mu=0.05, sigma=0.20, kappa=2.0, theta=0.20,
            sigma_v=0.50, rho=-0.50, iv_atm_t0=0.20) -> CalibratedParams:
    from datetime import date
    return CalibratedParams(
        mu_spot=mu, sigma_spot=sigma,
        iv_surface_t0={}, iv_atm_t0=iv_atm_t0,
        iv_kappa=kappa, iv_theta=theta, iv_sigma_v=sigma_v, iv_rho=rho,
        lookback_windows=[(date(2025, 1, 1), date(2025, 12, 31))],
        calibration_source="recent_window",
        embedding_query_period=None,
    )


def test_same_seed_produces_byte_identical_paths():
    gen = ForwardGBMTwoFactor(spot0=100.0, params=_params())
    s1, iv1 = gen.generate(n_paths=100, n_days=50, seed=42)
    s2, iv2 = gen.generate(n_paths=100, n_days=50, seed=42)
    assert np.array_equal(s1, s2)
    assert np.array_equal(iv1, iv2)


def test_different_seeds_produce_different_paths():
    gen = ForwardGBMTwoFactor(spot0=100.0, params=_params())
    s1, _ = gen.generate(n_paths=100, n_days=50, seed=1)
    s2, _ = gen.generate(n_paths=100, n_days=50, seed=2)
    assert not np.array_equal(s1, s2)


def test_path_correlation_matches_rho_at_scale():
    rho_input = -0.70
    gen = ForwardGBMTwoFactor(spot0=100.0, params=_params(rho=rho_input))
    s, iv = gen.generate(n_paths=10000, n_days=252, seed=7)
    d_logS = np.diff(np.log(s), axis=1)
    d_logIV = np.diff(np.log(iv), axis=1)
    observed = np.corrcoef(d_logS.flatten(), d_logIV.flatten())[0, 1]
    assert abs(observed - rho_input) < 0.03


def test_path_drift_matches_mu():
    mu_input = 0.10
    gen = ForwardGBMTwoFactor(spot0=100.0, params=_params(mu=mu_input, sigma=0.10))
    s, _ = gen.generate(n_paths=10000, n_days=252, seed=7)
    log_returns_terminal = np.log(s[:, -1] / s[:, 0])
    mean_annualized = np.mean(log_returns_terminal) * (252 / s.shape[1])
    expected = mu_input - 0.5 * 0.10**2
    assert abs(mean_annualized - expected) < 0.02


def test_path_vol_matches_sigma():
    sigma_input = 0.30
    gen = ForwardGBMTwoFactor(spot0=100.0, params=_params(mu=0.0, sigma=sigma_input))
    s, _ = gen.generate(n_paths=10000, n_days=252, seed=7)
    d_logS = np.diff(np.log(s), axis=1)
    observed_annualized = np.std(d_logS) * np.sqrt(252)
    assert abs(observed_annualized - sigma_input) < 0.01


def test_ou_iv_mean_reverts_to_theta():
    theta_input = 0.25
    gen = ForwardGBMTwoFactor(
        spot0=100.0, params=_params(theta=theta_input, iv_atm_t0=0.10, kappa=2.0),
    )
    _, iv = gen.generate(n_paths=2000, n_days=2520, seed=7)
    second_half = iv[:, 1260:]
    median_second_half = np.median(second_half)
    assert abs(median_second_half - theta_input) < 0.03
