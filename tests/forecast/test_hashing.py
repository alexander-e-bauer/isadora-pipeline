"""Content hash determinism tests."""
from __future__ import annotations

from datetime import date

from xyz.forecast.hashing import compute_forecast_content_hash


def _base_kwargs():
    return dict(
        dsl={"template": "covered_call", "selection": {"universe": ["AAPL"]}},
        t0=date(2025, 12, 31),
        horizon_days=252,
        n_paths=1000,
        forecast_seed=42,
        calibrated_params={
            "mu_spot": 0.05, "sigma_spot": 0.20,
            "iv_surface_t0": {"100.0_30": 0.20},
            "iv_atm_t0": 0.20,
            "iv_kappa": 2.0, "iv_theta": 0.20,
            "iv_sigma_v": 0.5, "iv_rho": -0.5,
            "lookback_windows": [["2024-12-31", "2025-12-31"]],
            "calibration_source": "embedding",
            "embedding_query_period": None,
        },
        t0_market_context={
            "realized_volatility": 0.21, "rsi": 55.0,
            "trend_strength": "moderate_uptrend",
        },
        results={
            "nav_bands": {"p50": [1.0, 1.001, 1.002]},
            "terminal_stats": {"mean": 1.05, "var_p5": 0.92},
        },
    )


def test_same_inputs_produce_same_hash():
    a = compute_forecast_content_hash(**_base_kwargs())
    b = compute_forecast_content_hash(**_base_kwargs())
    assert a == b


def test_different_seed_changes_hash():
    a = compute_forecast_content_hash(**_base_kwargs())
    kwargs = _base_kwargs()
    kwargs["forecast_seed"] = 43
    b = compute_forecast_content_hash(**kwargs)
    assert a != b


def test_different_t0_changes_hash():
    a = compute_forecast_content_hash(**_base_kwargs())
    kwargs = _base_kwargs()
    kwargs["t0"] = date(2025, 12, 30)
    b = compute_forecast_content_hash(**kwargs)
    assert a != b


def test_hash_is_64_hex_chars():
    h = compute_forecast_content_hash(**_base_kwargs())
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
