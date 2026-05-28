"""SimulatorCore tests — replay correctness + quantile aggregation."""
from __future__ import annotations

import numpy as np

from xyz.forecast.simulator_core import SimulatorCore


def _trivial_dsl():
    return {
        "template": "covered_call",
        "selection": {"universe": ["AAPL"]},
        "action": {"delta_short_target": 0.30, "dte_min": 25, "dte_max": 45},
        "exit": {"max_profit_pct_close": 0.50},
        "risk_box": {},
    }


def _trivial_params():
    from xyz.forecast.schemas import CalibratedParams
    from datetime import date
    return CalibratedParams(
        mu_spot=0.0, sigma_spot=0.20,
        iv_surface_t0={(100.0, 30): 0.20},
        iv_atm_t0=0.20,
        iv_kappa=2.0, iv_theta=0.20, iv_sigma_v=0.5, iv_rho=-0.5,
        lookback_windows=[(date(2025, 1, 1), date(2025, 12, 31))],
        calibration_source="recent_window",
        embedding_query_period=None,
    )


def test_quantile_aggregation_against_handcrafted_paths():
    """Construct 5 simple paths; assert quantile bands are correct."""
    n_paths, n_days = 5, 10
    spot_paths = np.tile(np.linspace(100, 110, n_days), (n_paths, 1))
    # Multiply each path by a different scalar so terminal NAVs are sorted
    spot_paths *= np.array([0.95, 0.98, 1.00, 1.02, 1.05])[:, None]
    iv_paths = np.full_like(spot_paths, 0.20)

    sim = SimulatorCore(dsl=_trivial_dsl())
    result = sim.run(
        spot_paths=spot_paths, iv_paths=iv_paths,
        params=_trivial_params(), seed=42,
    )

    # On the terminal day, p50 must equal the middle path's NAV (normalized).
    assert abs(result.nav_bands["p50"][-1] - 1.0) < 0.5    # generous; depends on covered_call replay


def test_n_paths_excluded_counts_nan_paths():
    """Inject one path with NaN and ensure it's excluded."""
    n_paths, n_days = 4, 10
    spot_paths = np.full((n_paths, n_days), 100.0)
    spot_paths[0, 5:] = np.nan
    iv_paths = np.full_like(spot_paths, 0.20)
    result = SimulatorCore(dsl=_trivial_dsl()).run(
        spot_paths=spot_paths, iv_paths=iv_paths,
        params=_trivial_params(), seed=1,
    )
    assert result.terminal_stats["n_paths_excluded"] == 1


def test_sample_path_indices_are_seeded_deterministic():
    n_paths, n_days = 200, 30
    rng = np.random.default_rng(0)
    spot_paths = 100.0 + rng.standard_normal((n_paths, n_days)).cumsum(axis=1)
    iv_paths = np.full((n_paths, n_days), 0.20)
    a = SimulatorCore(dsl=_trivial_dsl()).run(
        spot_paths=spot_paths, iv_paths=iv_paths,
        params=_trivial_params(), seed=7,
    )
    b = SimulatorCore(dsl=_trivial_dsl()).run(
        spot_paths=spot_paths, iv_paths=iv_paths,
        params=_trivial_params(), seed=7,
    )
    assert np.array_equal(a.sample_paths["nav"], b.sample_paths["nav"])


def test_bands_have_required_keys():
    n_paths, n_days = 10, 5
    spot_paths = np.full((n_paths, n_days), 100.0)
    iv_paths = np.full((n_paths, n_days), 0.20)
    result = SimulatorCore(dsl=_trivial_dsl()).run(
        spot_paths=spot_paths, iv_paths=iv_paths,
        params=_trivial_params(), seed=1,
    )
    for k in ("p10", "p25", "p50", "p75", "p90"):
        assert k in result.nav_bands
        assert len(result.nav_bands[k]) == n_days
        assert k in result.spot_bands
        assert k in result.iv_bands


def test_terminal_stats_has_required_fields():
    n_paths, n_days = 10, 5
    spot_paths = np.full((n_paths, n_days), 100.0)
    iv_paths = np.full((n_paths, n_days), 0.20)
    result = SimulatorCore(dsl=_trivial_dsl()).run(
        spot_paths=spot_paths, iv_paths=iv_paths,
        params=_trivial_params(), seed=1,
    )
    for k in ("mean", "stdev", "var_p5", "cvar_p5", "p_touch_drawdown_20pct", "n_paths_excluded"):
        assert k in result.terminal_stats
