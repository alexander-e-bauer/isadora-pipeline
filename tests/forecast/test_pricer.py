"""Black-Scholes pricer tests — closed-form sanity checks + vectorization."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from xyz.forecast.pricer import (
    bs_call_price,
    bs_call_delta,
    strike_from_delta,
)


def test_atm_call_price_matches_closed_form_scalar():
    spot, strike, dte, iv, r = 100.0, 100.0, 30, 0.20, 0.045
    T = dte / 365.0
    d1 = (np.log(spot / strike) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    expected = spot * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
    got = bs_call_price(np.array([spot]), strike, dte, np.array([iv]), r=r)
    assert got.shape == (1,)
    assert abs(got[0] - expected) < 1e-6


def test_put_call_parity_holds():
    spot, strike, dte, iv, r = 100.0, 95.0, 60, 0.25, 0.045
    T = dte / 365.0
    call = bs_call_price(np.array([spot]), strike, dte, np.array([iv]), r=r)[0]
    d1 = (np.log(spot / strike) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    put = strike * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    assert abs((call - put) - (spot - strike * np.exp(-r * T))) < 1e-6


def test_atm_delta_approximately_half():
    delta = bs_call_delta(np.array([100.0]), 100.0, 30, np.array([0.20]), r=0.045)[0]
    assert 0.48 < delta < 0.60


def test_strike_from_delta_round_trip():
    spot, dte, iv, r = 100.0, 30, 0.20, 0.045
    target_delta = 0.30
    K = strike_from_delta(np.array([spot]), dte, np.array([iv]), target_delta, r=r)[0]
    delta_back = bs_call_delta(np.array([spot]), K, dte, np.array([iv]), r=r)[0]
    assert abs(delta_back - target_delta) < 1e-4


def test_zero_iv_returns_intrinsic_value():
    spot, strike, dte, r = 110.0, 100.0, 30, 0.045
    T = dte / 365.0
    expected_intrinsic_pv = max(spot - strike * np.exp(-r * T), 0.0)
    got = bs_call_price(np.array([spot]), strike, dte, np.array([1e-8]), r=r)[0]
    assert abs(got - expected_intrinsic_pv) < 1e-4


def test_far_otm_returns_near_zero():
    got = bs_call_price(np.array([100.0]), 200.0, 30, np.array([0.20]), r=0.045)[0]
    assert got < 1e-3


def test_vectorization_matches_scalar_loop():
    spots = np.array([95.0, 100.0, 105.0, 110.0])
    ivs = np.array([0.18, 0.20, 0.22, 0.24])
    strike, dte, r = 100.0, 30, 0.045
    vec = bs_call_price(spots, strike, dte, ivs, r=r)
    loop = np.array([bs_call_price(np.array([s]), strike, dte, np.array([v]), r=r)[0]
                     for s, v in zip(spots, ivs)])
    assert np.allclose(vec, loop, atol=1e-10)
