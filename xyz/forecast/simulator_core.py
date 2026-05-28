"""SimulatorCore — runs the covered-call replay over generated paths.

Pure compute around xyz.forecast.pricer for option marks and
xyz.forecast.path_generator outputs for spot + IV trajectories.

Vectorized over paths; loops only over days. Per-path state is held
in flat arrays (cash, shares, open-short-call metadata).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from xyz.forecast.pricer import (
    bs_call_price,
    strike_from_delta,
)
from xyz.forecast.schemas import CalibratedParams


@dataclass
class SimulationResult:
    nav_bands: dict[str, list[float]]
    spot_bands: dict[str, list[float]]
    iv_bands: dict[str, list[float]]
    terminal_stats: dict
    terminal_nav_histogram: dict
    sample_paths: dict[str, np.ndarray]


_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
_QUANT_KEYS = ("p10", "p25", "p50", "p75", "p90")


class SimulatorCore:
    def __init__(self, *, dsl: dict) -> None:
        self._dsl = dsl

    def run(
        self, *,
        spot_paths: np.ndarray,         # (n_paths, n_days)
        iv_paths: np.ndarray,           # (n_paths, n_days), ATM IV reference
        params: CalibratedParams,
        seed: int,
    ) -> SimulationResult:
        n_paths, n_days = spot_paths.shape
        action = self._dsl.get("action") or {}
        delta_target = float(action.get("delta_short_target", 0.30))
        dte_mid = int((action.get("dte_min", 25) + action.get("dte_max", 45)) / 2)
        exit_rules = self._dsl.get("exit") or {}
        max_profit_pct = float(exit_rules.get("max_profit_pct_close", 0.50))

        # Per-path state
        cash = np.zeros(n_paths)
        shares = np.full(n_paths, 100.0)
        short_open = np.zeros(n_paths, dtype=bool)
        short_strike = np.zeros(n_paths)
        short_expiry_day = np.full(n_paths, -1, dtype=np.int64)
        short_premium_in = np.zeros(n_paths)
        nav_paths = np.zeros((n_paths, n_days))

        for t in range(n_days):
            spot_t = spot_paths[:, t]
            atm_iv_t = iv_paths[:, t]

            # --- Mark existing short positions ---
            short_liab = np.zeros(n_paths)
            if short_open.any():
                # Days-to-expiry per path; assume t0 surface ATM=iv_atm_t0
                dte_now = np.maximum(short_expiry_day - t, 1)
                # Skew-shifted contract IV: scale t0 surface entry by ATM ratio
                # Approx: use atm_iv_t for the contract (skew shape lost for v1
                # at non-t0 strikes — see spec §4.4).
                contract_iv = atm_iv_t * (1.0)   # multiplicative shift baseline
                # Vectorized price
                short_liab = bs_call_price(
                    spot_t, 100.0, dte_now[0], contract_iv,  # strike per-path next iteration
                )
                # Above is approximate; for per-path strike, loop:
                # (kept simple — refine if needed)
                liab = np.empty(n_paths)
                for i in range(n_paths):
                    if short_open[i]:
                        liab[i] = bs_call_price(
                            np.array([spot_t[i]]),
                            float(short_strike[i]),
                            int(max(short_expiry_day[i] - t, 1)),
                            np.array([float(atm_iv_t[i])]),
                        )[0]
                    else:
                        liab[i] = 0.0
                short_liab = liab

                # Exit rules
                profit_pct = np.where(
                    short_premium_in > 0,
                    (short_premium_in - short_liab) / np.maximum(short_premium_in, 1e-9),
                    0.0,
                )
                exit_max_profit = short_open & (profit_pct >= max_profit_pct)
                exit_expired = short_open & (t >= short_expiry_day)

                # ITM expiration → assignment: shares called away at strike
                itm = exit_expired & (spot_t > short_strike)
                cash[itm] += shares[itm] * short_strike[itm]
                shares[itm] = 0.0

                closing = exit_max_profit | exit_expired
                # Net P&L on closing trades
                cash[closing] += short_premium_in[closing] - short_liab[closing]
                short_open[closing] = False
                short_liab[closing] = 0.0

            # --- Open new short call where no open position ---
            opening = ~short_open
            if opening.any():
                # Strike from delta target, vectorized
                strikes = strike_from_delta(
                    spot=spot_t[opening], dte_days=dte_mid,
                    iv=atm_iv_t[opening], target_delta=delta_target,
                )
                # Premium received (positive)
                premium = bs_call_price(
                    spot_t[opening], strikes.mean() if strikes.size else 100.0,
                    dte_mid, atm_iv_t[opening],
                )
                # Per-path strike persistence
                idx = np.where(opening)[0]
                for j, i in enumerate(idx):
                    short_open[i] = True
                    short_strike[i] = float(strikes[j])
                    short_expiry_day[i] = t + dte_mid
                    short_premium_in[i] = float(
                        bs_call_price(
                            np.array([spot_t[i]]),
                            float(strikes[j]),
                            dte_mid,
                            np.array([float(atm_iv_t[i])]),
                        )[0]
                    )
                    cash[i] += short_premium_in[i]

            # Re-mark short liability for path-correct NAV
            liab_now = np.zeros(n_paths)
            for i in range(n_paths):
                if short_open[i]:
                    liab_now[i] = bs_call_price(
                        np.array([spot_t[i]]),
                        float(short_strike[i]),
                        int(max(short_expiry_day[i] - t, 1)),
                        np.array([float(atm_iv_t[i])]),
                    )[0]

            nav_paths[:, t] = cash + shares * spot_t - liab_now

        # Normalize each path so NAV[:, 0] = 1.0
        first = nav_paths[:, 0]
        first_safe = np.where(np.abs(first) < 1e-9, 1.0, first)
        nav_paths = nav_paths / first_safe[:, None]

        # Detect bad paths (NaN/Inf anywhere)
        bad = ~np.isfinite(nav_paths).all(axis=1)
        good_idx = np.where(~bad)[0]
        n_excluded = int(bad.sum())

        nav_good = nav_paths[good_idx]
        spot_good = spot_paths[good_idx]
        iv_good = iv_paths[good_idx]

        # Quantile aggregation
        nav_bands = _quantile_bands(nav_good)
        spot_bands = _quantile_bands(spot_good)
        iv_bands = _quantile_bands(iv_good)

        # Terminal stats
        term_nav = nav_good[:, -1] if nav_good.size else np.array([1.0])
        term_stats = {
            "mean": float(np.mean(term_nav)),
            "stdev": float(np.std(term_nav, ddof=1)) if term_nav.size > 1 else 0.0,
            "var_p5": float(np.quantile(term_nav, 0.05)),
            "cvar_p5": float(term_nav[term_nav <= np.quantile(term_nav, 0.05)].mean())
                       if (term_nav <= np.quantile(term_nav, 0.05)).any() else 0.0,
            "p_touch_drawdown_20pct": float(np.mean(np.min(nav_good, axis=1) <= 0.80))
                                      if nav_good.size else 0.0,
            "n_paths_excluded": n_excluded,
        }

        # Terminal-NAV histogram (20 buckets)
        t_min, t_max = float(np.min(term_nav)), float(np.max(term_nav))
        # Guard degenerate case (all paths identical) — NumPy 2.x raises
        # ValueError: "Too many bins for data range" when min==max with bins=20.
        if t_min == t_max:
            t_min, t_max = t_min - 0.5, t_max + 0.5
        edges = np.linspace(t_min, t_max, 21).tolist()
        counts = np.histogram(term_nav, bins=np.linspace(t_min, t_max, 21))[0].tolist()
        hist = {"edges": edges, "counts": counts}

        # Sample paths — 50 indices seeded by `seed`
        rng = np.random.default_rng(seed)
        k_sample = min(50, nav_good.shape[0])
        if k_sample > 0:
            sample_idx = rng.choice(nav_good.shape[0], size=k_sample, replace=False)
            sample_idx.sort()
            sample_paths = {
                "nav":  nav_good[sample_idx].tolist(),
                "spot": spot_good[sample_idx].tolist(),
                "iv":   iv_good[sample_idx].tolist(),
            }
        else:
            sample_paths = {"nav": [], "spot": [], "iv": []}

        # Convert ndarrays-as-band-values to list[float] for JSON
        return SimulationResult(
            nav_bands={k: v.tolist() for k, v in nav_bands.items()},
            spot_bands={k: v.tolist() for k, v in spot_bands.items()},
            iv_bands={k: v.tolist() for k, v in iv_bands.items()},
            terminal_stats=term_stats,
            terminal_nav_histogram=hist,
            sample_paths=sample_paths,
        )


def _quantile_bands(paths: np.ndarray) -> dict[str, np.ndarray]:
    if paths.size == 0:
        return {k: np.array([]) for k in _QUANT_KEYS}
    q = np.quantile(paths, list(_QUANTILES), axis=0)
    return {k: q[i] for i, k in enumerate(_QUANT_KEYS)}
