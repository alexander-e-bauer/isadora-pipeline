"""ForecastAgent — top-level orchestration for the forecast subagent.

No Claude calls. No LLM dependency. Pure compute + event emission.
Stochastic by design, fully deterministic given the seed.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone

from xyz.forecast.calibrator import Calibrator
from xyz.forecast.hashing import compute_forecast_content_hash
from xyz.forecast.path_generator import ForwardGBMTwoFactor
from xyz.forecast.regime_finder import RecentWindowFinder
from xyz.forecast.schemas import (
    CalibratedParams,
    ForecastArtifact,
    ForecastInput,
    T0MarketContext,
)
from xyz.forecast.simulator_core import SimulatorCore
from xyz.forecast.t0_context import build_t0_market_context

logger = logging.getLogger(__name__)


class ForecastAgent:
    def __init__(self, *, db, regime_finder=None) -> None:
        self._db = db
        # Default to RecentWindowFinder; production wiring injects the
        # EmbeddingRegimeFinder when an EmbeddingStore is available.
        self._regime_finder = regime_finder or RecentWindowFinder()

    def run(self, *, input: ForecastInput) -> ForecastArtifact:
        logger.info(
            "ForecastAgent.run firm=%s strategy=%s v%s t0=%s seed=%s n_paths=%s",
            input.firm_id, input.strategy_id, input.strategy_version,
            input.t0, input.forecast_seed, input.n_paths,
        )

        # 1. Calibrate
        calibrator = Calibrator(regime_finder=self._regime_finder, db=self._db)
        params = calibrator.calibrate(
            symbol=_symbol_from_dsl(input.dsl),
            t0=input.t0,
            overrides=input.overrides,
        )

        # 2. Build t0 market context
        ctx = build_t0_market_context(
            symbol=_symbol_from_dsl(input.dsl), t0=input.t0, db=self._db,
        )

        # 3. Generate paths
        spot0 = float(self._db.get_t0_spot(
            symbol=_symbol_from_dsl(input.dsl), t0=input.t0,
        )) if hasattr(self._db, "get_t0_spot") else 100.0
        gen = ForwardGBMTwoFactor(spot0=spot0, params=params)
        spot_paths, iv_paths = gen.generate(
            n_paths=input.n_paths, n_days=input.horizon_days,
            seed=input.forecast_seed,
        )

        # 4. Run replay
        sim = SimulatorCore(dsl=input.dsl)
        result = sim.run(
            spot_paths=spot_paths, iv_paths=iv_paths,
            params=params, seed=input.forecast_seed,
        )

        # 5. Pack everything for hashing + persistence
        params_dump = _calibrated_params_to_dict(params)
        ctx_dump = _t0_market_context_to_dict(ctx)
        results_dump = {
            "nav_bands": result.nav_bands,
            "spot_bands": result.spot_bands,
            "iv_bands": result.iv_bands,
            "terminal_stats": result.terminal_stats,
            "terminal_nav_histogram": result.terminal_nav_histogram,
            "sample_paths": result.sample_paths,
        }

        content_hash = compute_forecast_content_hash(
            dsl=input.dsl,
            t0=input.t0,
            horizon_days=input.horizon_days,
            n_paths=input.n_paths,
            forecast_seed=input.forecast_seed,
            calibrated_params=params_dump,
            t0_market_context=ctx_dump,
            results=results_dump,
        )

        artifact = ForecastArtifact(
            strategy_id=input.strategy_id,
            strategy_version=input.strategy_version,
            firm_id=input.firm_id,
            t0=input.t0,
            horizon_days=input.horizon_days,
            n_paths=input.n_paths,
            forecast_seed=input.forecast_seed,
            dsl_snapshot=input.dsl,
            calibrated_params=params_dump,
            t0_market_context=ctx_dump,
            calibration_source=params.calibration_source,
            research_artifact_id=input.research_artifact_id,
            nav_bands=result.nav_bands,
            spot_bands=result.spot_bands,
            iv_bands=result.iv_bands,
            terminal_stats=result.terminal_stats,
            terminal_nav_histogram=result.terminal_nav_histogram,
            sample_paths=result.sample_paths,
            content_hash=content_hash,
            generated_at=datetime.now(timezone.utc),
        )

        # 6. Emit forecast.result event
        self._db.emit_event(
            kind="forecast.result",
            firm_id=input.firm_id,
            actor_user_id=input.actor_user_id,
            payload=artifact.model_dump(mode="json", exclude={"generated_at"}),
        )

        return artifact


def _symbol_from_dsl(dsl: dict) -> str:
    universe = (dsl.get("selection") or {}).get("universe") or []
    if not universe:
        raise ValueError("DSL.selection.universe must contain at least one ticker")
    return universe[0]


def _calibrated_params_to_dict(params: CalibratedParams) -> dict:
    # iv_surface_t0 keys are tuples — stringify for JSON
    surface = {f"{k[0]:.4f}_{k[1]}": v for k, v in params.iv_surface_t0.items()}
    return {
        "mu_spot": params.mu_spot, "sigma_spot": params.sigma_spot,
        "iv_surface_t0": surface, "iv_atm_t0": params.iv_atm_t0,
        "iv_kappa": params.iv_kappa, "iv_theta": params.iv_theta,
        "iv_sigma_v": params.iv_sigma_v, "iv_rho": params.iv_rho,
        "lookback_windows": [
            [w[0].isoformat(), w[1].isoformat()]
            for w in params.lookback_windows
        ],
        "calibration_source": params.calibration_source,
        "embedding_query_period": (
            [params.embedding_query_period[0].isoformat(),
             params.embedding_query_period[1].isoformat()]
            if params.embedding_query_period else None
        ),
    }


def _t0_market_context_to_dict(ctx: T0MarketContext) -> dict:
    d = asdict(ctx)
    # tuples → lists for JSON
    for k in ("embedding_period_d", "embedding_period_w"):
        if d[k] is not None:
            d[k] = [d[k][0].isoformat(), d[k][1].isoformat()]
    return d
