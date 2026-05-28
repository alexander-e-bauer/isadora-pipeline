"""SHA-256 content hash for ForecastArtifact.

Spans inputs (DSL, t0, horizon, N, seed, calibrated params,
t0_market_context) AND outputs (results) at 6-decimal precision.

Excluded: generated_at timestamp, research_artifact_id (link),
observability metadata (underscore-prefixed keys, stripped by callers
before passing into here).
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any

from xyz.common.canonical import quantize_floats


def compute_forecast_content_hash(
    *,
    dsl: dict,
    t0: date,
    horizon_days: int,
    n_paths: int,
    forecast_seed: int,
    calibrated_params: dict,
    t0_market_context: dict,
    results: dict,
) -> str:
    canonical: dict[str, Any] = {
        "dsl": quantize_floats(dsl),
        "t0": t0.isoformat(),
        "horizon_days": horizon_days,
        "n_paths": n_paths,
        "forecast_seed": forecast_seed,
        "calibrated_params": quantize_floats(calibrated_params),
        "t0_market_context": quantize_floats(t0_market_context),
        "results": quantize_floats(results),
    }
    payload = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
