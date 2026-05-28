"""Schemas for the ForecastAgent (engine-side).

Three layers:
  - Pydantic models for HTTP transport (ForecastOverrides, ForecastInput).
  - Dataclasses for in-process state (CalibratedParams, T0MarketContext).
  - Pydantic model for the persisted artifact (ForecastArtifact) so it
    can be (de)serialized for emit + transport to server.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# DSL override block (lives inside ForecastInput)
# ---------------------------------------------------------------------------

class ForecastOverrides(BaseModel):
    """Optional knobs the advisor (or stress-test workflow) can set."""
    model_config = ConfigDict(extra="forbid")

    lookback_days: int | None = None             # default 252 if None
    k_similar_windows: int | None = None         # default 3 if None
    sigma_spot_mult: float = 1.0                 # multiplicative
    sigma_iv_mult: float = 1.0                   # multiplicative
    drift_override: float | None = None          # absolute; replaces calibrated μ
    rho_override: float | None = None            # absolute; replaces calibrated ρ
    calibration_source: Literal["embedding", "recent_window"] = "embedding"


# ---------------------------------------------------------------------------
# Inbound request — what /agents/forecast accepts
# ---------------------------------------------------------------------------

class ForecastInput(BaseModel):
    """POST /agents/forecast request body."""
    model_config = ConfigDict(extra="forbid")

    strategy_id: str
    strategy_version: int
    firm_id: str
    actor_user_id: str | None = None
    dsl: dict
    t0: date
    horizon_days: int = 252
    n_paths: int = Field(default=1000, ge=10, le=10_000)
    forecast_seed: int
    overrides: ForecastOverrides = Field(default_factory=ForecastOverrides)
    research_artifact_id: str | None = None


# ---------------------------------------------------------------------------
# In-process state — produced by Calibrator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibratedParams:
    # GBM on log-spot
    mu_spot: float
    sigma_spot: float
    # IV surface at t0 (skew shape preserved; level evolves stochastically)
    iv_surface_t0: dict[tuple[float, int], float]   # {(strike, dte): iv}
    iv_atm_t0: float
    # OU on log-IV
    iv_kappa: float
    iv_theta: float
    iv_sigma_v: float
    iv_rho: float
    # Provenance — frozen for hash determinism
    lookback_windows: list[tuple[date, date]]
    calibration_source: str                          # "embedding" | "recent_window"
    embedding_query_period: tuple[date, date] | None # MarketEmbDay row keyed to t0


# ---------------------------------------------------------------------------
# In-process state — produced by t0_context
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class T0MarketContext:
    # Numeric indicators (compute_batch_metrics on daily-resampled historical_data)
    realized_volatility: float
    historical_volatility: float
    var_5pct_historical: float
    cvar_5pct_historical: float
    rsi: float
    adx: float
    bollinger_width: float
    macd_hist: float
    z_score: float
    ewma_score: float
    sharpe_ratio_trailing: float
    sortino_ratio_trailing: float
    max_drawdown_trailing: float
    # Categorical regime labels (classify_* from market_analysis.py)
    trend_strength: str
    volatility_regime: str
    momentum_phase: str
    technical_signals: list[str]
    risk_level: str
    # News context at proposal
    news_at_t0_summary: str | None
    news_flags_at_t0: list[str]
    # Weekly broader-regime context
    weekly_summary: str | None
    weekly_news_flags: list[str]
    # Provenance
    computed_from: str                                # e.g. "daily_bars_252"
    embedding_period_d: tuple[date, date] | None
    embedding_period_w: tuple[date, date] | None


# ---------------------------------------------------------------------------
# Persisted artifact — what /agents/forecast returns AND what gets
# embedded in the forecast.result event payload
# ---------------------------------------------------------------------------

class ForecastArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Identity
    strategy_id: str
    strategy_version: int
    firm_id: str

    # Inputs — ALL hashed
    t0: date
    horizon_days: int
    n_paths: int
    forecast_seed: int
    dsl_snapshot: dict
    calibrated_params: dict                           # CalibratedParams dumped to dict
    t0_market_context: dict                           # T0MarketContext dumped to dict
    calibration_source: str

    # Lightweight context link — NOT hashed
    research_artifact_id: str | None

    # Outputs — ALL hashed
    nav_bands: dict[str, list[float]]                 # p10/p25/p50/p75/p90
    spot_bands: dict[str, list[float]]
    iv_bands: dict[str, list[float]]
    terminal_stats: dict                              # mean, stdev, var_p5, cvar_p5,
                                                      # p_touch_drawdown_20pct, n_paths_excluded
    terminal_nav_histogram: dict                      # {"edges": [...], "counts": [...]}
    sample_paths: dict                                # {"nav":[...], "spot":[...], "iv":[...]}

    # Provenance
    content_hash: str
    generated_at: datetime                            # NOT hashed (set after hash is computed)
