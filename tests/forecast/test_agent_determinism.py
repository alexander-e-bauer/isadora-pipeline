"""End-to-end determinism: same inputs → byte-identical content_hash."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from xyz.forecast.agent import ForecastAgent
from xyz.forecast.schemas import ForecastInput, ForecastOverrides


def _wire_fake_db():
    """Return a db object with all helpers the agent transitively needs."""
    rng = np.random.default_rng(0)
    n = 500
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="B").astype("int64") // 10**9,
        "open":   close * 0.999, "high": close * 1.005,
        "low":    close * 0.995, "close": close,
        "volume": rng.uniform(1e6, 5e6, n),
    })

    db = MagicMock()
    db.get_daily_ohlcv_df             = MagicMock(return_value=df)
    db.get_daily_log_returns          = MagicMock(return_value=np.diff(np.log(close)))
    db.get_atm_iv_series              = MagicMock(return_value=np.full(n, 0.20))
    db.get_t0_iv_surface              = MagicMock(return_value={(100.0, 30): 0.20})
    db.get_t0_iv_atm                  = MagicMock(return_value=0.20)
    db.get_t0_realized_volatility     = MagicMock(return_value=0.18)
    db.get_market_emb_day_at          = MagicMock(return_value=None)
    db.get_market_emb_week_at         = MagicMock(return_value=None)
    db.emit_event                     = MagicMock()
    return db


def _make_input(seed: int = 42) -> ForecastInput:
    return ForecastInput(
        strategy_id="strat-1", strategy_version=1, firm_id="firm-1",
        actor_user_id="user-1",
        dsl={
            "template": "covered_call",
            "selection": {"universe": ["AAPL"]},
            "action": {"delta_short_target": 0.30, "dte_min": 25, "dte_max": 45},
            "exit": {"max_profit_pct_close": 0.50},
            "risk_box": {},
        },
        t0=date(2025, 12, 31),
        horizon_days=30,
        n_paths=200,
        forecast_seed=seed,
        overrides=ForecastOverrides(calibration_source="recent_window"),
    )


def test_same_inputs_produce_byte_identical_artifact():
    db = _wire_fake_db()
    agent = ForecastAgent(db=db)
    a = agent.run(input=_make_input(seed=42))
    b = agent.run(input=_make_input(seed=42))
    assert a.content_hash == b.content_hash
    assert a.nav_bands == b.nav_bands
    assert a.terminal_stats == b.terminal_stats
    # generated_at differs but is excluded from hash
    assert a.generated_at != b.generated_at


def test_forecast_result_event_emitted():
    db = _wire_fake_db()
    ForecastAgent(db=db).run(input=_make_input(seed=42))
    # Exactly one forecast.result event emitted
    emit_calls = [c for c in db.emit_event.call_args_list
                  if c.kwargs.get("kind") == "forecast.result"]
    assert len(emit_calls) == 1
