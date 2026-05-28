"""Wall-clock perf test — guarded by @pytest.mark.perf; excluded from fast CI."""
from __future__ import annotations

import time
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from xyz.forecast.agent import ForecastAgent
from xyz.forecast.schemas import ForecastInput, ForecastOverrides


@pytest.mark.perf
def test_forecast_wall_clock_under_30s_for_default_size():
    """N=1000 paths × 252 days completes in < 30s on the CI runner."""
    rng = np.random.default_rng(0)
    n = 600
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="B").astype("int64") // 10**9,
        "open": close * 0.999, "high": close * 1.005,
        "low": close * 0.995, "close": close,
        "volume": rng.uniform(1e6, 5e6, n),
    })
    db = MagicMock()
    db.get_daily_ohlcv_df = MagicMock(return_value=df)
    db.get_daily_log_returns = MagicMock(return_value=np.diff(np.log(close)))
    db.get_atm_iv_series = MagicMock(return_value=np.full(n, 0.20))
    db.get_t0_iv_surface = MagicMock(return_value={(100.0, 30): 0.20})
    db.get_t0_iv_atm = MagicMock(return_value=0.20)
    db.get_t0_realized_volatility = MagicMock(return_value=0.18)
    db.get_market_emb_day_at = MagicMock(return_value=None)
    db.get_market_emb_week_at = MagicMock(return_value=None)
    db.emit_event = MagicMock()

    input_ = ForecastInput(
        strategy_id="s", strategy_version=1, firm_id="f",
        dsl={
            "template": "covered_call",
            "selection": {"universe": ["AAPL"]},
            "action": {"delta_short_target": 0.30, "dte_min": 25, "dte_max": 45},
            "exit": {"max_profit_pct_close": 0.50},
        },
        t0=date(2025, 12, 31),
        horizon_days=252, n_paths=1000, forecast_seed=42,
        overrides=ForecastOverrides(calibration_source="recent_window"),
    )

    start = time.time()
    ForecastAgent(db=db).run(input=input_)
    elapsed = time.time() - start
    assert elapsed < 30.0, f"forecast took {elapsed:.1f}s, exceeds 30s budget"
