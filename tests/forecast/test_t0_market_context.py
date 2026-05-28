"""T0MarketContext builder tests."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from xyz.forecast.t0_context import build_t0_market_context


def _stub_db(*, with_market_emb_day=True, with_market_emb_week=True):
    """Stub db with the helper methods t0_context needs."""
    db = MagicMock()
    # Daily-resampled OHLCV+vol DataFrame ending at t0 — 252 rows
    rng = np.random.default_rng(1)
    n = 252
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="B").astype("int64") // 10**9,
        "open":   close * 0.999,
        "high":   close * 1.005,
        "low":    close * 0.995,
        "close":  close,
        "volume": rng.uniform(1e6, 5e6, n),
    })
    db.get_daily_ohlcv_df = MagicMock(return_value=df)

    if with_market_emb_day:
        db.get_market_emb_day_at = MagicMock(return_value={
            "market_summary": "AAPL is in a Strong Uptrend with Medium volatility.",
            "trend_strength": "strong_uptrend",
            "volatility_regime": "medium",
            "momentum_phase": "bullish",
            "technical_signals": "rsi_overbought,macd_bullish",
            "risk_level": "medium",
            "news_headlines": "AAPL beat Q3 estimates",
            "news_flags": "earnings_beat",
            "period_start": date(2025, 12, 30),
            "period_end": date(2025, 12, 31),
        })
    else:
        db.get_market_emb_day_at = MagicMock(return_value=None)

    if with_market_emb_week:
        db.get_market_emb_week_at = MagicMock(return_value={
            "market_summary": "Broader weekly setup constructive.",
            "news_flags": "fed_minutes",
            "period_start": date(2025, 12, 22),
            "period_end": date(2025, 12, 29),
        })
    else:
        db.get_market_emb_week_at = MagicMock(return_value=None)

    return db


def test_numeric_indicators_present():
    db = _stub_db()
    ctx = build_t0_market_context(symbol="AAPL", t0=date(2025, 12, 31), db=db)
    assert isinstance(ctx.realized_volatility, float)
    assert isinstance(ctx.rsi, float)
    assert isinstance(ctx.macd_hist, float)
    assert isinstance(ctx.max_drawdown_trailing, float)
    assert isinstance(ctx.var_5pct_historical, float)
    assert isinstance(ctx.cvar_5pct_historical, float)


def test_categorical_labels_match_market_emb_day_when_available():
    db = _stub_db(with_market_emb_day=True)
    ctx = build_t0_market_context(symbol="AAPL", t0=date(2025, 12, 31), db=db)
    assert ctx.trend_strength == "strong_uptrend"
    assert ctx.volatility_regime == "medium"
    assert ctx.momentum_phase == "bullish"
    assert "rsi_overbought" in ctx.technical_signals
    assert "macd_bullish" in ctx.technical_signals
    assert ctx.risk_level == "medium"


def test_falls_back_to_classify_when_market_emb_day_missing():
    db = _stub_db(with_market_emb_day=False)
    ctx = build_t0_market_context(symbol="AAPL", t0=date(2025, 12, 31), db=db)
    # Categorical fields populated by calling classify_* on the daily row
    assert ctx.trend_strength.endswith("trend")
    assert ctx.volatility_regime in {"very_low", "low", "medium", "high", "very_high"}
    assert ctx.momentum_phase in {
        "extremely_overbought", "overbought", "bullish", "neutral",
        "bearish", "oversold", "extremely_oversold",
    }
    # No MarketEmbDay → period is None
    assert ctx.embedding_period_d is None


def test_weekly_context_from_market_emb_week():
    db = _stub_db(with_market_emb_week=True)
    ctx = build_t0_market_context(symbol="AAPL", t0=date(2025, 12, 31), db=db)
    assert "constructive" in (ctx.weekly_summary or "")
    assert "fed_minutes" in ctx.weekly_news_flags
    assert ctx.embedding_period_w == (date(2025, 12, 22), date(2025, 12, 29))


def test_weekly_context_optional():
    db = _stub_db(with_market_emb_week=False)
    ctx = build_t0_market_context(symbol="AAPL", t0=date(2025, 12, 31), db=db)
    assert ctx.weekly_summary is None
    assert ctx.weekly_news_flags == []
    assert ctx.embedding_period_w is None
