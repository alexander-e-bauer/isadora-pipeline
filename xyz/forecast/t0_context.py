"""T0MarketContext builder — snapshot of market state at proposal date.

Combines:
  - Numeric indicators from xyz.finazon_service.metrics.compute_batch_metrics
    on daily-resampled historical_data.
  - Categorical labels via the classify_* functions in
    xyz.finazon_service.market_analysis (preferring cached values
    from MarketEmbDay if available).
  - News context lifted directly from MarketEmbDay / MarketEmbWeek.

All values are pure functions of the t0 metrics row → deterministic
and reproducible, even if the underlying classifier rules change later.
"""
from __future__ import annotations

from datetime import date

from xyz.finazon_service.market_analysis import (
    assess_risk_level,
    classify_momentum_phase,
    classify_trend_strength,
    classify_volatility_regime,
    identify_technical_signals,
)
from xyz.finazon_service.metrics import compute_batch_metrics
from xyz.forecast.schemas import T0MarketContext


def build_t0_market_context(*, symbol: str, t0: date, db) -> T0MarketContext:
    df = db.get_daily_ohlcv_df(symbol=symbol, t0=t0, days=252)
    metrics_df = compute_batch_metrics(
        df,
        ma_windows=(20,),
        vol_window=20,
        rsi_window=14,
    )
    # Last row = t0
    row = metrics_df.iloc[-1].to_dict()

    # MarketEmbDay supplies the cached categorical labels + news at t0.
    emb_day = db.get_market_emb_day_at(symbol=symbol, t0=t0)
    if emb_day is not None:
        trend_strength    = emb_day["trend_strength"]
        volatility_regime = emb_day["volatility_regime"]
        momentum_phase    = emb_day["momentum_phase"]
        technical_signals = [s for s in emb_day["technical_signals"].split(",") if s]
        risk_level        = emb_day["risk_level"]
        news_at_t0_summary = emb_day.get("news_headlines")
        news_flags_at_t0 = [
            f for f in (emb_day.get("news_flags") or "").split(",") if f
        ]
        embedding_period_d = (emb_day["period_start"], emb_day["period_end"])
    else:
        # Fall back to live classification on the t0 metrics row.
        trend_strength    = classify_trend_strength(row)
        volatility_regime = classify_volatility_regime(row)
        momentum_phase    = classify_momentum_phase(row)
        technical_signals = [s for s in identify_technical_signals(row).split(",") if s]
        risk_level        = assess_risk_level(row)
        news_at_t0_summary = None
        news_flags_at_t0 = []
        embedding_period_d = None

    emb_week = db.get_market_emb_week_at(symbol=symbol, t0=t0)
    if emb_week is not None:
        weekly_summary    = emb_week["market_summary"]
        weekly_news_flags = [
            f for f in (emb_week.get("news_flags") or "").split(",") if f
        ]
        embedding_period_w = (emb_week["period_start"], emb_week["period_end"])
    else:
        weekly_summary    = None
        weekly_news_flags = []
        embedding_period_w = None

    return T0MarketContext(
        realized_volatility       = float(row.get("realized_volatility") or 0.0),
        historical_volatility     = float(row.get("historical_volatility") or 0.0),
        var_5pct_historical       = float(row.get("var") or 0.0),
        cvar_5pct_historical      = float(row.get("cvar") or 0.0),
        rsi                       = float(row.get("rsi") or 50.0),
        adx                       = float(row.get("adx") or 0.0),
        bollinger_width           = float(row.get("bollinger_width") or 0.0),
        macd_hist                 = float(row.get("macd_hist") or 0.0),
        z_score                   = float(row.get("z_score") or 0.0),
        ewma_score                = float(row.get("ewma_score") or 0.0),
        sharpe_ratio_trailing     = float(row.get("sharpe_ratio") or 0.0),
        sortino_ratio_trailing    = float(row.get("sortino_ratio") or 0.0),
        max_drawdown_trailing     = float(row.get("max_drawdown") or 0.0),
        trend_strength            = trend_strength,
        volatility_regime         = volatility_regime,
        momentum_phase            = momentum_phase,
        technical_signals         = technical_signals,
        risk_level                = risk_level,
        news_at_t0_summary        = news_at_t0_summary,
        news_flags_at_t0          = news_flags_at_t0,
        weekly_summary            = weekly_summary,
        weekly_news_flags         = weekly_news_flags,
        computed_from             = "daily_bars_252",
        embedding_period_d        = embedding_period_d,
        embedding_period_w        = embedding_period_w,
    )
