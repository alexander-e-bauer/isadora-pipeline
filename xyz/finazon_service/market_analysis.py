import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from config import OAI, logger
from typing import List, Dict, Any, Optional
from xyz.llm.embedding_generator import get_embedding

def safe_get(data, key, default):
    val = data.get(key)
    return default if val is None else val

# 1. --- Metric Window Retrieval ---

def get_metrics_window(session, ticker, timestamp, days=20):
    """
    Get metrics for a window of days before the given timestamp.
    """
    from xyz.finazon_service.sql_service import ComputedMetrics, HistoricalData, Ticker
    start_date = timestamp - timedelta(days=days)
    results = (
        session.query(ComputedMetrics, HistoricalData)
        .join(HistoricalData, ComputedMetrics.historical_data_id == HistoricalData.id)
        .filter(
            HistoricalData.ticker_id == ticker.id,
            HistoricalData.timestamp >= start_date,
            HistoricalData.timestamp <= timestamp
        )
        .order_by(HistoricalData.timestamp.desc())
        .all()
    )
    window_data = []
    for computed_metrics, historical_data in results:
        data_dict = {
            'timestamp': historical_data.timestamp,
            'close': historical_data.close,
            'volume': historical_data.volume,
            'sma': computed_metrics.sma,
            'ema': computed_metrics.ema,
            'rsi': computed_metrics.rsi,
            'macd': computed_metrics.macd,
            'adx': computed_metrics.adx,
            'volatility_30': computed_metrics.volatility_30,
            'stoch': computed_metrics.stoch,
            'willr': computed_metrics.willr,
            'cci': computed_metrics.cci,
            'mfi': computed_metrics.mfi
        }
        window_data.append(data_dict)
    return window_data

# 2. --- Market State Classification ---

def classify_trend_strength(data_dict: Dict[str, Any]) -> str:
    sma = safe_get(data_dict, 'sma', 0)
    ema = safe_get(data_dict, 'ema', 0)
    adx = safe_get(data_dict, 'adx', 0)
    direction = 'up' if ema > sma else 'down'
    if adx > 40:
        strength = 'very_strong'
    elif adx > 25:
        strength = 'strong'
    elif adx > 15:
        strength = 'moderate'
    else:
        strength = 'weak'
    return f"{strength}_{direction}trend"

def classify_volatility_regime(data_dict: Dict[str, Any]) -> str:
    volatility = safe_get(data_dict, 'volatility_30', 0)
    if volatility < 0.10:
        return 'very_low'
    elif volatility < 0.20:
        return 'low'
    elif volatility < 0.35:
        return 'medium'
    elif volatility < 0.50:
        return 'high'
    else:
        return 'very_high'

def classify_momentum_phase(data_dict: Dict[str, Any]) -> str:
    rsi = safe_get(data_dict, 'rsi', 50)
    stoch = safe_get(data_dict, 'stoch', 50)

    avg_momentum = (rsi + stoch) / 2
    if avg_momentum > 80:
        return 'extremely_overbought'
    elif avg_momentum > 70:
        return 'overbought'
    elif avg_momentum > 60:
        return 'bullish'
    elif avg_momentum > 40:
        return 'neutral'
    elif avg_momentum > 30:
        return 'bearish'
    elif avg_momentum > 20:
        return 'oversold'
    else:
        return 'extremely_oversold'

def identify_technical_signals(data_dict: Dict[str, Any]) -> str:
    signals = []
    rsi = safe_get(data_dict, 'rsi', 50)
    macd = safe_get(data_dict, 'macd', 0)
    willr = safe_get(data_dict, 'willr', -50)
    cci = safe_get(data_dict, 'cci', 0)

    if rsi > 70:
        signals.append('rsi_overbought')
    elif rsi < 30:
        signals.append('rsi_oversold')
    if macd > 0:
        signals.append('macd_bullish')
    else:
        signals.append('macd_bearish')
    if willr > -20:
        signals.append('willr_overbought')
    elif willr < -80:
        signals.append('willr_oversold')
    if cci > 100:
        signals.append('cci_overbought')
    elif cci < -100:
        signals.append('cci_oversold')
    return ','.join(signals) if signals else 'neutral'

def assess_risk_level(data_dict: Dict[str, Any]) -> str:
    volatility = safe_get(data_dict, 'volatility_30', 0)
    rsi = safe_get(data_dict, 'rsi', 50)
    adx = safe_get(data_dict, 'adx', 0)
    risk_score = 0
    if volatility > 0.4:
        risk_score += 3
    elif volatility > 0.25:
        risk_score += 2
    elif volatility > 0.15:
        risk_score += 1
    if rsi > 80 or rsi < 20:
        risk_score += 2
    elif rsi > 70 or rsi < 30:
        risk_score += 1
    if adx > 40:
        risk_score += 1
    if risk_score >= 4:
        return 'very_high'
    elif risk_score >= 3:
        return 'high'
    elif risk_score >= 2:
        return 'medium'
    elif risk_score >= 1:
        return 'low'
    else:
        return 'very_low'

# 3. --- Market State and Embedding Context Generation ---

def create_market_state_from_metrics(
    session,
    ticker_symbol: str,
    row_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Convert raw metrics into semantic market state dict.
    """
    from xyz.finazon_service.sql_service import Ticker
    ticker = session.query(Ticker).filter_by(symbol=ticker_symbol).first()
    if not ticker:
        logger.error(f"Ticker {ticker_symbol} not found in database")
        return None
    # Ensure dict format
    data_dict = row_data.to_dict() if hasattr(row_data, 'to_dict') else row_data
    timestamp = data_dict.get('timestamp', datetime.now())
    # Optionally: window_data = get_metrics_window(session, ticker, timestamp, days=20)
    trend_strength = classify_trend_strength(data_dict)
    volatility_regime = classify_volatility_regime(data_dict)
    momentum_phase = classify_momentum_phase(data_dict)
    technical_signals = identify_technical_signals(data_dict)
    risk_level = assess_risk_level(data_dict)
    rsi = data_dict.get('rsi', 50)
    if rsi > 70:
        market_position = 'overbought'
    elif rsi < 30:
        market_position = 'oversold'
    else:
        market_position = 'neutral'
    # Human-readable summary for quick display
    summary = generate_market_summary(
        ticker_symbol, trend_strength, volatility_regime,
        momentum_phase, market_position, risk_level
    )
    return {
        'trend_strength': trend_strength,
        'volatility_regime': volatility_regime,
        'momentum_phase': momentum_phase,
        'technical_signals': technical_signals,
        'market_position': market_position,
        'risk_level': risk_level,
        'market_summary': summary
    }

def generate_market_summary(
    ticker: str,
    trend: str,
    volatility: str,
    momentum: str,
    position: str,
    risk: str
) -> str:
    trend_clean = trend.replace('_', ' ').title()
    volatility_clean = volatility.replace('_', ' ').title()
    momentum_clean = momentum.replace('_', ' ').title()
    summary = f"{ticker} is currently in a {trend_clean} with {volatility_clean} volatility. "
    summary += f"Momentum indicators show {momentum_clean} conditions. "
    summary += f"The stock appears {position} with {risk} risk level."
    return summary

def create_embedding_text(market_state_data: Dict[str, Any], ticker_symbol: str) -> str:
    """
    Generate a detailed text string for semantic embedding.
    """
    return (
        f"Ticker: {ticker_symbol}\n"
        f"Trend Strength: {market_state_data.get('trend_strength')}\n"
        f"Volatility Regime: {market_state_data.get('volatility_regime')}\n"
        f"Momentum Phase: {market_state_data.get('momentum_phase')}\n"
        f"Technical Signals: {market_state_data.get('technical_signals')}\n"
        f"Market Position: {market_state_data.get('market_position')}\n"
        f"Risk Level: {market_state_data.get('risk_level')}\n"
        f"Summary: {market_state_data.get('market_summary')}"
    ).strip()

def create_openai_embedding_from_market_state(market_state_data: Dict[str, Any], ticker_symbol: str) -> Optional[List[float]]:
    """
    Generate OpenAI embedding vector from market state.
    """
    try:
        embedding_text = create_embedding_text(market_state_data, ticker_symbol)
        embedding_vector = get_embedding(embedding_text)

        return embedding_vector
    except Exception as e:
        logger.error(f"Error creating OpenAI embedding: {e}")
        return None
