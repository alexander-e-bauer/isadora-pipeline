# xyz/finazon_service/forecast_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import traceback
from config import logger

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
    logger.info("Prophet successfully imported")
except ImportError:
    logger.warning("Prophet not installed. Install with: pip install prophet")
    PROPHET_AVAILABLE = False

from xyz.finazon_service.sql_service import (
    get_db_session, Ticker, HistoricalData, ComputedMetrics,
    ForecastMetrics, ModelPerformance, store_forecast_metrics
)


class ForecastEngine:
    def __init__(self):
        # Fix: Remove the incorrect reference to _prophet_forecast
        self.models = {
            'prophet': 'prophet' if PROPHET_AVAILABLE else None,
        }
        logger.info(f"ForecastEngine initialized. Prophet available: {PROPHET_AVAILABLE}")

    def generate_forecasts_for_ticker(self, ticker_symbol: str, horizons: List[int] = [1, 7, 30]):
        """Generate forecasts for a specific ticker"""
        if not PROPHET_AVAILABLE:
            logger.error("Prophet not available for forecasting")
            return False

        try:
            with get_db_session() as session:
                ticker = session.query(Ticker).filter_by(symbol=ticker_symbol).first()
                if not ticker:
                    logger.error(f"Ticker {ticker_symbol} not found")
                    return False

                logger.info(f"Generating forecasts for {ticker_symbol}")

                # Generate forecasts for each horizon
                for horizon in horizons:
                    try:
                        self._generate_forecast(session, ticker, horizon, 'prophet')
                        logger.info(f"Generated {horizon}-day forecast for {ticker_symbol}")
                    except Exception as e:
                        logger.error(f"Failed to generate {horizon}-day forecast for {ticker_symbol}: {e}")

                return True

        except Exception as e:
            logger.error(f"Error generating forecasts for {ticker_symbol}: {e}")
            traceback.print_exc()
            return False

    def generate_daily_forecasts(self, ticker_symbols: List[str] = None):
        """Run daily forecast generation for specified tickers or all active tickers"""
        if not PROPHET_AVAILABLE:
            logger.error("Prophet not available for forecasting")
            return

        try:
            with get_db_session() as session:
                if ticker_symbols:
                    tickers = session.query(Ticker).filter(Ticker.symbol.in_(ticker_symbols)).all()
                else:
                    # Get tickers that have recent data (within last 7 days)
                    recent_cutoff = datetime.utcnow() - timedelta(days=7)
                    recent_timestamp = int(recent_cutoff.timestamp())

                    tickers = session.query(Ticker).join(HistoricalData).filter(
                        HistoricalData.timestamp >= recent_timestamp
                    ).distinct().all()

                logger.info(f"Generating forecasts for {len(tickers)} tickers")

                for ticker in tickers:
                    try:
                        # Generate forecasts for different horizons
                        for horizon in [1, 7, 30]:
                            self._generate_forecast(session, ticker, horizon, 'prophet')

                        # Evaluate model performance weekly (on Mondays)
                        if datetime.now().weekday() == 0:
                            self._evaluate_model_performance(session, ticker)

                        logger.info(f"Completed forecasts for {ticker.symbol}")

                    except Exception as e:
                        logger.error(f"Forecast failed for {ticker.symbol}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error in daily forecast generation: {e}")
            traceback.print_exc()

    def _generate_forecast(self, session, ticker: Ticker, horizon_days: int, model_type: str):
        """Generate Prophet forecast for a specific ticker and horizon"""
        try:
            logger.info(f"Starting forecast generation for {ticker.symbol}, horizon: {horizon_days} days")

            # Get historical data with computed metrics using ORM
            query_result = session.query(
                HistoricalData.timestamp,
                HistoricalData.close,
                HistoricalData.volume,
                HistoricalData.open,
                HistoricalData.high,
                HistoricalData.low,
                ComputedMetrics.volatility,
                ComputedMetrics.rsi,
                ComputedMetrics.macd,
                ComputedMetrics.bollinger_width,
                ComputedMetrics.historical_volatility
            ).outerjoin(
                ComputedMetrics, HistoricalData.id == ComputedMetrics.historical_data_id
            ).filter(
                HistoricalData.ticker_id == ticker.id,
                HistoricalData.close.isnot(None)
            ).order_by(HistoricalData.timestamp).all()

            if len(query_result) < 30:  # Need minimum data points
                logger.warning(f"Insufficient data for {ticker.symbol}: {len(query_result)} points")
                return

            logger.info(f"Retrieved {len(query_result)} data points for {ticker.symbol}")

            # Convert to DataFrame
            df = pd.DataFrame(query_result, columns=[
                'timestamp', 'close', 'volume', 'open', 'high', 'low',
                'volatility', 'rsi', 'macd', 'bollinger_width', 'historical_volatility'
            ])

            # Prepare data for Prophet
            df['ds'] = pd.to_datetime(df['timestamp'], unit='s')
            df['y'] = df['close']

            # Remove any duplicates and sort
            df = df.drop_duplicates(subset=['ds']).sort_values('ds')

            # Remove any infinite or extremely large values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=['ds', 'y'])

            if len(df) < 30:
                logger.warning(f"Insufficient clean data for {ticker.symbol}: {len(df)} points after cleaning")
                return

            logger.info(f"Using {len(df)} clean data points for {ticker.symbol}")

            # Initialize Prophet with financial market parameters
            model = Prophet(
                daily_seasonality=False,  # Financial markets don't have strong daily seasonality
                weekly_seasonality=True,  # Markets do have weekly patterns
                yearly_seasonality=True,  # Long-term cycles
                changepoint_prior_scale=0.05,  # Conservative for financial volatility
                seasonality_prior_scale=10,
                holidays_prior_scale=10,
                interval_width=0.95,
                changepoint_range=0.8  # Allow changepoints in 80% of history
            )

            # Add regressors if available and have sufficient non-null values
            regressor_columns = ['volume', 'volatility', 'rsi', 'macd', 'bollinger_width']
            added_regressors = []

            for col in regressor_columns:
                if col in df.columns and df[col].notna().sum() > len(df) * 0.5:  # 50% non-null
                    try:
                        # Clean the regressor data
                        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                        # Fill NaN values with column mean
                        df[col] = df[col].fillna(df[col].mean())
                        model.add_regressor(col)
                        added_regressors.append(col)
                        logger.info(f"Added regressor: {col}")
                    except Exception as e:
                        logger.warning(f"Could not add regressor {col}: {e}")

            # Prepare training data
            train_cols = ['ds', 'y'] + added_regressors
            train_data = df[train_cols].dropna()

            if len(train_data) < 30:
                logger.warning(
                    f"Insufficient clean data for {ticker.symbol}: {len(train_data)} points after regressor cleaning")
                return

            # Fit the model
            logger.info(f"Training Prophet model for {ticker.symbol} with {len(train_data)} data points")
            model.fit(train_data)

            # Create future dataframe
            future = model.make_future_dataframe(periods=horizon_days, freq='D')

            # Add regressor columns to future dataframe
            for col in added_regressors:
                if col in df.columns:
                    # Calculate recent average value for this regressor
                    recent_value = df[col].tail(30).mean()
                    if np.isnan(recent_value):
                        recent_value = df[col].mean()
                    if np.isnan(recent_value):
                        recent_value = 0

                    # Initialize the column with recent average
                    future[col] = recent_value

                    # For historical dates, try to use actual values from training data
                    for idx, future_row in future.iterrows():
                        # Find matching date in training data
                        matching_train = train_data[train_data['ds'] == future_row['ds']]
                        if not matching_train.empty and col in matching_train.columns:
                            actual_value = matching_train[col].iloc[0]
                            if not np.isnan(actual_value):
                                future.loc[idx, col] = actual_value

            # Generate forecast
            logger.info(f"Generating forecast for {ticker.symbol}")
            forecast = model.predict(future)

            # Store forecast results
            forecast_date = datetime.utcnow()
            forecast_records = []

            # Only store future predictions (not historical)
            future_forecast = forecast.tail(horizon_days)

            for i, (_, row) in enumerate(future_forecast.iterrows(), 1):
                target_date = forecast_date + timedelta(days=i)

                # Determine trend direction
                trend_direction = self._determine_trend_direction(row)

                forecast_record = {
                    'forecast_date': forecast_date,
                    'target_date': target_date,
                    'forecast_horizon': i,
                    'predicted_close': float(row['yhat']),
                    'confidence_lower': float(row['yhat_lower']),
                    'confidence_upper': float(row['yhat_upper']),
                    'model_version': '1.0',
                    'predicted_volatility': float(abs(row['yhat_upper'] - row['yhat_lower']) / row['yhat']) if row[
                                                                                                                   'yhat'] != 0 else 0,
                    'trend_direction': trend_direction,
                    'trend_strength': float(row.get('trend', 0)),
                    'seasonal_component': float(row.get('weekly', 0) + row.get('yearly', 0))
                }

                forecast_records.append(forecast_record)

            # Store in database
            store_forecast_metrics(ticker.id, forecast_records, model_type)

            logger.info(f"Successfully stored {len(forecast_records)} forecast points for {ticker.symbol}")

        except Exception as e:
            logger.error(f"Error generating forecast for {ticker.symbol}: {e}")
            traceback.print_exc()
            raise

    def _determine_trend_direction(self, forecast_row) -> str:
        """Determine trend direction from Prophet forecast components"""
        try:
            trend = forecast_row.get('trend', 0)
            yhat = forecast_row.get('yhat', 0)

            # Calculate trend as percentage change
            if yhat != 0:
                trend_pct = (trend / yhat) * 100

                if trend_pct > 2:
                    return 'bullish'
                elif trend_pct < -2:
                    return 'bearish'
                else:
                    return 'neutral'
            else:
                return 'neutral'

        except Exception:
            return 'neutral'

    def _evaluate_model_performance(self, session, ticker: Ticker):
        """Evaluate forecast accuracy against actual prices"""
        try:
            # Get forecasts made 7 days ago
            evaluation_date = datetime.utcnow() - timedelta(days=7)

            forecasts = session.query(ForecastMetrics).filter(
                ForecastMetrics.ticker_id == ticker.id,
                ForecastMetrics.forecast_date.between(
                    evaluation_date - timedelta(hours=12),
                    evaluation_date + timedelta(hours=12)
                )
            ).all()

            if not forecasts:
                logger.info(f"No forecasts to evaluate for {ticker.symbol}")
                return

            # Get actual prices for the forecasted dates
            actual_prices = {}
            for forecast in forecasts:
                target_timestamp = int(forecast.target_date.timestamp())

                # Find closest actual price within 4 hours
                actual_data = session.query(HistoricalData).filter(
                    HistoricalData.ticker_id == ticker.id,
                    HistoricalData.timestamp.between(
                        target_timestamp - 14400,  # 4 hours before
                        target_timestamp + 14400  # 4 hours after
                    )
                ).order_by(
                    (HistoricalData.timestamp - target_timestamp).label('time_diff')
                ).first()

                if actual_data:
                    actual_prices[forecast.id] = actual_data.close

            # Calculate performance metrics
            errors = []
            directional_correct = 0
            total_directional = 0

            for forecast in forecasts:
                if forecast.id in actual_prices:
                    actual = actual_prices[forecast.id]
                    predicted = forecast.predicted_close

                    # MAPE calculation
                    if actual != 0:
                        error = abs((actual - predicted) / actual)
                        errors.append(error)

                    # Directional accuracy (for multi-day forecasts)
                    if forecast.forecast_horizon > 1:
                        # Get previous actual price for direction comparison
                        prev_timestamp = int((forecast.target_date - timedelta(days=1)).timestamp())
                        prev_data = session.query(HistoricalData).filter(
                            HistoricalData.ticker_id == ticker.id,
                            HistoricalData.timestamp <= prev_timestamp
                        ).order_by(HistoricalData.timestamp.desc()).first()

                        if prev_data:
                            actual_direction = 'up' if actual > prev_data.close else 'down'
                            predicted_direction = 'up' if predicted > prev_data.close else 'down'

                            if actual_direction == predicted_direction:
                                directional_correct += 1
                            total_directional += 1

            if errors:
                mape = np.mean(errors) * 100
                rmse = np.sqrt(np.mean([(actual_prices[f.id] - f.predicted_close) ** 2
                                        for f in forecasts if f.id in actual_prices]))
                mae = np.mean([abs(actual_prices[f.id] - f.predicted_close)
                               for f in forecasts if f.id in actual_prices])

                directional_accuracy = (
                            directional_correct / total_directional * 100) if total_directional > 0 else None

                # Store performance metrics
                performance = ModelPerformance(
                    ticker_id=ticker.id,
                    model_type='prophet',
                    evaluation_date=datetime.utcnow(),
                    mape=mape,
                    rmse=rmse,
                    mae=mae,
                    directional_accuracy=directional_accuracy,
                    backtest_start=evaluation_date - timedelta(days=30),
                    backtest_end=evaluation_date
                )

                session.add(performance)
                session.commit()

                logger.info(f"Model performance for {ticker.symbol}: MAPE={mape:.2f}%, "
                            f"RMSE={rmse:.2f}, Directional Accuracy={directional_accuracy:.1f}%")

        except Exception as e:
            logger.error(f"Error evaluating model performance for {ticker.symbol}: {e}")
            traceback.print_exc()


# Convenience function for external use
def generate_forecasts_for_symbols(symbols: List[str]):
    """Generate forecasts for specific ticker symbols"""
    engine = ForecastEngine()
    for symbol in symbols:
        engine.generate_forecasts_for_ticker(symbol)


def run_daily_forecast_pipeline():
    """Run the daily forecast pipeline for all active tickers"""
    engine = ForecastEngine()
    engine.generate_daily_forecasts()
