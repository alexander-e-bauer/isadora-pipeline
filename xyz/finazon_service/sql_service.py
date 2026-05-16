import psycopg2
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union
import traceback
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey, BigInteger, Date, DateTime, Text, \
    Table, text, Boolean
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship, sessionmaker, declarative_base, Session
from contextlib import contextmanager
from config import DATABASE, logger
import pandas as pd
import numpy as np
from xyz.llm.embedding_generator import get_embedding

import logging
import json
from xyz.finazon_service.api_service import get_new_ticker_gen_data

Base = declarative_base()

# Create SQLAlchemy engine with connection pooling
DATABASE_URI = f"postgresql+psycopg2://{DATABASE.DB_USER}:{DATABASE.DB_PASSWORD}@{DATABASE.DB_HOST}/{DATABASE.DB_NAME}"
engine: Engine = create_engine(
    DATABASE_URI,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=15,  # Increased for production
    max_overflow=25,
    pool_timeout=30,  # Add timeout
    echo=False,  # Set to True for debugging
    connect_args={
        "connect_timeout": 10,
        "application_name": "FinancialDataPipeline"
    }
)

# Association table for a many-to-many relationship between Ticker and Document
ticker_document_association = Table(
    'ticker_document_association',
    Base.metadata,
    Column('ticker_id', Integer, ForeignKey('tickers.id'), primary_key=True),
    Column('document_id', Integer, ForeignKey('documents.id'), primary_key=True)
)


# [Keep all your existing model classes exactly the same - Ticker, HistoricalData, etc.]
class Ticker(Base):
    __tablename__ = 'tickers'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, unique=True, nullable=False)
    company_name = Column(String)
    general_info = Column(String)

    # Relationships with historical data and documents
    historical_data = relationship("HistoricalData", back_populates="ticker")
    documents = relationship("Document", secondary=ticker_document_association, back_populates="tickers")


class HistoricalData(Base):
    __tablename__ = 'historical_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(Integer, ForeignKey('tickers.id'), nullable=True)
    # Using a BigInteger for timestamps if they're provided in UNIX timestamp format.
    timestamp = Column(BigInteger, nullable=True, index=True)
    open = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)

    ticker = relationship("Ticker", back_populates="historical_data")
    # Establish one-to-one relationship with computed metrics
    metrics = relationship("ComputedMetrics", back_populates="historical_data", uselist=False)


class ComputedMetrics(Base):
    __tablename__ = 'computed_metrics'
    id = Column(Integer, primary_key=True, autoincrement=True)
    historical_data_id = Column(Integer, ForeignKey('historical_data.id'), nullable=False, unique=True)

    # Metrics - one column per DataFrame field
    log_return = Column(Float)
    price_change = Column(Float)
    price_change_pct = Column(Float)
    hourly_return = Column(Float)
    volatility = Column(Float)
    typical_price = Column(Float)
    vwap = Column(Float)
    sma_20 = Column(Float)
    ema_20 = Column(Float)
    dema_20 = Column(Float)
    tema_20 = Column(Float)
    wma_20 = Column(Float)
    trima_20 = Column(Float)
    sma = Column(Float)
    ema = Column(Float)
    dema = Column(Float)
    tema = Column(Float)
    wma = Column(Float)
    trima = Column(Float)
    historical_volatility = Column(Float)
    realized_volatility = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    bollinger_upper = Column(Float)
    bollinger_lower = Column(Float)
    bollinger_width = Column(Float)
    obv = Column(Float)
    cmf = Column(Float)
    z_score = Column(Float)
    ewma_score = Column(Float)
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    var = Column(Float)
    cvar = Column(Float)
    roc = Column(Float)
    rsi = Column(Float)
    stoch_k = Column(Float)
    stoch_d = Column(Float)
    stoch = Column(Float)
    stochrsi = Column(Float)
    willr = Column(Float)
    kama = Column(Float)
    mama = Column(Float)
    fama = Column(Float)
    t3 = Column(Float)
    adx = Column(Float)
    adxr = Column(Float)
    apo = Column(Float)
    ppo = Column(Float)
    mom = Column(Float)
    bop = Column(Float)
    cci = Column(Float)
    cmo = Column(Float)
    rocr = Column(Float)
    aroon = Column(Float)
    aroonosc = Column(Float)
    mfi = Column(Float)
    trix = Column(Float)
    ultosc = Column(Float)
    volatility_30 = Column(Float)

    historical_data = relationship("HistoricalData", back_populates="metrics")


# --- Aggregated Market Embedding Tables ---

class MarketEmbThirtyMin(Base):
    __tablename__ = 'market_emb_30min'
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(Integer, ForeignKey('tickers.id'))
    period_start = Column(DateTime, index=True)  # e.g. 2025-05-30 13:00:00
    period_end = Column(DateTime, index=True)
    trend_strength = Column(String(50))
    volatility_regime = Column(String(20))
    momentum_phase = Column(String(20))
    technical_signals = Column(String(100))
    market_position = Column(String(20))
    risk_level = Column(String(20))
    market_summary = Column(Text)
    news_headlines = Column(Text)
    embedding_vector = Column(ARRAY(Float))
    created_at = Column(DateTime, default=datetime.utcnow)


class MarketEmbDay(Base):
    __tablename__ = 'market_emb_day'
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(Integer, ForeignKey('tickers.id'))
    period_start = Column(Date, index=True)
    period_end = Column(Date, index=True)
    trend_strength = Column(String(50))
    volatility_regime = Column(String(20))
    momentum_phase = Column(String(20))
    technical_signals = Column(String(100))
    market_position = Column(String(20))
    risk_level = Column(String(20))
    market_summary = Column(Text)
    news_headlines = Column(Text)
    embedding_vector = Column(ARRAY(Float))
    news_flags = Column(Text)
    priority_score = Column(Integer, default=0)
    is_pinned = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class MarketEmbWeek(Base):
    __tablename__ = 'market_emb_week'
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(Integer, ForeignKey('tickers.id'))
    period_start = Column(Date, index=True)
    period_end = Column(Date, index=True)
    trend_strength = Column(String(50))
    volatility_regime = Column(String(20))
    momentum_phase = Column(String(20))
    technical_signals = Column(String(100))
    market_position = Column(String(20))
    risk_level = Column(String(20))
    market_summary = Column(Text)
    news_headlines = Column(Text)
    embedding_vector = Column(ARRAY(Float))
    created_at = Column(DateTime, default=datetime.utcnow)
    news_flags = Column(Text)
    priority_score = Column(Integer, default=0)
    is_pinned = Column(Boolean, default=False)


class ForecastMetrics(Base):
    __tablename__ = 'forecast_metrics'
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(Integer, ForeignKey('tickers.id'), nullable=False)
    forecast_date = Column(DateTime, nullable=False)  # When forecast was generated
    target_date = Column(DateTime, nullable=False)  # Date being forecasted
    forecast_horizon = Column(Integer, nullable=False)  # Days ahead (1, 7, 30, etc.)

    # Forecast values
    predicted_close = Column(Float)
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)

    # Model metadata
    model_type = Column(String(50))  # 'prophet', 'arima', 'lstm'
    model_version = Column(String(20))
    accuracy_score = Column(Float)  # MAPE, RMSE, etc.

    # Additional predictions
    predicted_volatility = Column(Float)
    trend_direction = Column(String(20))  # 'bullish', 'bearish', 'neutral'
    trend_strength = Column(Float)  # Prophet trend component
    seasonal_component = Column(Float)  # Prophet seasonal component

    created_at = Column(DateTime, default=datetime.utcnow)

    ticker = relationship("Ticker")


class ModelPerformance(Base):
    __tablename__ = 'model_performance'
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(Integer, ForeignKey('tickers.id'))
    model_type = Column(String(50))
    evaluation_date = Column(DateTime)

    # Performance metrics
    mape = Column(Float)  # Mean Absolute Percentage Error
    rmse = Column(Float)  # Root Mean Square Error
    mae = Column(Float)  # Mean Absolute Error
    directional_accuracy = Column(Float)  # % of correct up/down predictions

    # Evaluation period
    backtest_start = Column(DateTime)
    backtest_end = Column(DateTime)

    # Additional metrics
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)

    ticker = relationship("Ticker")


class ReferenceDocument(Base):
    __tablename__ = 'ref_documents'
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    published_at = Column(DateTime, default=datetime.utcnow, index=True)
    source = Column(String)
    url = Column(String)
    reasoning_analysis = Column(String)
    sentiment = Column(String)


class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    published_at = Column(DateTime, default=datetime.utcnow, index=True)
    source = Column(String)
    url = Column(String)
    reasoning_analysis = Column(String)
    sentiment = Column(String)
    vector_id = Column(String, unique=True, nullable=True)  # <-- Add this line!

    tickers = relationship("Ticker", secondary=ticker_document_association, back_populates="documents")


class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False)
    full_name = Column(String(200))
    email = Column(String(200), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    portfolios = relationship("Portfolio", back_populates="owner")


class Asset(Base):
    __tablename__ = 'assets'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)  # e.g. ticker, CUSIP, crypto symbol
    name = Column(String(200))
    type = Column(String(30))  # 'stock', 'bond', etc.
    currency = Column(String(10))
    region = Column(String(50))
    sector = Column(String(50))
    asset_metadata = Column(Text)  # Store as JSON string
    embedding = Column(ARRAY(Float))  # Semantic vector

    allocations = relationship("AssetAllocation", back_populates="asset")


class Portfolio(Base):
    __tablename__ = 'portfolios'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    owner_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'))
    total_value = Column(Float)
    currency = Column(String(10), default='USD')
    risk_profile = Column(String(30))  # 'conservative', 'balanced', etc.
    advisory_embedding = Column(ARRAY(Float))  # Semantic vector
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="portfolios")
    allocations = relationship("AssetAllocation", back_populates="portfolio", cascade="all, delete-orphan")
    recommendations = relationship("AdvisoryRecommendation", back_populates="portfolio", cascade="all, delete-orphan")


class AssetAllocation(Base):
    __tablename__ = 'asset_allocations'
    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id', ondelete='CASCADE'))
    asset_id = Column(Integer, ForeignKey('assets.id', ondelete='CASCADE'))
    weight = Column(Float)  # e.g. 0.18 for 18%
    value = Column(Float)   # absolute value in portfolio currency
    target_weight = Column(Float)  # desired allocation
    advisory_notes = Column(Text)
    embedding = Column(ARRAY(Float))  # Semantic vector for allocation rationale
    created_at = Column(DateTime, default=datetime.utcnow)

    portfolio = relationship("Portfolio", back_populates="allocations")
    asset = relationship("Asset", back_populates="allocations")


class AdvisoryRecommendation(Base):
    __tablename__ = 'advisory_recommendations'
    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id', ondelete='CASCADE'))
    created_at = Column(DateTime, default=datetime.utcnow)
    summary = Column(Text)
    details = Column(Text)
    embedding = Column(ARRAY(Float))  # Semantic vector
    rationale = Column(Text)
    action_items = Column(ARRAY(Text))  # Array of action items
    confidence_score = Column(Float)

    portfolio = relationship("Portfolio", back_populates="recommendations")




class LastProcessed(Base):
    __tablename__ = 'last_processed'

    ticker_symbol = Column(String, primary_key=True)
    last_timestamp = Column(Integer)


# Create session factory
SessionLocal = sessionmaker(bind=engine)

# REMOVED: Base.metadata.create_all(engine) from global scope
# ADDED: init_db function
def init_db():
    """Initialize database tables. Call this on app startup."""
    try:
        logger.info("Initializing database tables...")
        Base.metadata.create_all(engine)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # Don't raise here if you want the app to start even if DB is down temporarily,
        # but usually you want to know.

# Add to sql_service.py or create validation.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataValidator:
    @staticmethod
    def validate_historical_data(df: pd.DataFrame) -> tuple[bool, list]:
        """Validate historical data before insertion"""
        errors = []

        # Check required columns
        required_cols = ['timestamp', 'open', 'close', 'high', 'low', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # Check for null values in critical columns
        critical_nulls = df[['timestamp', 'close']].isnull().sum()
        if critical_nulls.any():
            errors.append(f"Null values in critical columns: {critical_nulls.to_dict()}")

        # Validate price relationships
        invalid_prices = df[
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
            ]
        if not invalid_prices.empty:
            errors.append(f"Invalid price relationships in {len(invalid_prices)} rows")

        # Check for negative values
        price_cols = ['open', 'close', 'high', 'low', 'volume']
        negative_values = (df[price_cols] < 0).sum()
        if negative_values.any():
            errors.append(f"Negative values found: {negative_values.to_dict()}")

        # Check timestamp ordering
        if not df['timestamp'].is_monotonic_increasing:
            errors.append("Timestamps are not in ascending order")

        return len(errors) == 0, errors

    @staticmethod
    def validate_metrics(metrics_dict: dict) -> tuple[bool, list]:
        """Validate computed metrics"""
        errors = []

        # Check for extreme values
        extreme_checks = {
            'rsi': (0, 100),
            'stoch': (0, 100),
            'willr': (-100, 0),
            'volatility_30': (0, 5),  # 500% max volatility
        }

        for metric, (min_val, max_val) in extreme_checks.items():
            if metric in metrics_dict:
                val = metrics_dict[metric]
                if val is not None and (val < min_val or val > max_val):
                    errors.append(f"{metric} value {val} outside expected range [{min_val}, {max_val}]")

        return len(errors) == 0, errors


def insert_historical_record_validated(ticker_id, timestamp, open_price, close_price,
                                       high_price, low_price, volume):
    """Enhanced version with validation"""
    # Create temporary dataframe for validation
    temp_df = pd.DataFrame([{
        'timestamp': timestamp,
        'open': open_price,
        'close': close_price,
        'high': high_price,
        'low': low_price,
        'volume': volume
    }])

    is_valid, errors = DataValidator.validate_historical_data(temp_df)
    if not is_valid:
        logger.error(f"Data validation failed: {errors}")
        return None

    # Proceed with the original insertion logic
    return insert_historical_record(ticker_id, timestamp, open_price, close_price,
                                    high_price, low_price, volume)


# --- Context Manager for Database Sessions ---

@contextmanager
def get_db_session():
    """Context manager for database sessions with proper cleanup"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


# --- Updated DB Utility Functions ---

def check_for_ticker(ticker):
    with get_db_session() as session:
        existing_ticker = session.query(Ticker).filter_by(symbol=ticker).first()
        if existing_ticker:
            # Detach from session so it can be used outside the context
            session.expunge(existing_ticker)
        return existing_ticker


def get_last_processed_timestamp(ticker_symbol):
    """
    Retrieve the most recent processed timestamp for a given ticker using a SQLAlchemy session.

    Args:
        ticker_symbol (str): The ticker symbol to query.

    Returns:
        datetime or None: The last processed timestamp for the ticker, or None if not found.
    """
    with get_db_session() as session:
        query = text("SELECT last_timestamp FROM last_processed WHERE ticker_symbol = :ticker_symbol")
        try:
            result = session.execute(query, {"ticker_symbol": ticker_symbol}).fetchone()
            return result[0] if result else None  # Returns the timestamp or None
        except Exception as e:
            logger.error(f"Error retrieving last processed timestamp: {e}")
            raise


def update_last_processed_timestamp(ticker_symbol, last_timestamp):
    """
    Update or insert the most recent timestamp for a given ticker using a SQLAlchemy session.

    Args:
        ticker_symbol (str): The ticker symbol to update.
        last_timestamp (datetime): The timestamp to update.
    """
    with get_db_session() as session:
        query = text("""
        INSERT INTO last_processed (ticker_symbol, last_timestamp)
        VALUES (:ticker_symbol, :last_timestamp)
        ON CONFLICT (ticker_symbol)
        DO UPDATE SET last_timestamp = EXCLUDED.last_timestamp;
        """)
        try:
            session.execute(query, {"ticker_symbol": ticker_symbol, "last_timestamp": last_timestamp})
            # Commit is handled by the context manager
        except Exception as e:
            logger.error(f"Error updating last processed timestamp: {e}")
            raise


def insert_new_ticker(symbol, company_name, general_info):
    """
    Insert a new ticker into the database.
    """
    with get_db_session() as session:
        try:
            # Serialize complex fields to JSON strings
            general_info_json = json.dumps(general_info)

            # Insert into the database
            new_ticker = Ticker(
                symbol=symbol,
                company_name=company_name,
                general_info=general_info_json,
            )
            session.add(new_ticker)
            session.flush()  # Get the ID without committing

            # Detach from session so it can be used outside the context
            session.expunge(new_ticker)
            return new_ticker
        except Exception as e:
            logger.error(f"Error inserting new ticker {symbol}: {e}")
            raise


def insert_new_ref_document(title, content, published_at, source, url, reasoning_analysis, sentiment):
    with get_db_session() as session:
        # Insert a document/news article
        new_ref_document = ReferenceDocument(
            title=title,
            content=content,
            published_at=published_at,
            source=source,
            url=url,
            reasoning_analysis=reasoning_analysis,
            sentiment=sentiment,
        )
        session.add(new_ref_document)


def insert_new_linked_document(ticker, title, content, published_at, source, url, reasoning_analysis, sentiment):
    with get_db_session() as session:
        # Insert a document/news article
        new_document = Document(
            title=title,
            content=content,
            published_at=published_at,
            source=source,
            url=url,
            reasoning_analysis=reasoning_analysis,
            sentiment=sentiment,
        )
        # Link the document with the ticker
        new_document.tickers.append(ticker)
        session.add(new_document)


def get_forecast_data(ticker_symbol: str, horizon_days: int = 30, model_type: str = 'prophet'):
    """Get recent forecast data for a ticker"""
    with get_db_session() as session:
        ticker = session.query(Ticker).filter_by(symbol=ticker_symbol).first()
        if not ticker:
            return None

        forecasts = session.query(ForecastMetrics).filter(
            ForecastMetrics.ticker_id == ticker.id,
            ForecastMetrics.forecast_horizon <= horizon_days,
            ForecastMetrics.model_type == model_type,
            ForecastMetrics.forecast_date >= datetime.utcnow() - timedelta(hours=6)
        ).order_by(ForecastMetrics.target_date).all()

        return forecasts


def store_forecast_metrics(ticker_id: int, forecasts: list, model_type: str = 'prophet'):
    """Store forecast results in database"""
    with get_db_session() as session:
        try:
            for forecast_data in forecasts:
                forecast_metric = ForecastMetrics(**forecast_data, ticker_id=ticker_id, model_type=model_type)
                session.add(forecast_metric)
            session.commit()
            logger.info(f"Stored {len(forecasts)} forecast metrics for ticker_id {ticker_id}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing forecast metrics: {e}")
            raise


def insert_historical_record(ticker_symbol, timestamp, open_, close_, high_, low_, volume_):
    with get_db_session() as session:
        # timestamp format; 1740700800
        # Check if the ticker already exists in the database
        existing_ticker = session.query(Ticker).filter_by(symbol=ticker_symbol).first()

        if existing_ticker:
            # Load the existing ticker ID
            ticker_id = existing_ticker.id
        else:
            # Create a new ticker if it doesn't exist
            logger.error(f"Ticker {ticker_symbol} not found when inserting historical data")
            return None

        try:
            # Insert a historical record for the ticker
            historical_record = HistoricalData(
                ticker_id=ticker_id,
                timestamp=int(timestamp),
                open=float(open_),
                close=float(close_),
                high=float(high_),
                low=float(low_),
                volume=float(volume_)
            )
            session.add(historical_record)
            session.flush()  # Get the ID without committing

            # Detach from session so it can be used outside the context
            session.expunge(historical_record)
            return historical_record
        except Exception as e:
            logger.error(f"Failed to insert HistoricalData for {ticker_symbol} at {timestamp}: {e}", exc_info=True)
            raise


def insert_computed_metrics(hd: HistoricalData, metrics: Union[dict, pd.Series]):
    """
    Insert or update ComputedMetrics for a given HistoricalData row,
    keyed solely by hd.id.

    Args:
        hd (HistoricalData): The HistoricalData row for which metrics are being calculated.
        metrics (Union[dict, pd.Series]): The computed metrics as a dictionary or pandas Series.

    Returns:
        tuple: (ComputedMetrics ORM object or None, metrics dict or None)
    """
    with get_db_session() as session:
        try:
            # 1) Normalize metrics → plain dict
            if hasattr(metrics, "to_dict"):
                data = metrics.to_dict()
            elif isinstance(metrics, dict):
                data = metrics
            else:
                data = dict(metrics)

            # 2) List of all ComputedMetrics columns except id, historical_data_id, and relationships
            metric_fields = [
                c.name for c in ComputedMetrics.__table__.columns
                if c.name not in ("id", "historical_data_id")
            ]

            # 3) Prepare ComputedMetrics data for insertion
            def to_float_safe(val):
                try:
                    return float(val) if val is not None and pd.notna(val) else None
                except Exception:
                    return None

            kwargs = {'historical_data_id': hd.id}
            for key in metric_fields:
                # Map DataFrame key to ORM column name (if they match)
                kwargs[key] = to_float_safe(data.get(key))

            # 4) Insert or update
            existing_cm = session.query(ComputedMetrics).filter_by(historical_data_id=hd.id).one_or_none()
            if existing_cm:
                logger.debug("Updating existing ComputedMetrics for hd.id=%s", hd.id)
                for key, value in kwargs.items():
                    setattr(existing_cm, key, value)
                cm = existing_cm
            else:
                cm = ComputedMetrics(**kwargs)
                session.add(cm)
                logger.debug("Inserted ComputedMetrics for hd.id=%s", hd.id)

            session.flush()  # Get the ID without committing

            # Detach from session so it can be used outside the context
            session.expunge(cm)

            # Return both the ORM object and the normalized metrics dict
            return cm, kwargs

        except Exception as e:
            logger.exception("Error inserting/updating computed metrics for HistoricalData.id=%s: %s", hd.id, e)
            raise


if __name__ == "__main__":
    x = 1