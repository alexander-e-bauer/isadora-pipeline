import psycopg2
from datetime import datetime
from typing import List, Dict, Any, Union
import traceback
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey, BigInteger, Date, DateTime, Text, Table, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship, sessionmaker, declarative_base, Session
from config import DATABASE, logger
import pandas as pd
import numpy as np
from xyz.llm.embedding_generator import get_embedding

import logging
import json
from xyz.finazon_service.api_service import get_new_ticker_gen_data


Base = declarative_base()

# Create SQLAlchemy engine
DATABASE_URI = f"postgresql+psycopg2://{DATABASE.DB_USER}:{DATABASE.DB_PASSWORD}@{DATABASE.DB_HOST}/{DATABASE.DB_NAME}"
engine: Engine = create_engine(DATABASE_URI)

# Association table for many-to-many relationship between Ticker and Document
ticker_document_association = Table(
    'ticker_document_association',
    Base.metadata,
    Column('ticker_id', Integer, ForeignKey('tickers.id'), primary_key=True),
    Column('document_id', Integer, ForeignKey('documents.id'), primary_key=True)
)

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

    historical_data = relationship("HistoricalData", back_populates="metrics")


# --- Aggregated Market Embedding Tables ---

class MarketEmbHour(Base):
    __tablename__ = 'market_emb_hour'
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(Integer, ForeignKey('tickers.id'))
    period_start = Column(Date, index=True)   # e.g. 2025-05-30 13:00:00
    period_end = Column(Date, index=True)
    trend_strength = Column(String(50))
    volatility_regime = Column(String(20))
    momentum_phase = Column(String(20))
    technical_signals = Column(String(100))
    market_position = Column(String(20))
    risk_level = Column(String(20))
    market_summary = Column(Text)
    news_headlines = Column(Text)   # JSON string of news (or just text blob)
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


class LastProcessed(Base):
    __tablename__ = 'last_processed'

    ticker_symbol = Column(String, primary_key=True)
    last_timestamp = Column(Integer)


Session = sessionmaker(bind=engine)
session = Session()

# Create all tables in the database
Base.metadata.create_all(engine)

# --- DB Utility Functions ---


def check_for_ticker(ticker):
    existing_ticker = session.query(Ticker).filter_by(symbol=ticker).first()
    return existing_ticker


def get_last_processed_timestamp(ticker_symbol, session=session):
    """
    Retrieve the most recent processed timestamp for a given ticker using a SQLAlchemy session.

    Args:
        ticker_symbol (str): The ticker symbol to query.
        session (sqlalchemy.orm.Session): Active SQLAlchemy session.

    Returns:
        datetime or None: The last processed timestamp for the ticker, or None if not found.
    """
    query = text("SELECT last_timestamp FROM last_processed WHERE ticker_symbol = :ticker_symbol")
    try:
        result = session.execute(query, {"ticker_symbol": ticker_symbol}).fetchone()
        return result[0] if result else None  # Returns the timestamp or None
    except Exception as e:
        print(f"Error retrieving last processed timestamp: {e}")
        raise





def update_last_processed_timestamp(ticker_symbol, last_timestamp, session=session):
    """
    Update or insert the most recent timestamp for a given ticker using a SQLAlchemy session.

    Args:
        ticker_symbol (str): The ticker symbol to update.
        last_timestamp (datetime): The timestamp to update.
        session (sqlalchemy.orm.Session): Active SQLAlchemy session.
    """
    query = text("""
    INSERT INTO last_processed (ticker_symbol, last_timestamp)
    VALUES (:ticker_symbol, :last_timestamp)
    ON CONFLICT (ticker_symbol)
    DO UPDATE SET last_timestamp = EXCLUDED.last_timestamp;
    """)
    try:
        session.execute(query, {"ticker_symbol": ticker_symbol, "last_timestamp": last_timestamp})
        session.commit()  # Commit the transaction
    except Exception as e:
        print(f"Error updating last processed timestamp: {e}")
        session.rollback()  # Roll back if there's an error
        raise



def insert_new_ticker(symbol, company_name, general_info):
    """
    Insert a new ticker into the database.
    """
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
        session.commit()
        return new_ticker
    except Exception as e:
        session.rollback()
        raise e


def insert_new_ref_document(title, content, published_at, source, url, reasoning_analysis, sentiment):
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
    session.commit()


def insert_new_linked_document(ticker, title, content, published_at, source, url, reasoning_analysis, sentiment):
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
    session.commit()



def insert_historical_record(ticker_symbol, timestamp, open_, close_, high_, low_, volume_, session = session):
    # timestamp format; 1740700800
    # Check if the ticker already exists in the database
    existing_ticker = session.query(Ticker).filter_by(symbol=ticker_symbol).first()

    if existing_ticker:
        # Load the existing ticker ID
        ticker_id = existing_ticker.id
    else:
        # Create a new ticker if it doesn't exist
        new_ticker = insert_new_ticker(ticker_symbol)
        session.add(new_ticker)
        session.commit()
        ticker_id = new_ticker.id
        print(f"Ticker {ticker_symbol} created with ID {ticker_id}.")

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
        session.commit()
        return historical_record
    except Exception as e:
        logger.error(f"Failed to insert HistoricalData for {ticker_symbol} at {timestamp}: {e}", exc_info=True)
        session.rollback()
        return None

def insert_computed_metrics(
        hd: HistoricalData,
        metrics: Union[dict, pd.Series],
        session = session
):
    """
    Insert or update ComputedMetrics for a given HistoricalData row,
    keyed solely by hd.id.

    Args:
        hd (HistoricalData): The HistoricalData row for which metrics are being calculated.
        metrics (Union[dict, pd.Series]): The computed metrics as a dictionary or pandas Series.

    Returns:
        tuple: (ComputedMetrics ORM object or None, metrics dict or None)
    """
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

        # Commit the transaction
        session.commit()
        # Return both the ORM object and the normalized metrics dict
        return cm, kwargs

    except Exception as e:
        logger.exception("Error inserting/updating computed metrics for HistoricalData.id=%s: %s", hd.id, e)
        session.rollback()
        return None, None



if __name__ == "__main__":
    x = 1