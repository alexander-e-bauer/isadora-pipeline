import logging
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

import time
from config import PINECONE_API_KEY, PINECONE_HOST, logger, key
from xyz.finazon_service.retrive_data import FinazonService
from xyz.finazon_service.sql_service import (
    get_last_processed_timestamp,
    update_last_processed_timestamp,
    insert_historical_record,
    insert_computed_metrics,
    insert_new_ticker,
    check_for_ticker,
)
from xyz.finazon_service.api_service import get_new_ticker_gen_data
from datetime import datetime
from pinecone import Pinecone
import pandas as pd

# Initialize Pinecone and Flask
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_HOST)
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[]
)

# Utility to fetch and store ticker data
def fetch_and_store_ticker_data(ticker_symbol):
    """
    Fetch general and financial data for a ticker and store it in the database.
    """
    logger.info(f"Fetching data for ticker: {ticker_symbol}")
    try:

        # Check if ticker already exists
        existing_ticker = check_for_ticker(ticker_symbol)
        if existing_ticker:
            logger.info(f"Ticker {ticker_symbol} already exists in DB.")
            return existing_ticker  # Return the ORM object!

        # Fetch general and financial data
        general_data, ticker_name = get_new_ticker_gen_data(ticker_symbol)

        # Insert the new ticker into the database
        new_ticker = insert_new_ticker(ticker_symbol, ticker_name, general_data)
        logger.info(f"Ticker {ticker_symbol} data stored successfully.")
        return new_ticker
    except Exception as e:
        logger.error(f"Error fetching or storing ticker data for {ticker_symbol}: {e}")
        return None


import numpy as np

def update_time_series_data(ticker, interval='30m', start_year=2025, batch_size=100):
    logger.info(f"Starting time-series data update for ticker: {ticker}")
    retriever = FinazonService(rate_limit_per_minute=5)
    from xyz.finazon_service.metrics import compute_batch_metrics

    # Check/create ticker
    existing_ticker = check_for_ticker(ticker)
    company_name = existing_ticker.company_name
    if not existing_ticker:
        try:
            general_data, ticker_name = get_new_ticker_gen_data(ticker)
            insert_new_ticker(
                symbol=ticker,
                company_name=ticker_name if ticker_name else f"Company {ticker}",
                general_info=general_data if general_data else {"info": "Placeholder general info"},
            )
            logger.info(f"Ticker {ticker} created successfully.")
        except Exception as e:
            logger.error(f"Error creating new ticker {ticker}: {e}")
            return

    last_timestamp = get_last_processed_timestamp(ticker)
    if last_timestamp:
        try:
            last_timestamp = int(last_timestamp)
            start_time = datetime.utcfromtimestamp(last_timestamp)
        except ValueError as e:
            logger.error(f"Invalid last timestamp for {ticker}: {last_timestamp}. Error: {e}")
            return
    else:
        start_time = datetime(start_year, 1, 1)
        logger.info(f"No last timestamp found. Fetching data starting from {start_time}.")

    start_time = FinazonService.format_start_time(start_time)

    try:
        new_data = retriever.fetch_time_series(ticker=ticker, interval=interval, existing_df=None, start_time=start_time)
        if new_data is None or new_data.empty:
            logger.warning(f"No new data found for {ticker}.")
            return
    except Exception as e:
        logger.error(f"Error fetching time-series data for {ticker}: {e}")
        return

    # --- Batching with Overlap ---
    # Find the largest window in your metrics (adjust as needed!)
    largest_window = 48  # e.g., for volatility_30, adjust if you have a larger window
    n = len(new_data)
    for i in range(0, n, batch_size):
        # Always include previous largest_window-1 rows for context
        start_idx = max(0, i - largest_window + 1)
        end_idx = min(i + batch_size, n)
        batch = new_data.iloc[start_idx:end_idx].copy()

        # Compute metrics
        try:
            metrics_df = compute_batch_metrics(batch)
        except Exception as e:
            logger.error(f"Error computing metrics for {ticker} batch {i}-{end_idx}: {e}")
            continue

        # Only insert the "new" rows for this batch (not the overlap)
        insert_start = i if i > 0 else 0
        insert_rows = metrics_df.iloc[insert_start - start_idx:end_idx - start_idx]
        REQUIRED_FIELDS = ['timestamp', 'open', 'close', 'high', 'low', 'volume']

        for _, row in insert_rows.iterrows():

            row['timestamp'] = int(row['timestamp'])
            if any(pd.isnull(row[field]) for field in REQUIRED_FIELDS):
                logger.error(f"Skipping row with missing REQUIRED values: {row}")
                continue
            # Insert historical record
            historical_record = insert_historical_record(
                ticker_symbol=ticker,
                timestamp=int(row['timestamp']),
                open_=float(row['open']),
                close_=float(row['close']),
                high_=float(row['high']),
                low_=float(row['low']),
                volume_=int(row['volume'])
            )

            if not historical_record:
                logger.error(
                    f"Skipping computed metrics and market state for {ticker} at {row['timestamp']} due to failed historical insert.")
                continue

            try:
                # Build computed_metrics dict using your schema

                metric_cols = [
                    'sma', 'ema', 'dema', 'tema', 'wma', 'trima',
                    'kama', 'mama', 't3', 'log_return', 'volatility_30', 'vwap',
                    'macd', 'macdext', 'signal_line', 'rsi', 'stoch', 'stochrsi',
                    'willr', 'adx', 'adxr', 'apo', 'ppo', 'mom', 'bop', 'cci', 'cmo',
                    'roc', 'rocr', 'aroon', 'aroonosc', 'mfi', 'trix', 'ultosc',
                    'price_change', 'price_change_pct', 'hourly_return', 'volatility',
                    'historical_volatility', 'realized_volatility', 'typical_price',
                    'sma_20', 'ema_20', 'dema_20', 'tema_20', 'wma_20', 'trima_20',
                    'macd_signal', 'macd_hist', 'bollinger_upper', 'bollinger_lower',
                    'bollinger_width', 'obv', 'cmf', 'z_score', 'ewma_score',
                    'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'var', 'cvar',
                    'stoch_k', 'stoch_d', 'fama'
                ]
                computed_metrics_json = {}
                for col in metric_cols:
                    val = row.get(col, None)
                    if isinstance(val, (np.floating, float)) and (np.isnan(val) or val is None):
                        computed_metrics_json[col] = None
                    else:
                        computed_metrics_json[col] = float(val) if val is not None else None

                cm_obj, cm_data = insert_computed_metrics(historical_record, computed_metrics_json)

                #if cm_obj and cm_data:
                #    robust_process_and_store_market_states(ticker_orm, historical_record, cm_data)

            except Exception as e:
                logger.error(f"Error inserting record for {ticker} at timestamp {row['timestamp']}: {e}")
                logger.error(f"Row data: {row}")


    # Update last processed timestamp
    try:
        latest_timestamp = new_data['timestamp'].max()
        update_last_processed_timestamp(ticker, latest_timestamp)
        logger.info(f"Updated last processed timestamp for {ticker} to {latest_timestamp}.")
    except Exception as e:
        logger.error(f"Error updating last processed timestamp for {ticker}: {e}")

    # ---- AGGREGATE AND EMBED ----
    try:
        from xyz.finazon_service.aggregate_embeddings import process_aggregated_embeddings  # Adjust the import as needed
        logger.info(f"Starting multi-scale aggregation and embedding for {ticker}...")
        process_aggregated_embeddings(ticker, company_name)
        logger.info(f"Aggregation and embedding complete for {ticker}.")
    except Exception as e:
        logger.error(f"Error during aggregation and embedding for {ticker}: {e}")


def initialize_database():
    tickers = ['AAPL', 'GOOG', 'TSLA']

    for ticker in tickers:
        try:
            # Step 1: Fetch and store general ticker information
            logger.info(f"Fetching general data for {ticker}")
            ticker_obj = fetch_and_store_ticker_data(ticker)
            time.sleep(2)

            # Step 2: Fetch and store time series data
            logger.info(f"Fetching time series data for {ticker}")
            update_time_series_data(ticker, interval='30m')


            logger.info(f"Successfully processed {ticker}")

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            continue


def check_api_key():
    auth = request.headers.get('Authorization')
    if not auth or auth != f"Bearer {key}":
        return False
    return True

@app.route('/run-pipeline', methods=['POST'])
@limiter.limit("1 per 5 minutes")
def run_pipeline():
    if not check_api_key():
        return jsonify({'error': 'Unauthorized'}), 401
    initialize_database()
    return "Pipeline complete", 200


@app.route('/')
def hello_world():
    logger.info("Received request to root endpoint.")
    return 'Hello World!'


if __name__ == '__main__':
    logger.info("Starting Flask application.")
    app.run()
