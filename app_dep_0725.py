import logging
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
import threading
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
    get_db_session, insert_historical_record_validated
)
from xyz.finazon_service.api_service import get_new_ticker_gen_data
from xyz.finazon_service.forecast_engine import ForecastEngine
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


def update_time_series_data(ticker, interval='30m', start_year=2025, start_month=4, start_day=1, batch_size=100):
    logger.info(f"Starting time-series data update for ticker: {ticker}")
    retriever = FinazonService(rate_limit_per_minute=5)
    from xyz.finazon_service.metrics import compute_batch_metrics
    from xyz.finazon_service.backfill import backfill_recent_metrics

    # Check/create ticker
    existing_ticker = check_for_ticker(ticker)

    if not existing_ticker:
        try:
            general_data, ticker_name = get_new_ticker_gen_data(ticker)
            insert_new_ticker(
                symbol=ticker,
                company_name=ticker_name if ticker_name else f"Company {ticker}",
                general_info=general_data if general_data else {"info": "Placeholder general info"},
            )
            logger.info(f"Ticker {ticker} created successfully.")
            # Re-fetch the ticker after creation
            existing_ticker = check_for_ticker(ticker)
        except Exception as e:
            logger.error(f"Error creating new ticker {ticker}: {e}")
            return False

    company_name = existing_ticker.company_name if existing_ticker else f"Company {ticker}"

    # Get last processed timestamp
    last_timestamp = get_last_processed_timestamp(ticker)
    if last_timestamp:
        try:
            last_timestamp = int(last_timestamp)
            start_time = datetime.utcfromtimestamp(last_timestamp)
            logger.info(f"Resuming from {start_time} for {ticker}")
        except ValueError as e:
            logger.error(f"Invalid last timestamp for {ticker}: {last_timestamp}. Error: {e}")
            return False
    else:
        start_time = datetime(start_year, start_month, start_day)
        logger.info(f"No last timestamp found. Fetching data starting from {start_time}.")

    start_time = FinazonService.format_start_time(start_time)

    # Fetch new data
    try:
        new_data = retriever.fetch_time_series(
            ticker=ticker,
            interval=interval,
            existing_df=None,
            start_time=start_time
        )
        if new_data is None or new_data.empty:
            logger.info(f"No new data found for {ticker}.")
            # Even if no new data, run backfill to ensure consistency
            logger.info(f"Running backfill for consistency check on {ticker}")
            backfill_success = backfill_recent_metrics(
                ticker,
                lookback_periods=200,
                update_periods=50
            )
            if backfill_success:
                logger.info(f"Backfill completed for {ticker}")

            # Still run aggregation and forecasting on existing data
            success = run_post_processing(ticker, company_name)
            return success

    except Exception as e:
        logger.error(f"Error fetching time-series data for {ticker}: {e}")
        return False

    logger.info(f"Processing {len(new_data)} new records for {ticker}")

    # --- Enhanced Batching with Overlap for Better Metrics ---
    largest_window = 200  # Increased window for better metric calculation
    n = len(new_data)
    total_inserted = 0

    for i in range(0, n, batch_size):
        try:
            # Always include previous largest_window-1 rows for context
            start_idx = max(0, i - largest_window + 1)
            end_idx = min(i + batch_size, n)
            batch = new_data.iloc[start_idx:end_idx].copy()

            logger.debug(f"Processing batch {i // batch_size + 1}: rows {start_idx} to {end_idx}")

            # Compute metrics with full context
            try:
                metrics_df = compute_batch_metrics(batch)
            except Exception as e:
                logger.error(f"Error computing metrics for {ticker} batch {i}-{end_idx}: {e}")
                continue

            # Only insert the "new" rows for this batch (not the overlap)
            actual_new_start = max(0, i)
            actual_new_end = min(i + batch_size, n)

            # Calculate the slice indices for the metrics_df
            slice_start = actual_new_start - start_idx
            slice_end = actual_new_end - start_idx
            insert_rows = metrics_df.iloc[slice_start:slice_end]

            REQUIRED_FIELDS = ['timestamp', 'open', 'close', 'high', 'low', 'volume']

            batch_inserted = 0
            for _, row in insert_rows.iterrows():
                try:
                    row['timestamp'] = int(row['timestamp'])

                    # Validate required fields
                    if any(pd.isnull(row[field]) for field in REQUIRED_FIELDS):
                        logger.warning(f"Skipping row with missing required values for {ticker} at {row['timestamp']}")
                        continue

                    # Insert historical record
                    historical_record = insert_historical_record_validated(
                        ticker_symbol=ticker,
                        timestamp=int(row['timestamp']),
                        open_=float(row['open']),
                        close_=float(row['close']),
                        high_=float(row['high']),
                        low_=float(row['low']),
                        volume_=int(row['volume'])
                    )

                    if not historical_record:
                        logger.warning(f"Failed to insert historical record for {ticker} at {row['timestamp']}")
                        continue

                    # Build computed metrics dictionary
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
                        'stoch_k', 'stoch_d', 'fama', 'volatility_30',
                    ]

                    computed_metrics_json = {}
                    for col in metric_cols:
                        val = row.get(col, None)
                        if isinstance(val, (np.floating, float)) and (np.isnan(val) or val is None):
                            computed_metrics_json[col] = None
                        else:
                            computed_metrics_json[col] = float(val) if val is not None else None

                    # Insert computed metrics
                    cm_obj, cm_data = insert_computed_metrics(historical_record, computed_metrics_json)

                    if cm_obj and cm_data:
                        batch_inserted += 1
                        total_inserted += 1

                except Exception as e:
                    logger.error(
                        f"Error inserting record for {ticker} at timestamp {row.get('timestamp', 'unknown')}: {e}")
                    continue

            logger.info(f"Batch {i // batch_size + 1} completed: {batch_inserted} records inserted for {ticker}")

        except Exception as e:
            logger.error(f"Error processing batch {i // batch_size + 1} for {ticker}: {e}")
            continue

    logger.info(f"Completed batch processing for {ticker}: {total_inserted} total records inserted")

    # Update last processed timestamp
    try:
        latest_timestamp = new_data['timestamp'].max()
        update_last_processed_timestamp(ticker, latest_timestamp)
        logger.info(f"Updated last processed timestamp for {ticker} to {latest_timestamp}")
    except Exception as e:
        logger.error(f"Error updating last processed timestamp for {ticker}: {e}")

    # --- BACKFILL FOR CONSISTENCY ---
    logger.info(f"Starting backfill for {ticker} to ensure metric consistency")
    try:
        # Backfill more records after bulk insert to ensure consistency
        backfill_periods = min(100, total_inserted + 50)  # Backfill recent records plus some buffer
        backfill_success = backfill_recent_metrics(
            ticker,
            lookback_periods=200,  # Use full context
            update_periods=backfill_periods  # Update recent records
        )

        if backfill_success:
            logger.info(f"Backfill completed successfully for {ticker}")
        else:
            logger.warning(f"Backfill had issues for {ticker}, but continuing with pipeline")

    except Exception as e:
        logger.error(f"Backfill failed for {ticker}: {e}")
        # Don't fail the entire pipeline if backfill fails
        logger.warning(f"Continuing pipeline for {ticker} despite backfill failure")

    # --- POST-PROCESSING: AGGREGATION, EMBEDDING, AND FORECASTING ---
    success = run_post_processing(ticker, company_name)

    if success:
        logger.info(f"✅ Complete pipeline finished successfully for {ticker}")
        return True
    else:
        logger.warning(f"⚠️ Pipeline completed with some issues for {ticker}")
        return False


def run_post_processing(ticker, company_name):
    """
    Run the post-processing steps: aggregation, embedding, and forecasting.

    Args:
        ticker: Ticker symbol
        company_name: Company name for the ticker

    Returns:
        bool: True if all steps completed successfully
    """
    success_flags = []

    # ---- AGGREGATE AND EMBED ----
    try:
        from xyz.finazon_service.aggregate_embeddings import process_aggregated_embeddings
        logger.info(f"🔄 Starting multi-scale aggregation and embedding for {ticker}...")
        process_aggregated_embeddings(ticker, company_name)
        logger.info(f"✅ Aggregation and embedding complete for {ticker}")
        success_flags.append(True)
    except Exception as e:
        logger.error(f"❌ Error during aggregation and embedding for {ticker}: {e}")
        success_flags.append(False)

    # ---- FORECASTING ----
    try:
        logger.info(f"🔮 Starting forecast generation for {ticker}...")
        forecast_engine = ForecastEngine()
        forecast_success = forecast_engine.generate_forecasts_for_ticker(
            ticker_symbol=ticker,
            horizons=[1, 7, 30]  # 1-day, 7-day, and 30-day forecasts
        )
        if forecast_success:
            logger.info(f"✅ Forecasts generated successfully for {ticker}")
            success_flags.append(True)
        else:
            logger.warning(f"⚠️ Forecast generation failed for {ticker}")
            success_flags.append(False)
    except Exception as e:
        logger.error(f"❌ Error generating forecasts for {ticker}: {e}")
        success_flags.append(False)

    # Return True if at least one post-processing step succeeded
    return any(success_flags)


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


def run_pipeline_async(request_data):
    """Run the pipeline in a separate thread"""
    try:
        logger.info(f"🚀 Starting async pipeline execution with data: {request_data}")
        start_time = time.time()

        # Run your existing pipeline logic
        initialize_database()

        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"✅ Pipeline completed successfully in {duration:.2f} seconds")

    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        logger.exception("Full pipeline error traceback:")


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

    try:
        # Get request data
        request_data = request.get_json() or {}

        # Add metadata
        request_data.update({
            'timestamp': datetime.utcnow().isoformat(),
            'trigger_source': request_data.get('trigger_source', 'api_call'),
            'message': request_data.get('message', 'Pipeline execution triggered')
        })

        # Start pipeline in background thread
        thread = threading.Thread(target=run_pipeline_async, args=(request_data,))
        thread.daemon = True
        thread.start()

        logger.info(f"🎯 Pipeline triggered successfully - running in background")

        # Return immediately with 202 Accepted
        return jsonify({
            "status": "accepted",
            "message": "Pipeline started successfully and running in background",
            "timestamp": request_data['timestamp'],
            "trigger_source": request_data['trigger_source'],
            "request_data": request_data
        }), 202

    except Exception as e:
        logger.error(f"Error triggering pipeline: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to trigger pipeline: {str(e)}"
        }), 500


# Optional: Keep a synchronous version for testing/debugging
@app.route('/run-pipeline-sync', methods=['POST'])
#@limiter.limit("1 per 10 minutes")
def run_pipeline_sync():
    """Synchronous version of the pipeline (for testing)"""
    if not check_api_key():
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        logger.info("🔄 Running pipeline synchronously...")
        start_time = time.time()

        initialize_database()

        end_time = time.time()
        duration = end_time - start_time

        return jsonify({
            "status": "completed",
            "message": "Pipeline completed successfully",
            "duration_seconds": duration,
            "timestamp": datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Synchronous pipeline failed: {e}")
        return jsonify({
            "status": "error",
            "message": f"Pipeline failed: {str(e)}"
        }), 500


@app.route('/')
def hello_world():
    logger.info("Received request to root endpoint.")
    return 'Hello World!'


if __name__ == '__main__':
    logger.info("Starting Flask application.")
    app.run()