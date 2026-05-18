import json
import logging
import asyncio
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import sys
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from xyz.finazon_service.sql_service import init_db

from config import PINECONE_API_KEY, PINECONE_HOST, logger, key, ANTHROPIC_API_KEY
from xyz.observability import (
    RequestIdMiddleware,
    configure_logging,
)
from xyz.finazon_service.retrive_data import FinazonService
from xyz.finazon_service.sql_service import (
    get_last_processed_timestamp,
    update_last_processed_timestamp,
    insert_historical_record,
    insert_computed_metrics,
    insert_new_ticker,
    check_for_ticker,
    get_db_session,
    insert_historical_record_validated
)
from xyz.finazon_service.api_service import get_new_ticker_gen_data
from xyz.finazon_service.forecast_engine import ForecastEngine
from pinecone import Pinecone

# Import monitoring
from xyz.finazon_service.monitoring import monitor, MonitoredOperation

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_HOST)

# Install structured JSON logging before FastAPI is built so the
# "Application startup complete" line carries the documented schema.
# Idempotent — safe under reloaders / repeated test imports.
configure_logging()

# Initialize FastAPI
app = FastAPI(
    title="Financial Data Pipeline API",
    description="Advanced financial data processing pipeline with comprehensive monitoring",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# RequestIdMiddleware must be outermost so the correlation id is set
# before auth + handler code runs.  add_middleware prepends to the
# stack, so installing this first puts it on the outside.
app.add_middleware(RequestIdMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()


# Pydantic models
class PipelineRequest(BaseModel):
    tickers: Optional[List[str]] = Field(default=["AAPL", "GOOG", "TSLA"],
                                         description="List of ticker symbols to process")
    interval: Optional[str] = Field(default="30m", description="Data interval (30m, 1h, 1d)")
    start_year: Optional[int] = Field(default=2025, description="Start year for data fetching")
    start_month: Optional[int] = Field(default=4, description="Start month for data fetching")
    start_day: Optional[int] = Field(default=1, description="Start day for data fetching")
    batch_size: Optional[int] = Field(default=100, description="Batch size for processing")
    force_refresh: Optional[bool] = Field(default=False, description="Force refresh of existing data")
    trigger_source: Optional[str] = Field(default="api_call", description="Source of the trigger")
    message: Optional[str] = Field(default="Pipeline execution triggered", description="Custom message")


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    summary: Dict[str, Any]
    operation_breakdown: Dict[str, Any]
    top_errors: Dict[str, int]
    active_operations: int
    system_health: Optional[Dict[str, Any]]


class TickerPerformanceResponse(BaseModel):
    ticker: str
    operations_count: int
    success_rate: float
    total_records_processed: int
    total_processing_time: float
    avg_records_per_second: float
    last_operation: Optional[str]


# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != key:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials


# Rate limiting (simple in-memory implementation)
class RateLimiter:
    def __init__(self):
        self.requests = {}
        self.max_requests = 5
        self.window_seconds = 300  # 5 minutes

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        if client_ip not in self.requests:
            self.requests[client_ip] = []

        # Clean old requests
        self.requests[client_ip] = [req_time for req_time in self.requests[client_ip]
                                    if now - req_time < self.window_seconds]

        # Check if under limit
        if len(self.requests[client_ip]) >= self.max_requests:
            return False

        # Add current request
        self.requests[client_ip].append(now)
        return True


rate_limiter = RateLimiter()


# Rate limiting dependency
async def check_rate_limit(request: Request):
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
    return True

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application...")
    init_db()


# Business logic functions with monitoring
async def fetch_and_store_ticker_data(ticker_symbol: str) -> Optional[Any]:
    """Fetch general and financial data for a ticker and store it in the database."""
    with MonitoredOperation(ticker_symbol, 'fetch_ticker_data') as op:
        logger.info(f"Fetching data for ticker: {ticker_symbol}")
        try:
            # Check if ticker already exists
            existing_ticker = check_for_ticker(ticker_symbol)
            if existing_ticker:
                logger.info(f"Ticker {ticker_symbol} already exists in DB.")
                op.add_data('status', 'existing_ticker')
                return existing_ticker

            # Fetch general and financial data
            general_data, ticker_name = get_new_ticker_gen_data(ticker_symbol)

            # Insert the new ticker into the database
            new_ticker = insert_new_ticker(ticker_symbol, ticker_name, general_data)
            logger.info(f"Ticker {ticker_symbol} data stored successfully.")

            op.add_records(1)  # One ticker processed
            op.add_data('status', 'new_ticker_created')
            op.add_data('ticker_name', ticker_name)

            return new_ticker

        except Exception as e:
            logger.error(f"Error fetching or storing ticker data for {ticker_symbol}: {e}")
            op.add_error(f"Ticker data fetch failed: {str(e)}")
            return None


async def update_time_series_data(ticker: str, interval: str = '30m', start_year: int = 2025,
                                  start_month: int = 4, start_day: int = 1, batch_size: int = 100) -> bool:
    """Enhanced time-series data update with comprehensive monitoring"""
    with MonitoredOperation(ticker, 'time_series_update', batch_size) as main_op:
        logger.info(f"Starting time-series data update for ticker: {ticker}")
        retriever = FinazonService(rate_limit_per_minute=5)
        from xyz.finazon_service.metrics import compute_batch_metrics
        from xyz.finazon_service.backfill import backfill_recent_metrics

        # Check/create ticker
        existing_ticker = check_for_ticker(ticker)

        if not existing_ticker:
            try:
                with MonitoredOperation(ticker, 'create_ticker') as create_op:
                    general_data, ticker_name = get_new_ticker_gen_data(ticker)
                    insert_new_ticker(
                        symbol=ticker,
                        company_name=ticker_name if ticker_name else f"Company {ticker}",
                        general_info=general_data if general_data else {"info": "Placeholder general info"},
                    )
                    logger.info(f"Ticker {ticker} created successfully.")
                    create_op.add_records(1)
                    create_op.add_data('ticker_name', ticker_name)

                    # Re-fetch the ticker after creation
                    existing_ticker = check_for_ticker(ticker)
            except Exception as e:
                logger.error(f"Error creating new ticker {ticker}: {e}")
                main_op.add_error(f"Ticker creation failed: {str(e)}")
                return False

        company_name = existing_ticker.company_name if existing_ticker else f"Company {ticker}"

        # Get last processed timestamp
        last_timestamp = get_last_processed_timestamp(ticker)
        if last_timestamp:
            try:
                last_timestamp = int(last_timestamp)
                start_time = datetime.utcfromtimestamp(last_timestamp)
                logger.info(f"Resuming from {start_time} for {ticker}")
                main_op.add_data('resume_from', start_time.isoformat())
            except ValueError as e:
                logger.error(f"Invalid last timestamp for {ticker}: {last_timestamp}. Error: {e}")
                main_op.add_error(f"Invalid timestamp: {str(e)}")
                return False
        else:
            start_time = datetime(start_year, start_month, start_day)
            logger.info(f"No last timestamp found. Fetching data starting from {start_time}.")
            main_op.add_data('start_from', start_time.isoformat())

        start_time = FinazonService.format_start_time(start_time)

        # Fetch new data with monitoring
        try:
            with MonitoredOperation(ticker, 'fetch_time_series') as fetch_op:
                new_data = retriever.fetch_time_series(
                    ticker=ticker,
                    interval=interval,
                    existing_df=None,
                    start_time=start_time
                )

                if new_data is None or new_data.empty:
                    logger.info(f"No new data found for {ticker}.")
                    fetch_op.add_data('status', 'no_new_data')

                    # Run backfill for consistency
                    with MonitoredOperation(ticker, 'backfill_consistency') as backfill_op:
                        logger.info(f"Running backfill for consistency check on {ticker}")
                        backfill_success = backfill_recent_metrics(
                            ticker,
                            lookback_periods=200,
                            update_periods=50
                        )
                        backfill_op.add_data('success', backfill_success)
                        if backfill_success:
                            logger.info(f"Backfill completed for {ticker}")

                    # Still run post-processing
                    success = await run_post_processing(ticker, company_name)
                    main_op.add_data('post_processing_success', success)
                    return success

                fetch_op.add_records(len(new_data))
                fetch_op.add_data('data_points', len(new_data))

        except Exception as e:
            logger.error(f"Error fetching time-series data for {ticker}: {e}")
            main_op.add_error(f"Time series fetch failed: {str(e)}")
            return False

        logger.info(f"Processing {len(new_data)} new records for {ticker}")
        main_op.add_data('total_new_records', len(new_data))

        # Enhanced Batching with Monitoring
        largest_window = 200
        n = len(new_data)
        total_inserted = 0

        for i in range(0, n, batch_size):
            batch_num = i // batch_size + 1

            with MonitoredOperation(ticker, f'batch_processing', batch_size) as batch_op:
                try:
                    # Prepare batch with context
                    start_idx = max(0, i - largest_window + 1)
                    end_idx = min(i + batch_size, n)
                    batch = new_data.iloc[start_idx:end_idx].copy()

                    logger.debug(f"Processing batch {batch_num}: rows {start_idx} to {end_idx}")
                    batch_op.add_data('batch_number', batch_num)
                    batch_op.add_data('start_idx', start_idx)
                    batch_op.add_data('end_idx', end_idx)

                    # Compute metrics with monitoring
                    with MonitoredOperation(ticker, 'compute_metrics') as metrics_op:
                        try:
                            metrics_df = compute_batch_metrics(batch)
                            metrics_op.add_records(len(metrics_df))
                        except Exception as e:
                            logger.error(f"Error computing metrics for {ticker} batch {i}-{end_idx}: {e}")
                            metrics_op.add_error(f"Metrics computation failed: {str(e)}")
                            continue

                    # Insert records with monitoring
                    actual_new_start = max(0, i)
                    actual_new_end = min(i + batch_size, n)
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
                                logger.warning(
                                    f"Skipping row with missing required values for {ticker} at {row['timestamp']}")
                                continue

                            # Insert historical record
                            historical_record = insert_historical_record_validated(
                                ticker_id=ticker,
                                timestamp=int(row['timestamp']),
                                open_price=float(row['open']),
                                close_price=float(row['close']),
                                high_price=float(row['high']),
                                low_price=float(row['low']),
                                volume=int(row['volume'])
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
                            batch_op.add_error(f"Record insertion failed: {str(e)}")
                            continue

                    batch_op.add_records(batch_inserted)
                    batch_op.add_data('batch_inserted', batch_inserted)
                    logger.info(f"Batch {batch_num} completed: {batch_inserted} records inserted for {ticker}")

                except Exception as e:
                    logger.error(f"Error processing batch {batch_num} for {ticker}: {e}")
                    batch_op.add_error(f"Batch processing failed: {str(e)}")
                    continue

        main_op.add_records(total_inserted)
        main_op.add_data('total_inserted', total_inserted)
        logger.info(f"Completed batch processing for {ticker}: {total_inserted} total records inserted")

        # Update last processed timestamp
        try:
            latest_timestamp = new_data['timestamp'].max()
            update_last_processed_timestamp(ticker, latest_timestamp)
            logger.info(f"Updated last processed timestamp for {ticker} to {latest_timestamp}")
            main_op.add_data('latest_timestamp', latest_timestamp)
        except Exception as e:
            logger.error(f"Error updating last processed timestamp for {ticker}: {e}")
            main_op.add_error(f"Timestamp update failed: {str(e)}")

        # Backfill with monitoring
        with MonitoredOperation(ticker, 'backfill') as backfill_op:
            logger.info(f"Starting backfill for {ticker} to ensure metric consistency")
            try:
                backfill_periods = min(100, total_inserted + 50)
                backfill_success = backfill_recent_metrics(
                    ticker,
                    lookback_periods=200,
                    update_periods=backfill_periods
                )

                backfill_op.add_data('success', backfill_success)
                backfill_op.add_data('backfill_periods', backfill_periods)

                if backfill_success:
                    logger.info(f"Backfill completed successfully for {ticker}")
                else:
                    logger.warning(f"Backfill had issues for {ticker}, but continuing with pipeline")

            except Exception as e:
                logger.error(f"Backfill failed for {ticker}: {e}")
                backfill_op.add_error(f"Backfill failed: {str(e)}")
                logger.warning(f"Continuing pipeline for {ticker} despite backfill failure")

        # Post-processing
        success = await run_post_processing(ticker, company_name)
        main_op.add_data('post_processing_success', success)

        if success:
            logger.info(f"✅ Complete pipeline finished successfully for {ticker}")
            return True
        else:
            logger.warning(f"⚠️ Pipeline completed with some issues for {ticker}")
            return False


async def run_post_processing(ticker: str, company_name: str) -> bool:
    """Run post-processing steps with monitoring"""
    success_flags = []

    # Aggregation and embedding with monitoring
    with MonitoredOperation(ticker, 'aggregation_embedding') as agg_op:
        try:
            from xyz.finazon_service.aggregate_embeddings import process_aggregated_embeddings
            logger.info(f"🔄 Starting multi-scale aggregation and embedding for {ticker}...")
            process_aggregated_embeddings(ticker, company_name)
            logger.info(f"✅ Aggregation and embedding complete for {ticker}")
            agg_op.add_data('status', 'success')
            success_flags.append(True)
        except Exception as e:
            logger.error(f"❌ Error during aggregation and embedding for {ticker}: {e}")
            agg_op.add_error(f"Aggregation failed: {str(e)}")
            success_flags.append(False)

    # Forecasting with monitoring
    with MonitoredOperation(ticker, 'forecasting') as forecast_op:
        try:
            logger.info(f"🔮 Starting forecast generation for {ticker}...")
            forecast_engine = ForecastEngine()
            forecast_success = forecast_engine.generate_forecasts_for_ticker(
                ticker_symbol=ticker,
                horizons=[1, 7, 30]  # 1-day, 7-day, and 30-day forecasts
            )

            forecast_op.add_data('horizons', [1, 7, 30])
            forecast_op.add_data('success', forecast_success)

            if forecast_success:
                logger.info(f"✅ Forecasts generated successfully for {ticker}")
                success_flags.append(True)
            else:
                logger.warning(f"⚠️ Forecast generation failed for {ticker}")
                forecast_op.add_error("Forecast generation returned False")
                success_flags.append(False)
        except Exception as e:
            logger.error(f"❌ Error generating forecasts for {ticker}: {e}")
            forecast_op.add_error(f"Forecast error: {str(e)}")
            success_flags.append(False)

    return any(success_flags)


async def initialize_database_async(tickers: List[str], interval: str = '30m',
                                    start_year: int = 2025, start_month: int = 8,
                                    start_day: int = 1, batch_size: int = 100):
    """Async version of database initialization"""
    results = {}

    for ticker in tickers:
        try:
            with MonitoredOperation(ticker, 'full_pipeline') as pipeline_op:
                # Step 1: Fetch and store general ticker information
                logger.info(f"Fetching general data for {ticker}")
                ticker_obj = await fetch_and_store_ticker_data(ticker)

                if not ticker_obj:
                    results[ticker] = {"status": "failed", "step": "ticker_data",
                                       "error": "Failed to fetch ticker data"}
                    pipeline_op.add_error("Ticker data fetch failed")
                    continue

                # Step 2: Fetch and store time series data
                logger.info(f"Fetching time series data for {ticker}")
                success = await update_time_series_data(
                    ticker, interval=interval, start_year=start_year,
                    start_month=start_month, start_day=start_day, batch_size=batch_size
                )

                if success:
                    results[ticker] = {"status": "success", "message": f"Successfully processed {ticker}"}
                    pipeline_op.add_data('final_status', 'success')
                    logger.info(f"Successfully processed {ticker}")
                else:
                    results[ticker] = {"status": "partial", "message": f"Partially processed {ticker}"}
                    pipeline_op.add_data('final_status', 'partial')
                    logger.warning(f"Partially processed {ticker}")

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            results[ticker] = {"status": "failed", "error": str(e)}
            continue

    return results


async def run_pipeline_async(request_data: Dict[str, Any]):
    """Run the pipeline asynchronously.

    The originating request's correlation id is threaded in via
    ``request_data['_propagated_request_id']`` (set by run_pipeline()
    before scheduling the task).  We re-install it into the ContextVar
    here so every log line + emit_event call inside the background
    pipeline carries the same ``request_id`` as the originating
    ``/run-pipeline`` request (Task 5.2 AC #1).
    """
    propagated = request_data.get('_propagated_request_id')
    rid_token = None
    if propagated:
        from xyz.observability.request_id import request_id_var
        rid_token = request_id_var.set(propagated)
    try:
        logger.info(f"🚀 Starting async pipeline execution with data: {request_data}")
        start_time = time.time()

        # Record system health before starting
        monitor.record_system_health()

        # Extract parameters
        tickers = request_data.get('tickers', ['AAPL', 'GOOG', 'TSLA'])
        interval = request_data.get('interval', '30m')
        start_year = request_data.get('start_year', 2025)
        start_month = request_data.get('start_month', 4)
        start_day = request_data.get('start_day', 1)
        batch_size = request_data.get('batch_size', 100)

        # Run pipeline
        results = await initialize_database_async(
            tickers=tickers,
            interval=interval,
            start_year=start_year,
            start_month=start_month,
            start_day=start_day,
            batch_size=batch_size
        )

        end_time = time.time()
        duration = end_time - start_time

        # Record final system health
        monitor.record_system_health()

        # Log results
        successful_tickers = [t for t, r in results.items() if r['status'] == 'success']
        failed_tickers = [t for t, r in results.items() if r['status'] == 'failed']

        logger.info(
            f"✅ Pipeline completed in {duration:.2f} seconds. "
            f"Success: {len(successful_tickers)}, Failed: {len(failed_tickers)}"
        )

        return {
            "status": "completed",
            "duration": duration,
            "results": results,
            "summary": {
                "total_tickers": len(tickers),
                "successful": len(successful_tickers),
                "failed": len(failed_tickers),
                "success_rate": len(successful_tickers) / len(tickers) if tickers else 0
            }
        }

    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        logger.exception("Full pipeline error traceback:")
        return {
            "status": "failed",
            "error": str(e),
            "duration": time.time() - start_time if 'start_time' in locals() else 0
        }
    finally:
        # Release the propagated correlation id so the next background
        # task sees a clean ContextVar (the request scope already reset).
        if rid_token is not None:
            from xyz.observability.request_id import request_id_var
            request_id_var.reset(rid_token)


# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Financial Data Pipeline API",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "pipeline": "/run-pipeline",
            "health": "/health",
            "monitoring": "/monitoring/*",
            "docs": "/docs"
        }
    }


@app.post("/run-pipeline")
async def run_pipeline(
        request: PipelineRequest,
        background_tasks: BackgroundTasks,
        _: str = Depends(verify_token),
        __: bool = Depends(check_rate_limit)
):
    """
    Trigger the financial data pipeline

    This endpoint starts the data processing pipeline in the background.
    The pipeline will:
    1. Fetch ticker information
    2. Process time series data
    3. Compute technical indicators
    4. Generate embeddings
    5. Create forecasts
    """
    try:
        # Capture the middleware-set correlation id BEFORE the request
        # scope tears down (background tasks run outside the request
        # ContextVar lifetime).  Thread it into run_pipeline_async via
        # ``_propagated_request_id`` so every log line + emit_event call
        # inside the background pipeline carries the same id — without
        # this, AC #1 fails for the pipeline's primary workload.
        from xyz.observability.request_id import get_request_id
        propagated_rid = get_request_id()

        # Convert request to dict and add metadata
        request_data = request.dict()
        request_data.update({
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': propagated_rid or f"req_{int(time.time())}",
            '_propagated_request_id': propagated_rid,
        })

        # Start pipeline in background
        background_tasks.add_task(run_pipeline_async, request_data)

        logger.info(f"🎯 Pipeline triggered successfully - running in background for tickers: {request.tickers}")

        # Return immediately with 202 Accepted
        return JSONResponse(
            status_code=202,
            content={
                "status": "accepted",
                "message": "Pipeline started successfully and running in background",
                "timestamp": request_data['timestamp'],
                "request_id": request_data['request_id'],
                "tickers": request.tickers,
                "estimated_duration": f"{len(request.tickers) * 2 - 5} minutes",
                "monitor_endpoint": "/health"
            }
        )

    except Exception as e:
        logger.error(f"Error triggering pipeline: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"Failed to trigger pipeline: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@app.get("/health", response_model=HealthResponse)
async def get_health(hours: int = 24):
    """
    Get comprehensive pipeline health status

    Returns detailed health metrics including:
    - Overall success rates
    - Operation breakdowns
    - System resource usage
    - Error summaries
    """
    try:
        # Record current system health
        monitor.record_system_health()

        health_data = monitor.get_health_summary(hours=hours)
        return HealthResponse(**health_data)
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving health status: {str(e)}")


@app.get("/monitoring/ticker/{ticker_symbol}", response_model=TickerPerformanceResponse)
async def get_ticker_performance(ticker_symbol: str, hours: int = 24):
    """Get performance metrics for a specific ticker"""
    try:
        performance_data = monitor.get_ticker_performance(ticker_symbol, hours=hours)
        return TickerPerformanceResponse(**performance_data)
    except Exception as e:
        logger.error(f"Error getting ticker performance for {ticker_symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving ticker performance: {str(e)}")


# ---------------------------------------------------------------------------
# Agent routes
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    firm_id: int
    account_id: Optional[int] = None
    symbol: Optional[str] = None
    brief: Optional[str] = None
    actor_user_id: Optional[int] = None


@app.post("/agents/research")
def research_endpoint(
    body: ResearchRequest,
    _: str = Depends(verify_token),
):
    """Invoke the RESEARCH subagent for a ticker, account, or free-form brief.

    At least one of ``symbol`` or ``brief`` must be provided; otherwise a 422
    is returned.  The agent gathers context from Polygon + the engine DB,
    calls Claude, hashes the result, and persists a ``research.artifact``
    event.  Returns the ResearchArtifact as JSON.
    """
    from xyz.agents.research import ResearchAgent
    from xyz.agents.schemas import ResearchInput
    from xyz.agents.lib.anthropic_client import AnthropicClient
    from xyz.polygon_service.options_client import OptionsClient
    from xyz.tenant.db import get_tenant_session

    if not body.symbol and not body.brief:
        raise HTTPException(status_code=422, detail="At least one of 'symbol' or 'brief' must be provided.")

    agent = ResearchAgent(
        anthropic_client=AnthropicClient(),
        options_client=OptionsClient(),
        db_session_factory=get_tenant_session,
    )
    try:
        artifact = agent.run(ResearchInput(**body.model_dump()))
    except json.JSONDecodeError as exc:
        # Claude returned non-JSON (refusal, content-filter, truncation).
        # Surface as 502 so the caller can distinguish upstream LLM failure
        # from a bad request.
        raise HTTPException(
            status_code=502,
            detail=f"LLM response was not valid JSON: {exc.msg}",
        ) from exc
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM response missing required field: {exc}",
        ) from exc
    return artifact.model_dump(mode="json")


# ---------------------------------------------------------------------------
# AUTHOR + DSL-validation routes (Task 4.2)
# ---------------------------------------------------------------------------

class AuthorRequest(BaseModel):
    firm_id: int
    brief: str = Field(..., min_length=1)
    actor_user_id: Optional[int] = None
    target_account_ids: List[int] = Field(default_factory=list)


@app.post("/agents/author")
def author_endpoint(
    body: AuthorRequest,
    _: str = Depends(verify_token),
):
    """Invoke the AUTHOR subagent.

    Body: ``{firm_id, brief, actor_user_id?, target_account_ids?}``.

    Returns the AuthorArtifact (``{template, dsl, rationale, generated_at,
    content_hash}``).  Persists a ``strategy.draft`` event side-effect.

    Error mapping
    -------------
    - 422 on body validation (FastAPI default).
    - 502 on upstream LLM failures: invalid JSON, missing required field,
      or DSL that fails schema validation.  The 502 distinguishes "the
      model misbehaved" from "the caller sent bad input".
    """
    from xyz.agents.author import AuthorAgent
    from xyz.agents.schemas import AuthorInput
    from xyz.agents.lib.anthropic_client import AnthropicClient
    from xyz.tenant.db import get_tenant_session

    agent = AuthorAgent(
        anthropic_client=AnthropicClient(),
        db_session_factory=get_tenant_session,
    )
    try:
        artifact = agent.run(AuthorInput(**body.model_dump()))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM response was not valid JSON: {exc.msg}",
        ) from exc
    except (KeyError, ValueError) as exc:
        # KeyError → Claude omitted a required top-level key.
        # ValueError → DSL failed JSON-Schema validation (the agent guards
        # the audit log from holding a malformed draft).
        raise HTTPException(
            status_code=502,
            detail=f"AUTHOR produced an invalid response: {exc}",
        ) from exc
    return artifact.model_dump(mode="json")


class BacktestRequest(BaseModel):
    strategy_id: int
    strategy_version: int
    firm_id: int
    actor_user_id: Optional[int] = None
    start_date: str  # ISO date YYYY-MM-DD
    end_date: str
    dsl: Dict[str, Any]


@app.post("/agents/backtest")
def backtest_endpoint(
    body: BacktestRequest,
    _: str = Depends(verify_token),
):
    """Invoke the BACKTEST subagent (Task 4.3).

    Body: ``{strategy_id, strategy_version, firm_id, actor_user_id?,
    start_date, end_date, dsl}``.

    Returns the BacktestArtifact (``{strategy_id, strategy_version,
    firm_id, start_date, end_date, metrics, n_trades, content_hash,
    generated_at}``) and emits a ``backtest.result`` event.

    The artifact is the immutable, hashed audit row required by §10 of
    the north-star spec.  The engine does NOT write into the server's
    ``backtest_results`` table — the caller is responsible for POSTing
    the result to server's ``POST /backtests`` to persist it.

    Error mapping
    -------------
    - 422 on body validation (FastAPI default).
    - 400 if the DSL is unsupported (e.g. ``cash_secured_put`` in v1)
      or if no chain data exists for the underlying in the window.
    """
    from datetime import date as _date
    from xyz.agents.backtest import BacktestAgent
    from xyz.agents.schemas import BacktestInput
    from xyz.tenant.db import get_tenant_session

    try:
        start = _date.fromisoformat(body.start_date)
        end = _date.fromisoformat(body.end_date)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"start_date / end_date must be ISO YYYY-MM-DD: {exc}",
        ) from exc

    agent = BacktestAgent(db_session_factory=get_tenant_session)
    try:
        artifact = agent.run(BacktestInput(
            strategy_id=body.strategy_id,
            strategy_version=body.strategy_version,
            firm_id=body.firm_id,
            actor_user_id=body.actor_user_id,
            start_date=start,
            end_date=end,
            dsl=body.dsl,
        ))
    except ValueError as exc:
        # DSL unsupported template, inverted dates, empty chain — all map
        # to client-error 400.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return artifact.model_dump(mode="json")


class ProposeRequest(BaseModel):
    firm_id: int
    deployment_id: int
    actor_user_id: Optional[int] = None


@app.post("/agents/propose")
def propose_endpoint(
    body: ProposeRequest,
    _: str = Depends(verify_token),
):
    """Invoke the PROPOSE subagent (Task 4.4).

    Body: ``{firm_id, deployment_id, actor_user_id?}``.

    Returns the ProposeArtifact (``{deployment_id, firm_id, tickets,
    reason, generated_at}``).  A trigger-miss returns ``tickets=[]``
    with a ``reason`` string — NOT an error — per acceptance criterion
    1.  Each emitted ticket also lands in the engine's audit chain as
    a ``ticket.proposed`` event.

    The engine does NOT write into the server's ``trades`` table — the
    caller (dashboard or orchestrator) POSTs each returned ticket to
    server's ``POST /trades`` to persist it.  This mirrors the BACKTEST
    transport contract from Task 4.3.

    Error mapping
    -------------
    - 422 on body validation (FastAPI default).
    - 400 on hard data-shape errors raised as ValueError by the agent
      (these indicate a bug upstream — e.g. a deployment whose
      strategy_id is dangling — not a trigger miss).
    - 502 on any other internal error (DB constraint, event-chain
      failure, etc.) so a raw stack trace is not leaked to the caller.
    """
    from xyz.agents.propose import ProposeAgent
    from xyz.agents.schemas import ProposeInput
    from xyz.tenant.db import get_tenant_session

    agent = ProposeAgent(db_session_factory=get_tenant_session)
    try:
        artifact = agent.run(ProposeInput(**body.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("PROPOSE internal failure")
        raise HTTPException(
            status_code=502,
            detail=f"propose_internal_error: {type(exc).__name__}",
        ) from exc
    return artifact.model_dump(mode="json")


class ValidateDslRequest(BaseModel):
    dsl: Dict[str, Any]


@app.post("/agents/validate-dsl")
def validate_dsl_endpoint(
    body: ValidateDslRequest,
    _: str = Depends(verify_token),
):
    """Validate a Strategy DSL against the v1 schema.

    Returns ``{valid: bool, errors: [str, ...]}``.  Always 200 — even when
    the DSL is invalid — so callers can branch on the body without trying
    to distinguish "real failure" from "DSL has errors".  The server-side
    PATCH hook in ``server-fastapi-wt/app/routes/strategies.py`` consumes
    this endpoint to gate dsl_json updates.
    """
    # Lazy import so route registration doesn't pull jsonschema for
    # unrelated endpoints.
    from xyz.dsl.validate import validate_dsl

    valid, errors = validate_dsl(body.dsl)
    return {"valid": valid, "errors": errors}


if __name__ == '__main__':
    logger.info("🚀 STARTING FINANCIAL DATA PIPELINE API")


    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8900,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("👋 Shutting down...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)