import logging
from celery_app import celery_app
from xyz.finazon_service.sql_service import session, Ticker
from app import update_time_series_data

logger = logging.getLogger(__name__)


# Apply rate limiting to the task - 5 tasks per minute (300 per hour)
@celery_app.task(rate_limit='5/m')
def update_ticker_data(ticker_symbol):
    """
    Celery task to update data for a single ticker.
    """
    try:
        logger.info(f"Updating data for ticker: {ticker_symbol}")
        update_time_series_data(ticker_symbol, interval="15m")
        logger.info(f"Successfully updated data for ticker: {ticker_symbol}")
    except Exception as e:
        logger.error(f"Error updating ticker {ticker_symbol}: {e}")


@celery_app.task
def update_all_tickers():
    """
    Celery task to trigger parallel updates for all tickers.
    """
    logger.info("Starting parallel updates for all tickers.")
    tickers = session.query(Ticker).all()

    # Calculate total number of tickers
    total_tickers = len(tickers)
    logger.info(f"Found {total_tickers} tickers to update")

    for ticker in tickers:
        # Spawn a separate task for each ticker
        update_ticker_data.delay(ticker.symbol)

    logger.info("All ticker update tasks have been dispatched.")
