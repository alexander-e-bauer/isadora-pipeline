import logging
from datetime import datetime
from celery_app import celery_app
from xyz.finazon_service.sql_service import (
    Session,
    check_for_ticker,
)
from app import update_time_series_data, fetch_and_store_ticker_data
from tasks.tasks import update_ticker_data
from config import logger




TICKERS = ['AAPL', 'GOOGL', 'TSLA']

def ensure_ticker_exists(ticker):
    """Make sure the ticker exists in the database"""
    try:
        session = Session()
        if not check_for_ticker(ticker):
            logger.info(f"Initializing new ticker: {ticker}")
            fetch_and_store_ticker_data(ticker)
        session.close()
        return True
    except Exception as e:
        logger.error(f"Error ensuring ticker {ticker} exists: {e}")
        return False

@celery_app.task
def update_all_tickers():
    """Update data for all tickers"""
    logger.info("Starting update for all tickers")
    for ticker in TICKERS:
        try:
            if ensure_ticker_exists(ticker):
                logger.info(f"Queuing update for {ticker}")
                update_ticker_data.delay(ticker)
            else:
                logger.error(f"Failed to ensure ticker exists: {ticker}")
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {e}")
    logger.info("Finished queuing updates for all tickers")


def main():
    """Main function to run the update process"""
    logger.info("Starting stock update service")
    try:
        update_all_tickers.delay()
        celery_app.worker_main(['worker', '--loglevel=info'])
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    logger.info("Heroku Scheduler triggered update_stocks.py")
    main()
