from app import fetch_and_store_ticker_data, update_time_series_data
import logging
import time
import pandas as pd

from xyz.finazon_service.retrive_data import FinazonService
import metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_database():
    tickers = ['AAPL', 'GOOG', 'TSLA']

    for ticker in tickers:
        try:
            # Step 1: Fetch and store general ticker information
            logger.info(f"Fetching general data for {ticker}")
            fetch_and_store_ticker_data(ticker)
            time.sleep(2)

            # Step 2: Fetch and store time series data
            logger.info(f"Fetching time series data for {ticker}")
            update_time_series_data(ticker, interval='15m')


            logger.info(f"Successfully processed {ticker}")

        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            continue

if __name__ == "__main__":
    logger.info("Starting database initialization...")
    initialize_database()
    logger.info("Database initialization completed")
