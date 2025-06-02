from xyz.finazon_service.sql_service import check_for_ticker, get_last_processed_timestamp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_database():
    tickers = ['AAPL', 'GOOGL']

    for ticker in tickers:
        logger.info(f"\nChecking {ticker}:")

        # Check if ticker exists
        exists = check_for_ticker(ticker)
        logger.info(f"Ticker exists in database: {exists}")

        if exists:
            # Get last processed timestamp
            last_timestamp = get_last_processed_timestamp(ticker)
            logger.info(f"Last processed timestamp: {last_timestamp}")

            # You could add more checks here based on what data you expect


if __name__ == "__main__":
    verify_database()
