import requests
import logging

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_endpoints.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Base URL of the Flask application
BASE_URL = "http://127.0.0.1:5000"  # Update this if your app is running on a different host or port

def test_fetch_ticker_data(ticker):
    """
    Test the /fetch/<ticker> endpoint.
    """
    url = f"{BASE_URL}/fetch/{ticker}"
    logger.info(f"Testing endpoint: {url}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            logger.info(f"Success: {response.text}")
        else:
            logger.error(f"Failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Error testing /fetch/{ticker}: {e}")

def test_update_ticker_data(ticker):
    """
    Test the /update/<ticker> endpoint.
    """
    url = f"{BASE_URL}/update/{ticker}"
    logger.info(f"Testing endpoint: {url}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            logger.info(f"Success: {response.text}")
        else:
            logger.error(f"Failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Error testing /update/{ticker}: {e}")

if __name__ == "__main__":
    # Tickers to test
    tickers = ["AAPL", "TSLA", "GOOGL"]
    tickers = ["AAPL"]
    # Test /fetch/<ticker> endpoint
    for ticker in tickers:
        test_fetch_ticker_data(ticker)

    # Test /update/<ticker> endpoint
    for ticker in tickers:
        test_update_ticker_data(ticker)