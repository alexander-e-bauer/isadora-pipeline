import finnhub

from finnhub import Client as FinnhubClient
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
import time

finnhub_client = finnhub.Client(api_key="d10e721r01qlsac8iah0d10e721r01qlsac8iahg")




@retry(stop=stop_after_attempt(10), wait=wait_exponential_jitter(max=240, initial=2, exp_base=2))
def get_ticker_news(ticker_symbol, limit=10, published_from=None, published_to=None):
    """
    Fetch news articles for a given ticker symbol with optional date filters.

    Parameters:
        ticker_symbol (str): The stock ticker symbol (e.g., "AAPL" or "TSLA").
        limit (int): The maximum number of news articles to fetch.
        published_from (str): (Optional) The earliest date to fetch articles (format: YYYY-MM-DD).
        published_to (str): (Optional) The latest date to fetch articles (format: YYYY-MM-DD).

    Returns:
        list: A list of news articles retrieved from Finnhub.
    """
    try:
        # Validate required input dates since Finnhub's API expects them
        if not published_from or not published_to:
            raise ValueError("Both 'published_from' and 'published_to' must be specified in YYYY-MM-DD format.")

        # Fetch articles from the Finnhub API
        response = finnhub_client.company_news(
            ticker_symbol,
            _from=published_from,
            to=published_to
        )

        # Limit the articles if needed
        news = response[:limit]  # Slice the response to match the limit

        # Print the article titles for confirmation
        for article in news:
            print(f"{article.get('datetime')} - {article.get('headline')}")

        return news

    except Exception as e:
        print(f"Error while fetching news for {ticker_symbol}: {e}")
        raise

if __name__ == "__main__":
    news = get_ticker_news(
        ticker_symbol="AAPL",
        limit=5,
        published_from="2025-06-01",
        published_to="2025-06-10"
    )

    for article in news:
        print(article)  # Print each news article for further inspection

