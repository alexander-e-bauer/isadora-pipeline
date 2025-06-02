import requests
import time
from polygon import RESTClient
from polygon.rest.models import TickerNews
from xyz.llm.embedding_generator import get_embedding
from tenacity import retry, stop_after_attempt, wait_exponential, wait_exponential_jitter
import requests
import os
import dotenv
import re
import pandas as pd
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from config import OAI
import tiktoken
dotenv.load_dotenv()
dummy_embeddings = False
# Set up the API URL and the API key

polygon_api_key = os.getenv('POLYGON_KEY')
polygon_client = RESTClient(api_key=polygon_api_key)
av_api_key = os.getenv('AV_KEY')

@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(max=60, initial=5, exp_base=5))
def get_new_ticker_gen_data(ticker_symbol):
    try:
        # Polygon API URL
        polygon_url = f'https://api.polygon.io/v3/reference/tickers/{ticker_symbol}'

        # Send the GET request to Polygon API
        polygon_response = requests.get(polygon_url, params={"apiKey": polygon_api_key})
        polygon_response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        # Parse Polygon API response
        polygon_data = polygon_response.json()
        polygon_data = polygon_data.get("results")
        name = polygon_data.get("name")

        return polygon_data, name

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the API request: {e}")
    except ValueError as ve:
        print(f"Data validation error: {ve}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

    # Return None if there was an error
    return None


@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(max=60, initial=5, exp_base=5))
def get_ticker_news(ticker_symbol, limit=10, published_from=None, published_to=None):
    """
    Fetch news articles for a given ticker symbol with optional date filters.

    Parameters:
        ticker_symbol (str): The stock ticker symbol (e.g., "AAPL" or "TSLA").
        limit (int): The maximum number of news articles to fetch.
        published_from (str): (Optional) The earliest date to fetch articles (format: YYYY-MM-DD or ISO 8601).
        published_to (str): (Optional) The latest date to fetch articles (format: YYYY-MM-DD or ISO 8601).

    Returns:
        list: A list of news articles (TickerNews objects).
    """
    try:
        news = []

        # Build filters for the `published_utc` parameter
        filters = {}
        if published_from:
            filters["published_utc_gte"] = published_from  # Greater than or equal to
        if published_to:
            filters["published_utc_lte"] = published_to  # Less than or equal to

        # Fetch articles with filters
        for index, n in enumerate(polygon_client.list_ticker_news(
                ticker=ticker_symbol,
                order="asc",
                limit="50",  # Fetch in bulk; we'll limit manually
                sort="published_utc",
                **filters,  # Apply date filters dynamically
        )):
            if isinstance(n, TickerNews):
                news.append(n)
                print(f"{n.published_utc:<25}{n.title:<15}")

            # Respect the limit
            if index + 1 >= limit:
                break  # Stop after fetching the desired number of articles
        time.sleep(6)

        return news

    except Exception as ex:
        print(f"An unexpected error occurred while fetching news for {ticker_symbol}: {ex}")
        raise  # Retry due to the @retry decorator


def num_tokens(text: str, model: str = OAI.gpt4o) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(str(text)))


def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str, str]:
    """Split a string in two, on a delimiter, trying to balance tokens on each side."""
    for i in string:
        chunks = i.split(delimiter)
        if len(chunks) == 1:
            return [string, ""]  # no delimiter found
        elif len(chunks) == 2:
            return chunks  # no need to search for halfway point
        else:
            total_tokens = num_tokens(string)
            halfway = total_tokens // 2
            best_diff = halfway
            for i, chunk in enumerate(chunks):
                left = delimiter.join(chunks[: i + 1])
                left_tokens = num_tokens(left)
                diff = abs(halfway - left_tokens)
                if diff >= best_diff:
                    break
                else:
                    best_diff = diff
            left = delimiter.join(chunks[:i])
            right = delimiter.join(chunks[i:])
            return [left, right]


def truncated_string(
    string: str,
    model: str,
    max_tokens: int,
    print_warning: bool = True,
) -> str:
    """Truncate a string to a maximum number of tokens."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(str(string))
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated_string


def split_strings_from_subsection(
    subsection,
    max_tokens: int = 8192,
    model: str = OAI.gpt4o,
    max_recursion: int = 5,
) -> list[str]:
    """
    Split a subsection into a list of subsections, each with no more than max_tokens.
    Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
    """
    print(subsection)
    string = subsection
    num_tokens_in_string = num_tokens(string)
    # if length is fine, return string
    if num_tokens_in_string <= max_tokens:
        return [string]
    # if recursion hasn't found a split after X iterations, just truncate
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
    # otherwise, split in half and recurse
    else:
        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimiter(string, delimiter=delimiter)
            if left == "" or right == "":
                # if either half is empty, retry with a more fine-grained delimiter
                continue
            else:
                # recurse on each half
                results = []
                for half in [left, right]:
                    half_subsection = (subsection, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results
    # otherwise no split was found, so just truncate (should be very rare)
    return [truncated_string(string, model=model, max_tokens=max_tokens)]


def extract_article_data(articles):
    """
    Extracts specific fields (title, article_url, insights, publisher) from a list of TickerNews articles.

    Parameters:
        articles (list): List of TickerNews objects.

    Returns:
        list: List of dictionaries containing the extracted data.
    """
    extracted_data = []

    for article in articles:
        # Extract relevant fields
        article_data = {
            "title": article.title,
            "article_url": article.article_url,
            "insights": [
                {
                    "sentiment": insight.sentiment,
                    "reasoning": insight.sentiment_reasoning,
                    "ticker": insight.ticker,
                }
                for insight in article.insights
            ] if article.insights else "No insights available",
            "publisher": {
                "name": article.publisher.name,
                "homepage_url": article.publisher.homepage_url,
            } if article.publisher else "No publisher details available",
        }
        extracted_data.append(article_data)

    return extracted_data





# Function to scrape text using Playwright
def scrape_clean_text(url):
    """
    Scrapes and cleans webpage text, retaining only relevant article content: headline, author, date, and main body.
    Parameters:
        url (str): The URL of the webpage to scrape.
    Returns:
        str: Cleaned article text for chatbot use.
    """
    import re
    from playwright.sync_api import sync_playwright
    from bs4 import BeautifulSoup

    def is_irrelevant_line(line):
        """Identify if a line of text is irrelevant based on patterns."""
        patterns_to_skip = [
            r"enable accessibility|accessibility|accessibility menu",
            r"\.+%|\|.*?\|",  # Excess symbols
            r"(click here|read more|become a member|join the motley fool|premium investing services|learn more|don't miss out|available when you join)",
            r"(sponsored|advertisement|related articles|view premium services|current price|today's change|arrow-thin-down|arrow-left|arrow-right|daily stock gainers|daily stock losers|most active stocks)",
            r"^\s*[-—]+\s*$",  # Horizontal lines or single dashes
            r"^\s*$",  # Empty lines
        ]
        for pattern in patterns_to_skip:
            if re.search(pattern, line, flags=re.IGNORECASE):
                return True
        return False

    def extract_main_content(soup):
        """
        Try to extract the main article content.
        Fallback to largest <article>, <main>, or <div> block with lots of text.
        """
        # Try common article containers
        for selector in ["article", "main", "div.article-body", "div#main-article", "div#article", "section.article"]:
            tag = soup.select_one(selector)
            if tag and len(tag.get_text(strip=True)) > 200:
                return tag

        # Fallback: find the largest block of text
        candidates = soup.find_all(['article', 'main', 'section', 'div'], recursive=True)
        best = max(candidates, key=lambda t: len(t.get_text(strip=True)), default=None)
        return best if best and len(best.get_text(strip=True)) > 200 else soup

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=120000)
            page.wait_for_timeout(3000)  # Wait for lazy-loaded content
            content = page.content()
            browser.close()

            soup = BeautifulSoup(content, "html.parser")

            # Remove irrelevant tags
            for tag in soup(["script", "style", "meta", "nav", "footer", "aside", "iframe", "noscript", "link", "header"]):
                tag.decompose()

            # Extract headline
            headline = ""
            for selector in ["h1", ".article-title", ".headline"]:
                h = soup.select_one(selector)
                if h and len(h.get_text(strip=True)) > 5:
                    headline = h.get_text(strip=True)
                    break

            # Extract author and date if present
            author = ""
            date = ""
            # Try common selectors for author/date
            if soup.find(attrs={"itemprop": "author"}):
                author = soup.find(attrs={"itemprop": "author"}).get_text(strip=True)
            if soup.find(attrs={"itemprop": "datePublished"}):
                date = soup.find(attrs={"itemprop": "datePublished"}).get_text(strip=True)
            # Fallback: regex search in text for date formats
            raw_text = soup.get_text(separator="\n", strip=True)
            date_match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2},? \d{4}", raw_text)
            if date_match:
                date = date_match.group(0)

            # Extract main article content
            main_content = extract_main_content(soup)
            if main_content:
                article_lines = []
                for line in main_content.get_text(separator="\n", strip=True).splitlines():
                    line = line.strip()
                    if len(line) > 0 and not is_irrelevant_line(line):
                        article_lines.append(line)
                article_body = "\n".join(article_lines)
            else:
                article_body = "No article content found."

            # Compose clean output
            output = ""
            if headline:
                output += f"{headline}\n"
            if author:
                output += f"By: {author}\n"
            if date:
                output += f"Date: {date}\n"
            output += article_body

            # Final cleanup: remove repeated whitespace
            output = re.sub(r"\n{3,}", "\n\n", output)
            return output.strip() or "No relevant content found."

    except Exception as e:
        return f"Error during scraping: {str(e)}"



def create_news_summary_for_period(articles, tokens):
    """
    Returns a single summary string for all articles in the period.
    """
    texts = []
    for article in articles:
        title = article.get("title", "")
        url = article.get("article_url", "")
        scraped_text = scrape_clean_text(url)
        # You could use only the headline, or headline + first 200 chars, etc.
        snippet = scraped_text.replace('\n', ' ')
        response = OAI.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Summaraize the provided article, focus on the most important information. "
                                    "DO NOT provide any filler, conversation/explanations, markdown, "
                                    "or any other extras; just the article summary that is all."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{snippet}"
                        }
                    ]
                }
            ],
            response_format={
                "type": "text"
            },
            temperature=1,
            max_completion_tokens=tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        generated_text = response.choices[0].message.content
        texts.append(f"{generated_text}")
    summary = "\n".join(texts)
    return summary


# Run the function
if __name__ == "__main__":
    # Example: Fetch news for Apple (AAPL) with a date filter
    ticker_symbol = "AAPL"
    from_date = "2025-04-20"  # Articles published on or after this date
    to_date = "2025-05-21"    # Articles published on or before this date

    # Fetch news articles using your existing function
    articles = get_ticker_news(ticker_symbol, limit=1, published_from=from_date, published_to=to_date)

    # Extract structured data from the articles
    extracted_data = extract_article_data(articles)
    print("Extracted article metadata:")
    for article in extracted_data:
        print(article)
        print("-" * 40)

    # Generate a summary for all articles in the period
    print("\n=== News Summary for Period ===")
    summary = create_news_summary_for_period(extracted_data)
    print(summary)
