import requests
import time
from datetime import timedelta
from polygon import RESTClient
from polygon.rest.models import TickerNews
import spacy

from xyz.llm.embedding_generator import get_embedding
from tenacity import retry, stop_after_attempt, wait_exponential, wait_exponential_jitter
import requests
import os
import dotenv

import re
import pandas as pd
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from config import OAI, logger
import tiktoken
dotenv.load_dotenv()
dummy_embeddings = False
# Set up the API URL and the API key

polygon_api_key = os.getenv('POLYGON_KEY')
polygon_client = RESTClient(api_key=polygon_api_key)
av_api_key = os.getenv('AV_KEY')

scrape_cache = {}
# Enhanced api_service.py improvements
import time
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RateLimitedSession:
    def __init__(self, calls_per_minute=5):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0

        # Configure session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get(self, *args, **kwargs):
        # Rate limiting
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        self.last_call = time.time()
        return self.session.get(*args, **kwargs)


# Global rate-limited session
api_session = RateLimitedSession(calls_per_minute=5)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((requests.RequestException, ConnectionError))
)
def get_new_ticker_gen_data(ticker_symbol):
    """Enhanced version with better error handling"""
    try:
        polygon_url = f'https://api.polygon.io/v3/reference/tickers/{ticker_symbol}'

        response = api_session.get(
            polygon_url,
            params={"apiKey": polygon_api_key},
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        results = data.get("results")

        if not results:
            logger.warning(f"No results found for ticker {ticker_symbol}")
            return None, None

        name = results.get("name", f"Company {ticker_symbol}")
        return results, name

    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching data for {ticker_symbol}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for {ticker_symbol}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error for {ticker_symbol}: {e}")
        raise

@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(max=60, initial=5, exp_base=5))
def get_new_ticker_gen_data_dep(ticker_symbol):
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


@retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(max=120, initial=10, exp_base=10))
def get_ticker_news_polygon(ticker_symbol, limit=10, published_from=None, published_to=None):
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
                print(f"Fetched article: {n.published_utc:<25}{n.title:<15}")

            # Respect the limit
            if index + 1 >= limit:
                break  # Stop after fetching the desired number of articles
        time.sleep(6)

        return news

    except Exception as ex:
        print(f"An unexpected error occurred while fetching news for {ticker_symbol}: {ex}")
        raise  # Retry due to the @retry decorator


# When you need article text:
def get_article_text(url):
    if url not in scrape_cache:
        scrape_cache[url] = scrape_clean_text_with_timeout(url)
    return scrape_cache[url]


def has_relevant_keywords(article, ticker_symbol, company_name, scrape_cache):
    title = getattr(article, "title", "") or ""
    description = getattr(article, "description", "") or ""
    url = getattr(article, "article_url", "") or ""
    tickers = getattr(article, "tickers", [])
    publisher = getattr(article, "publisher", None)

    # Accept if ticker is in the tickers field
    if ticker_symbol in tickers:
        return True

    # Accept if company name or ticker in title or description
    if (ticker_symbol in title or company_name.lower() in title.lower() or
        ticker_symbol in description or company_name.lower() in description.lower()):
        return True

    # Accept if company/ticker appears at least once in body
    body = scrape_cache.get(url, "")
    if (ticker_symbol in body or company_name.lower() in body.lower()):
        return True

    # Accept if publisher is known financial news source
    if publisher and publisher.name.lower() in {"barron's", "marketwatch", "bloomberg", "reuters", "cnbc", "yahoo finance"}:
        return True

    # LOG WHY IT FAILED
    print(f"Filtered out: {title[:80]} | {url} | tickers: {tickers} | publisher: {getattr(publisher, 'name', None)}")
    return False


def filter_articles(articles, ticker_symbol, company_name, scrape_cache):
    filtered = []
    for article in articles:
        # Use the correct attribute name for url
        title = getattr(article, "title", None)
        url = getattr(article, "article_url", None)
        if not url:
            print("Article missing 'article_url' attribute:", article)
            continue

        #body = get_article_text(url)  # This will handle the cache

        # Check if the article passes the filters
        if not has_relevant_keywords(article, ticker_symbol, company_name, scrape_cache):
            print(f"has relevant keywords fail {title}")
            continue


        filtered.append(article)

    print(f"Filtered Articles: {filtered}")
    return filtered


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
    #print(subsection)
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
    Extracts specific fields (title, article_url, publisher) from a list of TickerNews articles.
    """
    extracted_data = []
    for article in articles:
        article_data = {
            "title": getattr(article, "headline", ""),
            "article_url": getattr(article, "article_url", ""),
            "publisher": {
                "name": getattr(getattr(article, "publisher", None), "name", ""),
                "homepage_url": getattr(getattr(article, "publisher", None), "homepage_url", ""),
            } if getattr(article, "publisher", None) else "No publisher details available",
        }
        extracted_data.append(article_data)
    return extracted_data


import concurrent.futures

def scrape_clean_text_with_timeout(url, timeout=60):
    """
    Calls scrape_clean_text but skips if it takes longer than `timeout` seconds.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(scrape_clean_text, url)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"Timeout scraping {url}")
            return "Timeout occurred"
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return f"Error: {e}"




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
        """
        Identify if a line of text is irrelevant based on patterns,
        by using regex for keywords and unnecessary content.
        """
        patterns_to_skip = [
            r"enable accessibility|accessibility menu",  # Accessibility messages
            r"\.+%|\|.*?\|",  # Percentages or excess symbols
            r"(click here|read more|become a member|join the motley fool|premium investing services|learn more|don't miss out|available when you join)",
            r"(sponsored|advertisement|related articles|view premium services|current price|today's change|arrow-thin-down|arrow-left|arrow-right|daily stock gainers|daily stock losers|most active stocks)",
            r"(upgrade to read|access premium news articles|already have a subscription?|privacy policy|terms and conditions|reference ID)",
            # Subscription and disclaimers
            r"(press & hold to confirm|access denied|validate with expert research|you are not a bot|human not bot)",
            # Bot or confirmation messages
            r"^\s*[-—]+\s*$",  # Horizontal lines or single dashes
            r"^\s*$",  # Empty lines or whitespace
            r"\bad\b|\bundo\b",  # Ad labels like 'Ad', 'Undo'
            r"(has relevant keywords fail|error during scraping|timeout exceeded)",  # Failure-related phrases
            r"(premium|subscription plan required|sign in|upgrade|more info)",  # Premium/paywall mentions
            r"(terms and privacy|privacy dashboard|more info|contact support)"  # Miscellaneous site disclaimers
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
            page.goto(url, timeout=30000)
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


def create_weekly_summary_for_period(weekly_text, company_name, tokens):
    response = OAI.client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Summaraize the provided article summaries into a comprehensive weekly summary,"
                                " focus on the most important information. "
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
                        "text": f"Given the following article, provide a concise summary focusing only on information relevant to {company_name}." \
                                "Ignore information not related to {company_name}. " \
                                "Article: " \
                                f"{weekly_text} " \
                                "Relevant summary:"

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
    print(f"Generated Weekly Summary: {generated_text}")
    return generated_text


def create_news_summary_for_period(ticker_symbol, company_name, start_date, end_date, articles=None):
    """
    Create a comprehensive news summary for a given period with proper content extraction.
    """
    if not articles:
        articles = get_ticker_news_polygon(
            ticker_symbol,
            limit=50,
            published_from=start_date,
            published_to=end_date
        )

    if not articles:
        return f"No news articles found for {company_name} ({ticker_symbol}) from {start_date} to {end_date}."

    # Filter articles for relevance
    filtered_articles = filter_articles(articles, ticker_symbol, company_name, scrape_cache)

    if not filtered_articles:
        return f"No relevant news articles found for {company_name} ({ticker_symbol}) from {start_date} to {end_date}."

    print(f"Processing {len(filtered_articles)} filtered articles for summary...")

    # Extract article data with content
    article_summaries = []
    for article in filtered_articles:
        try:
            # Get basic article info
            title = getattr(article, "title", "No title")
            description = getattr(article, "description", "")
            url = getattr(article, "article_url", "")
            published = getattr(article, "published_utc", "")

            # Try to get article content
            article_content = ""
            if url:
                try:
                    article_content = get_article_text(url)
                    if article_content and len(article_content) > 100:
                        # Truncate very long articles to manage token limits
                        article_content = article_content[:2000] + "..." if len(
                            article_content) > 2000 else article_content
                except Exception as e:
                    print(f"Could not scrape content from {url}: {e}")
                    article_content = description  # Fallback to description

            # Create article summary
            article_info = f"""
Article: {title}
Published: {published}
Description: {description}
Content: {article_content if article_content else "Content not available"}
URL: {url}
---"""

            article_summaries.append(article_info)

        except Exception as e:
            print(f"Error processing article: {e}")
            continue

    if not article_summaries:
        return f"Could not extract content from articles for {company_name} ({ticker_symbol})."

    # Combine all articles
    combined_articles = "\n".join(article_summaries)

    # Create the prompt for AI summarization
    prompt = f"""
Please analyze the following news articles about {company_name} ({ticker_symbol}) from {start_date} to {end_date} and create a comprehensive summary.

Focus on:
1. Key developments and announcements
2. Financial performance or guidance
3. Product launches or updates
4. Leadership changes
5. Regulatory or legal matters
6. Market reactions and analyst opinions
7. Strategic partnerships or acquisitions

Articles to analyze:
{combined_articles}

Please provide a concise but comprehensive summary of the most important developments during this period. If there are no significant developments, please state that clearly but still mention any minor news or market activity that occurred.
"""

    try:
        # Check token count and truncate if necessary
        token_count = num_tokens(prompt)
        if token_count > 15000:  # Leave room for response
            print(f"Prompt too long ({token_count} tokens), truncating articles...")
            # Take first half of articles
            truncated_articles = article_summaries[:len(article_summaries) // 2]
            combined_articles = "\n".join(truncated_articles)
            prompt = f"""
Please analyze the following news articles about {company_name} ({ticker_symbol}) from {start_date} to {end_date} and create a comprehensive summary.

Focus on:
1. Key developments and announcements
2. Financial performance or guidance
3. Product launches or updates
4. Leadership changes
5. Regulatory or legal matters
6. Market reactions and analyst opinions
7. Strategic partnerships or acquisitions

Articles to analyze (truncated due to length):
{combined_articles}

Please provide a concise but comprehensive summary of the most important developments during this period.
"""

        # Get AI summary
        response = OAI.chat.completions.create(
            model=OAI.gpt4o,
            messages=[
                {"role": "system",
                 "content": "You are a financial news analyst. Provide clear, factual summaries of news events."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        summary = response.choices[0].message.content.strip()
        print(f"Generated summary: {summary[:200]}...")
        return summary

    except Exception as e:
        print(f"Error generating AI summary: {e}")
        # Fallback to basic summary
        titles = [getattr(article, "title", "No title") for article in filtered_articles[:5]]
        return f"Found {len(filtered_articles)} articles about {company_name} including: {'; '.join(titles[:3])}..."


def fetch_news_for_period_with_flags(ticker_symbol, company_name, start_date, end_date):
    """
    Fetch and summarize news for a specific period with timeline flags.

    Args:
        ticker_symbol (str): Stock ticker symbol
        company_name (str): Company name
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        str: News summary with flags
    """
    try:
        # Fetch articles for the period
        articles = get_ticker_news_polygon(
            ticker_symbol,
            limit=50,
            published_from=start_date,
            published_to=end_date
        )

        if not articles:
            return f"No news articles found for {company_name} ({ticker_symbol}) from {start_date} to {end_date}.\n\nFLAGS: []"

        # Filter articles for relevance
        filtered_articles = filter_articles(articles, ticker_symbol, company_name, scrape_cache)

        if not filtered_articles:
            return f"No relevant news articles found for {company_name} ({ticker_symbol}) from {start_date} to {end_date}.\n\nFLAGS: []"

        print(f"Filtered Articles: {filtered_articles}")

        # Create summary with flags - FIXED: Now passing all required arguments
        summary_with_flags = create_news_summary_for_period_with_flags(
            ticker_symbol, company_name, start_date, end_date, filtered_articles
        )

        return summary_with_flags

    except Exception as e:
        logger.error(f"Error fetching flagged news for {ticker_symbol}: {e}")
        return f"Error processing news for {company_name} ({ticker_symbol}): {str(e)}\n\nFLAGS: []"


def create_news_summary_for_period_with_flags(ticker_symbol, company_name, start_date, end_date, articles=None):
    """
    Create a news summary with timeline flags for a given period.

    Args:
        ticker_symbol (str): Stock ticker symbol
        company_name (str): Company name
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        articles (list, optional): Pre-filtered articles to use

    Returns:
        str: News summary with flags
    """
    try:
        # If no articles provided, fetch them
        if not articles:
            articles = get_ticker_news_polygon(
                ticker_symbol,
                limit=50,
                published_from=start_date,
                published_to=end_date
            )

            if articles:
                articles = filter_articles(articles, ticker_symbol, company_name, scrape_cache)

        if not articles:
            return f"No news available\n\nFLAGS: []"

        # Extract article data with content
        article_summaries = []
        for article in articles:
            try:
                # Get basic article info
                title = getattr(article, "title", "No title")
                description = getattr(article, "description", "")
                url = getattr(article, "article_url", "")
                published = getattr(article, "published_utc", "")

                # Try to get article content
                article_content = ""
                if url:
                    try:
                        article_content = get_article_text(url)
                        if article_content and len(article_content) > 100:
                            # Truncate very long articles to manage token limits
                            article_content = article_content[:2000] + "..." if len(
                                article_content) > 2000 else article_content
                    except Exception as e:
                        print(f"Could not scrape content from {url}: {e}")
                        article_content = description  # Fallback to description

                # Create article summary
                article_info = f"""
Article: {title}
Published: {published}
Description: {description}
Content: {article_content if article_content else "Content not available"}
URL: {url}
---"""

                article_summaries.append(article_info)

            except Exception as e:
                print(f"Error processing article: {e}")
                continue

        if not article_summaries:
            return f"Could not extract content from articles for {company_name} ({ticker_symbol}).\n\nFLAGS: []"

        # Combine all articles
        combined_articles = "\n".join(article_summaries)

        # Create the prompt for AI summarization with flags
        prompt = f"""
Please analyze the following news articles about {company_name} ({ticker_symbol}) from {start_date} to {end_date} and create a comprehensive summary.

Focus on:
1. Key developments and announcements
2. Financial performance or guidance  
3. Product launches or updates
4. Leadership changes
5. Regulatory or legal matters
6. Market reactions and analyst opinions
7. Strategic partnerships or acquisitions

Then identify which timeline flags apply from these options:
- EARNINGS: Earnings reports, financial results, guidance updates
- M&A: Mergers, acquisitions, partnerships, deals
- PRODUCT: New products, services, launches, updates
- LEADERSHIP: CEO changes, executive appointments, leadership news
- REGULATORY: Government actions, regulatory changes, compliance
- LEGAL: Lawsuits, legal settlements, court decisions
- ANALYST: Analyst upgrades/downgrades, price target changes
- CRISIS: Major negative events, scandals, crises
- MARKET: General market reactions, stock movements

Articles to analyze:
{combined_articles}

Please provide your response in this exact format:
[Your comprehensive summary here]

FLAGS: [FLAG1, FLAG2, FLAG3] (or FLAGS: [] if none apply)
"""

        # Check token count and truncate if necessary
        token_count = num_tokens(prompt)
        if token_count > 15000:  # Leave room for response
            print(f"Prompt too long ({token_count} tokens), truncating articles...")
            # Take first half of articles
            truncated_articles = article_summaries[:len(article_summaries) // 2]
            combined_articles = "\n".join(truncated_articles)
            prompt = f"""
Please analyze the following news articles about {company_name} ({ticker_symbol}) from {start_date} to {end_date} and create a comprehensive summary.

Focus on key developments, financial performance, product updates, leadership changes, regulatory matters, market reactions, and strategic partnerships.

Then identify applicable timeline flags: EARNINGS, M&A, PRODUCT, LEADERSHIP, REGULATORY, LEGAL, ANALYST, CRISIS, MARKET

Articles to analyze (truncated due to length):
{combined_articles}

Format: [Summary]

FLAGS: [FLAG1, FLAG2] or FLAGS: []
"""

        # Get AI summary with flags
        response = OAI.client.chat.completions.create(
            model=OAI.gpt4o,
            messages=[
                {"role": "system",
                 "content": "You are a financial news analyst. Provide clear, detailed. factual summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        summary_with_flags = response.choices[0].message.content.strip()
        print(f"Generated summary with flags: {summary_with_flags[:200]}...")
        return summary_with_flags

    except Exception as e:
        print(f"Error generating AI summary with flags: {e}")
        # Fallback summary
        if articles:
            titles = [getattr(article, "title", "No title") for article in articles[:3]]
            return f"Found {len(articles)} articles about {company_name} including: {'; '.join(titles)}...\n\nFLAGS: []"
        else:
            return f"No news available\n\nFLAGS: []"


def create_timeline_news_summary_for_period(extracted_data, company_name, max_tokens=512):
    """
    Creates news summary with timeline-optimized flags for chart pins.
    Focus on high-impact, actionable events only.
    """
    if not extracted_data:
        return "No relevant news found.\nFLAGS: []"

    # Use your existing summary creation logic
    base_summary = create_news_summary_for_period(extracted_data, company_name, max_tokens - 100)

    # Timeline-focused flagging prompt
    flagging_prompt = f"""
  Analyze this news summary for {company_name} and add ONLY significant flags that would be relevant for a stock price timeline chart.

  News Summary: {base_summary}

  Add flags in this exact format at the end:
  FLAGS: [FLAG1] [FLAG2]

  Available flags (use MAXIMUM 2, only if highly relevant):
  - EARNINGS: Earnings releases, guidance changes
  - M&A: Mergers, acquisitions, major partnerships  
  - REGULATORY: FDA approvals, government decisions, major regulatory news
  - LEADERSHIP: CEO changes, major executive appointments/departures
  - PRODUCT: Major product launches, breakthrough announcements
  - LEGAL: Significant lawsuits, settlements, investigations
  - ANALYST: Major analyst actions (upgrades/downgrades with significant price targets)
  - SPLIT: Stock splits, dividends, buybacks
  - CRISIS: Major negative events, recalls, scandals

  ONLY use flags for events that typically move stock prices significantly.
  Skip routine news, minor updates, or general market commentary.
  If no significant events, use: FLAGS: []
  """

    try:
        response = OAI.client.chat.completions.create(
            model=OAI.gpt4o,
            messages=[{"role": "user", "content": flagging_prompt}],
            max_tokens=120,
            temperature=0.1
        )

        flagged_summary = response.choices[0].message.content.strip()
        return flagged_summary

    except Exception as e:
        logger.error(f"Error adding timeline flags: {e}")
        return f"{base_summary}\nFLAGS: []"


def parse_news_flags(summary_with_flags):
    """
    Extract flags from a flagged summary for database storage.
    Returns: (clean_summary, flags_list)
    """
    import re

    # Extract flags using regex
    flag_match = re.search(r'FLAGS:\s*\[(.*?)\]', summary_with_flags, re.DOTALL)
    if flag_match:
        flags_text = flag_match.group(1)
        # Split by comma and clean up
        if flags_text.strip():
            flags = [flag.strip() for flag in flags_text.split(',') if flag.strip()]
        else:
            flags = []
        # Remove FLAGS section from summary
        summary_without_flags = re.sub(r'\nFLAGS:.*$', '', summary_with_flags, flags=re.MULTILINE | re.DOTALL)
        return summary_without_flags.strip(), flags

    return summary_with_flags, []


class TimelinePinManager:
    def __init__(self):
        # Priority scoring for timeline pins
        self.pin_priorities = {
            'EARNINGS': 90,
            'M&A': 95,
            'REGULATORY': 85,
            'LEADERSHIP': 80,
            'CRISIS': 100,
            'SPLIT': 75,
            'PRODUCT': 70,
            'LEGAL': 65,
            'ANALYST': 60
        }

    def calculate_priority_score(self, flags):
        """Calculate priority score based on flags."""
        if not flags:
            return 0
        return max([self.pin_priorities.get(flag, 0) for flag in flags])

    def should_create_timeline_pin(self, flags, priority_score):
        """Determine if news should appear as timeline pin."""
        if not flags:
            return False

        # Pin high-impact events
        high_impact_flags = ['EARNINGS', 'M&A', 'REGULATORY', 'LEADERSHIP', 'CRISIS']
        if any(flag in flags for flag in high_impact_flags):
            return True

        # Pin other events with decent priority
        return priority_score >= 65


def create_weekly_summary_for_period_with_flags(extracted_data, company_name, max_tokens=2048):
    """
    Enhanced weekly summary with flags in a single API call.
    """
    if not extracted_data:
        return "No relevant news found for this week.\nFLAGS: []"

    # Prepare weekly articles text
    articles_text = ""
    for i, article in enumerate(extracted_data[:20], 1):  # More articles for weekly
        title = article.get('title', 'No title')
        content = article.get('content', article.get('description', 'No content'))[:400]
        published = article.get('published_utc', 'Unknown date')
        articles_text += f"\n{i}. [{published}] {title}\n{content}\n"

    # Weekly-focused prompt
    combined_prompt = f"""
  Analyze the following week's news articles about {company_name} and create a comprehensive weekly summary with timeline flags.

  Weekly Articles:
  {articles_text}

  Create a weekly summary that:
  1. Captures the week's major themes and developments
  2. Highlights significant market-moving events
  3. Shows progression of stories throughout the week
  4. Identifies key events suitable for timeline visualization

  Format your response as:
  [Your weekly news summary here]

  FLAGS: [FLAG1, FLAG2]

  Available flags (use MAXIMUM 2 for the most significant events):
  - EARNINGS: Earnings releases, guidance changes
  - M&A: Mergers, acquisitions, major partnerships  
  - REGULATORY: FDA approvals, government decisions, major regulatory news
  - LEADERSHIP: CEO changes, major executive appointments/departures
  - PRODUCT: Major product launches, breakthrough announcements
  - LEGAL: Significant lawsuits, settlements, investigations
  - ANALYST: Major analyst actions (upgrades/downgrades with significant price targets)
  - SPLIT: Stock splits, dividends, buybacks
  - CRISIS: Major negative events, recalls, scandals

  Focus on events that would be important enough to pin on a stock price timeline chart.
  If no major events occurred this week respond with a general summary of the contents of the summaries then use: FLAGS: []
  """

    try:
        response = OAI.client.chat.completions.create(
            model=OAI.gpt4o,
            messages=[{"role": "user", "content": combined_prompt}],
            max_tokens=max_tokens,
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Error creating weekly flagged summary for {company_name}: {e}")
        return f"Error processing weekly news for {company_name}\nFLAGS: []"


# Enhanced weekly processing function
def fetch_and_summarize_weekly_articles_cached_with_flags(
        ticker_symbol, company_name, week_start, week_end, daily_summaries
):
    """
    Enhanced weekly summary that includes flags, optimized for single API call.
    """
    # First, try to use daily summaries approach (your existing logic)
    daily_texts = []
    current_day = week_start
    while current_day <= week_end:
        day_str = current_day.strftime('%Y-%m-%d')
        summary = daily_summaries.get(day_str, None)
        if (
                summary
                and summary.strip()
                and summary.strip().lower() not in (
                "false",
                "no information relevant information found.",
                "no relevant info found (by bot)",
                "timeout occurred"
        )
        ):
            daily_texts.append(summary)
        current_day += timedelta(days=1)

    if daily_texts:
        # Use daily summaries to create weekly summary with flags
        combined_daily_text = "\n\n".join(daily_texts)

        weekly_prompt = f"""
      Based on these daily news summaries for {company_name} during the week of {week_start} to {week_end}, 
      create a comprehensive weekly summary with timeline flags.

      Daily Summaries:
      {combined_daily_text}

      Create a weekly summary that synthesizes the key themes and adds appropriate timeline flags.

      Format your response as:
      [Your weekly summary here]

      FLAGS: [FLAG1, FLAG2]

      Available flags (use MAXIMUM 2 for the most significant weekly events):
      - EARNINGS: Earnings releases, guidance changes
      - M&A: Mergers, acquisitions, major partnerships  
      - REGULATORY: FDA approvals, government decisions, major regulatory news
      - LEADERSHIP: CEO changes, major executive appointments/departures
      - PRODUCT: Major product launches, breakthrough announcements
      - LEGAL: Significant lawsuits, settlements, investigations
      - ANALYST: Major analyst actions (upgrades/downgrades with significant price targets)
      - SPLIT: Stock splits, dividends, buybacks
      - CRISIS: Major negative events, recalls, scandals

      If no major events occurred this week respond with a general summary of the contents of the articles, then at the end use: FLAGS: []
      """

        try:
            response = OAI.client.chat.completions.create(
                model=OAI.gpt4o,
                messages=[{"role": "user", "content": weekly_prompt}],
                max_tokens=512,
                temperature=0.2
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error creating weekly summary from daily summaries: {e}")
            return f"Weekly summary unavailable for {company_name}\nFLAGS: []"

    else:
        # Fallback: fetch weekly articles directly
        try:
            week_start_str = week_start.strftime('%Y-%m-%d')
            week_end_str = week_end.strftime('%Y-%m-%d')

            articles = get_ticker_news_polygon(ticker_symbol, limit=20,
                                               published_from=week_start_str,
                                               published_to=week_end_str)
            relevant_articles = filter_articles(articles, ticker_symbol, company_name, scrape_cache=scrape_cache)
            extracted_data = extract_article_data(relevant_articles)

            # Use the enhanced weekly summary function
            return create_weekly_summary_for_period_with_flags(extracted_data, company_name, 2048)

        except Exception as e:
            logger.error(f"Error fetching weekly articles for {ticker_symbol}: {e}")
            return f"No weekly news available for {company_name}\nFLAGS: []"




# Run the function
if __name__ == "__main__":
    # Example: Fetch news for Apple (AAPL) with a date filter
    ticker_symbol = "AAPL"
    from_date = "2025-04-20"  # Articles published on or after this date
    to_date = "2025-05-21"    # Articles published on or before this date

    # Fetch news articles using your existing function
    #articles = get_ticker_news(ticker_symbol, limit=10, published_from=from_date, published_to=to_date)

    #relevant_articles = filter_articles(articles, ticker_symbol, 'Apple')

    # Extract structured data from the articles
    #extracted_data = extract_article_data(relevant_articles)
    #print("Extracted article metadata:")
    #for article in extracted_data:
        #print(article)
        #print("-" * 40)

    # Generate a summary for all articles in the period
    #print("\n=== News Summary for Period ===")
    #summary = create_news_summary_for_period(extracted_data, 256)
    #print(summary)
