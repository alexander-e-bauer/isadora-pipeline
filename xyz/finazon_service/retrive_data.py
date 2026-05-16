import time
from datetime import datetime
from threading import Lock
from finazon_grpc_python.time_series_service import TimeSeriesService, GetTimeSeriesRequest
from finazon_grpc_python.common.errors import FinazonGrpcRequestError
from finazon_grpc_python.common.utils import convert_response_to_pandas
from config import FINAZON_KEY
import pandas as pd

# Initialize the global service
service = TimeSeriesService(FINAZON_KEY)


def retry_with_backoff(func, retries=3, backoff_factor=2):
    """
    Retry a function with exponential backoff
    """
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt < retries - 1:
                wait_time = backoff_factor ** attempt
                time.sleep(wait_time)
            else:
                raise e


class FinazonService:
    def __init__(self, rate_limit_per_minute=5):
        self.rate_limit_per_minute = rate_limit_per_minute
        self.request_interval = 60 / rate_limit_per_minute  # Time in seconds between requests
        self.lock = Lock()  # Initialize a threading lock

    def reconnect_service(self):
        """
        Reinitialize the TimeSeriesService with thread safety
        """
        global service
        with self.lock:  # Use self.lock instead of retriever.lock
            service = TimeSeriesService(FINAZON_KEY)

    @staticmethod
    def format_start_time(start_time):
        """
        Format the start_time parameter to match the expected format (Unix timestamp).
        """
        if isinstance(start_time, int):
            # Assume it's already a Unix timestamp
            return start_time
        elif isinstance(start_time, datetime):
            # Convert datetime to Unix timestamp
            return int(start_time.timestamp())
        elif isinstance(start_time, str):
            # Try to parse the string into a datetime object and convert to Unix timestamp
            try:
                parsed_time = datetime.fromisoformat(start_time)
                return int(parsed_time.timestamp())
            except ValueError:
                raise ValueError("Invalid string format for start_time. Expected ISO 8601 format.")
        else:
            raise ValueError("Invalid start_time format. Must be int (Unix), str (ISO 8601), or datetime.")

    def fetch_time_series(self, ticker, interval='30m', dataset='us_stocks_essential', existing_df=None, start_time=1740787200):
        """
        Fetch time series data for a ticker with pagination and rate limiting
        """
        page = 1
        all_data = []

        while True:
            try:
                # Create the request with start_at if available
                request = GetTimeSeriesRequest(
                    ticker=ticker,
                    dataset=dataset,
                    interval=interval,
                    page_size=100,
                    page=page,
                    start_at=start_time,  # Fetch only new data
                    order="asc"  # Fetch in chronological order
                )

                # Send the request
                response = retry_with_backoff(lambda: service.get_time_series(request))

                # Convert the processed response to a pandas DataFrame
                df = convert_response_to_pandas(response)

                # Debug: Print the DataFrame
                #print(f"DataFrame for page {page}:\n {df.head(n=3)}\n")

                if df.empty:
                    print(f"No more data to retrieve. Total pages retrieved: {page-1}")
                    break

                # Append the data to the list
                all_data.append(df)

                # Increment the page number for the next request
                page += 1
                #print(f'Turning to page {page}\n')

                # Respect the rate limit
                time.sleep(self.request_interval)

            except FinazonGrpcRequestError as e:
                print(f"Received error, code: {e.code}, message: {e.message}")
                break
            except Exception as e:
                # Handle closed channel error and reconnect
                if "closed channel" in str(e).lower():
                    self.reconnect_service()  # Call the instance method instead of the undefined function
                    continue  # Retry the request after reconnecting
                else:
                    raise e

        # Combine all pages into a single DataFrame
        new_data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

        # Merge new data with existing data
        if existing_df is not None:
            # Append new rows and drop duplicates
            combined_df = pd.concat([existing_df, new_data]).drop_duplicates(subset='timestamp').reset_index(
                drop=True)
        else:
            combined_df = new_data

        return combined_df


if __name__ == "__main__":
    retriever = FinazonService(rate_limit_per_minute=5)
    df = retriever.fetch_time_series(ticker='GOOG', interval='15m')
    print(f"Final Result:\n\n {df}")
