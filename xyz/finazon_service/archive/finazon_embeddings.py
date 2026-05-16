

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import requests
import re
import json
import numpy as np
import pandas as pd
import tiktoken
from xyz.finazon_service.metrics import compute_batch_metrics
from config import DATABASE, OAI, PINECONE_API_KEY, PINECONE_HOST
from psycopg2.extras import Json, RealDictCursor
import logging
from pinecone import Pinecone

# Create Pinecone Index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_HOST)

logger = logging.getLogger(__name__)





def num_tokens(text):
    try:
        # Get the encoding for the model
        encoding = tiktoken.encoding_for_model('gpt-4o')
        # Encode the text to count tokens
        encoded_text = encoding.encode(text, disallowed_special=())
        token_count = len(encoded_text)
        print(f"Token count: {token_count}")
        return token_count
    except Exception as e:
        print(f"Error in num_tokens: {e}")
        return None


def remove_stuff(text: str) -> str:
    """Remove punctuation (except in URLs), newline, tab characters, and large spaces."""
    # Pattern to identify URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Find all URLs using the pattern
    urls = re.findall(url_pattern, text)
    # Replace URLs with a placeholder to avoid altering them
    placeholder = "URL_PLACEHOLDER"
    for url in urls:
        text = text.replace(url, placeholder)

    # Remove large spaces (5 or more spaces)
    text = re.sub(r' {5,}', ' ', text)

    # Restore URLs from placeholders
    for url in urls:
        text = text.replace(placeholder, url, 1)

    return text


def get_embedding(text_to_embed):
    """
    Generates an embedding for the given text using OpenAI's API.
    """
    text_to_embed = remove_stuff(text_to_embed)
    # Check the number of tokens
    token_count = num_tokens(text_to_embed)
    max_token_limit = 8192  # Adjust based on your model's token limit

    if token_count is None or token_count > max_token_limit:
        print(f"Text exceeds the token limit ({max_token_limit} tokens). Skipping embedding.")
        return None

    try:
        # Embed a line of text
        response = OAI.client.embeddings.create(
            model=OAI.embedding3,
            input=[text_to_embed]
        )
        # Extract the AI output embedding as a list of floats
        embedding = response.data[0].embedding
        print(f"---\nEmbedding generated successfully for text: {text_to_embed[:100]}...")
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def save_to_pinecone(data_id, embedding, metadata, namespace="default"):
    """
    Saves financial time series data and its embedding to Pinecone.

    Parameters:

    """
    try:

        # Format the data for Pinecone
        record = {
            "id": str(data_id),
            "values": embedding,
            "metadata": metadata
        }

        # Upsert the record into Pinecone
        index.upsert(vectors=[record], namespace=namespace)

    except Exception as e:
        print(f"Error: {e}")


def process_and_store_embeddings(df, ticker_symbol, batch_size=100):
    """
    Process time series data in batches and store embeddings in Pinecone.

    Parameters:
    - df: DataFrame containing time series data
    - ticker_symbol: Stock ticker symbol
    - batch_size: Number of rows to process in each batch
    """
    if df is None or df.empty:
        logger.info(f"No data to process for {ticker_symbol}")
        return

    # Sort by timestamp to ensure chronological order
    df = df.sort_values('timestamp').copy()

    # Compute metrics for the entire dataset to capture trends properly
    logger.info(f"Computing metrics for {len(df)} rows of {ticker_symbol} data")
    metrics_df = compute_batch_metrics(df)

    # Process in batches to manage memory usage
    total_rows = len(df)
    batch_count = (total_rows + batch_size - 1) // batch_size  # Ceiling division

    logger.info(f"Processing {total_rows} rows in {batch_count} batches of size {batch_size}")

    vectors_to_upsert = []

    for i in range(0, total_rows, batch_size):
        batch_end = min(i + batch_size, total_rows)
        batch_df = df.iloc[i:batch_end]
        batch_metrics = metrics_df.iloc[i:batch_end]

        logger.info(f"Processing batch {i // batch_size + 1}/{batch_count} for {ticker_symbol} ({batch_end - i} rows)")

        for idx, (_, row) in enumerate(batch_df.iterrows()):
            metrics_row = batch_metrics.iloc[idx]

            # Create a summary of the data point
            timestamp = int(row['timestamp'])
            summary_of_data = (f"Time series data for {ticker_symbol} at timestamp {timestamp}: "
                               f"Open: {row['open']:.2f}, Close: {row['close']:.2f}, "
                               f"High: {row['high']:.2f}, Low: {row['low']:.2f}, Volume: {row['volume']}")

            # Create metadata for Pinecone
            metadata = {
                "ticker": ticker_symbol,
                "timestamp": timestamp,
                "open": float(row["open"]),
                "close": float(row["close"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "volume": float(row["volume"]),
                "summary": summary_of_data,
                "type": "historical_data_with_metrics"
            }

            # Create hour, day, and month embeddings (simple one-hot encoding)
            hour_embedding = [0] * 24
            hour_embedding[int(metrics_row['hour_of_day'])] = 1

            day_embedding = [0] * 7
            day_embedding[int(metrics_row['day_of_week'])] = 1

            month_embedding = [0] * 12
            month_embedding[int(metrics_row['month_of_year']) - 1] = 1

            # Create the comprehensive embedding dictionary
            embedding_dict = {
                # Raw Data
                "metadata": metadata,
                "open": float(row["open"]),
                "close": float(row["close"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "volume": float(row["volume"]),

                # Returns
                "hourly_return": float(metrics_row["hourly_return"]),
                "log_return": float(metrics_row["log_return"]),
                "vwap": float(metrics_row["vwap"]),

                # Volatility
                "historical_volatility": float(metrics_row["historical_volatility"]),
                "realized_volatility": float(metrics_row["realized_volatility"]),

                # Technical Indicators
                "sma_5": float(metrics_row["sma_5"]),
                "ema_5": float(metrics_row["ema_5"]),
                "sma_20": float(metrics_row["sma_20"]),
                "ema_20": float(metrics_row["ema_20"]),
                "sma_50": float(metrics_row["sma_50"]),
                "ema_50": float(metrics_row["ema_50"]),
                "sma_200": float(metrics_row["sma_200"]),
                "ema_200": float(metrics_row["ema_200"]),
                "rsi": float(metrics_row["rsi"]),
                "macd": float(metrics_row["macd"]),
                "roc": float(metrics_row["roc"]),
                "bollinger_upper": float(metrics_row["bollinger_upper"]),
                "bollinger_lower": float(metrics_row["bollinger_lower"]),
                "bollinger_width": float(metrics_row["bollinger_width"]),

                # Volume Indicators
                "obv": float(metrics_row["obv"]),
                "cmf": float(metrics_row["cmf"]),

                # Anomaly Detection
                "z_score": float(metrics_row["z_score"]),
                "mahalanobis_distance": float(metrics_row["mahalanobis_distance"]),
                "cusum_score": float(metrics_row["cusum_score"]),
                "ewma_score": float(metrics_row["ewma_score"]),

                # RL and Risk Metrics
                "sharpe_ratio": float(metrics_row["sharpe_ratio"]),
                "sortino_ratio": float(metrics_row["sortino_ratio"]),
                "max_drawdown": float(metrics_row["max_drawdown"]),
                "var": float(metrics_row["var"]),
                "cvar": float(metrics_row["cvar"]),

                # Contextual Embeddings
                "cosine_similarity": float(metrics_row["cosine_similarity"]),
                "hour_of_day": hour_embedding,
                "day_of_week": day_embedding,
                "month_of_year": month_embedding,

                # Advanced (placeholders)
                "hurst_exponent": float(metrics_row["hurst_exponent"]),
                "approx_entropy": float(metrics_row["approx_entropy"]),
                "regime_probability": float(metrics_row["regime_probability"]),
            }

            # Convert to string for embedding
            data_str = json.dumps(embedding_dict)

            # Generate embedding
            embedding = get_embedding(data_str)

            if embedding:
                # Add to batch for upserting
                vectors_to_upsert.append({
                    "id": f"{ticker_symbol}_{timestamp}",
                    "values": embedding,
                    "metadata": metadata
                })

        # Batch upsert to Pinecone when we have enough vectors or at the end of a batch
        if len(vectors_to_upsert) >= 100 or batch_end == total_rows:
            try:
                logger.info(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone")
                index.upsert(vectors=vectors_to_upsert, namespace="finance")
                vectors_to_upsert = []  # Clear the list after upserting
            except Exception as e:
                logger.error(f"Error upserting vectors to Pinecone: {e}")

    logger.info(f"Completed processing {total_rows} rows for {ticker_symbol}")
    return True

