import os
import json
from cachetools import TTLCache
import numpy as np
import pandas as pd
import redis

from config import logger


def get_redis_client():
    try:
        redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
        client = redis.from_url(redis_url)
        client.ping()  # Test connection
        return client
    except Exception as e:
        logger.warning(f"Redis connection failed: {str(e)}")
        return None

# Initialize Redis client
redis_client = get_redis_client()
USE_REDIS = redis_client is not None
if not USE_REDIS:
    logger.warning("Using in-memory cache instead of Redis")
    in_memory_cache = TTLCache(maxsize=100, ttl=24*60*60)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        return json.JSONEncoder.default(self, obj)

def get_cache_key(ticker, start_date, end_date):
    """Generate a unique cache key for the prediction."""
    return f"market_prediction:{ticker}:{start_date}:{end_date}"


def get_cached_prediction(ticker, start_date, end_date):
    """Retrieve cached prediction if it exists."""
    cache_key = get_cache_key(ticker, start_date, end_date)

    if USE_REDIS:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
    else:
        if cache_key in in_memory_cache:
            return in_memory_cache[cache_key]

    return None


def save_prediction_to_cache(ticker, start_date, end_date, prediction_data):
    """Save prediction data to cache."""
    cache_key = get_cache_key(ticker, start_date, end_date)

    if USE_REDIS:
        redis_client.setex(
            cache_key,
            24 * 60 * 60,  # 24 hours in seconds
            json.dumps(prediction_data, cls=NumpyEncoder)
        )
    else:
        in_memory_cache[cache_key] = prediction_data