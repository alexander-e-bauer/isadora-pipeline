"""Cluster-wide demo agent rate limiter.

A single Redis sorted-set key shared across all engine instances bounds
the total demo-flavored agent calls per minute. Defends against a
misconfigured server forwarding more demo traffic than expected, or
against a sudden burst that overwhelms the haiku quota.

Fail-closed on Redis outage: the caller (each /agents/* route) sees a
redis.exceptions.ConnectionError bubble up and translates to 503.
"""
from __future__ import annotations

import os
import secrets
import time

import redis.asyncio as redis_async
from fastapi import HTTPException


_KEY = "demo:engine:rate"
_WINDOW_SECONDS = 60

_redis_client: redis_async.Redis | None = None


def get_redis() -> redis_async.Redis:
    """Lazy singleton async Redis client (URL from REDIS_URL env var)."""
    global _redis_client
    if _redis_client is None:
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        _redis_client = redis_async.from_url(url, decode_responses=True)
    return _redis_client


async def enforce_demo_agent_rate_limit() -> None:
    """Sliding-window cluster-wide rate limit on demo agent calls.

    Reads DEMO_AGENT_RATE_LIMIT_PER_MINUTE at call time (allows config-flip
    without restart). Default limit: 10/minute across the cluster.

    Same member-uniqueness pattern as the server's IP limiter: member =
    f"{now}:{secrets.token_hex(8)}" so multiple calls in the same wall-
    clock second don't collapse into a single ZADD entry.
    """
    limit = int(os.environ.get("DEMO_AGENT_RATE_LIMIT_PER_MINUTE", "10"))
    now = int(time.time())
    cutoff = now - _WINDOW_SECONDS
    member = f"{now}:{secrets.token_hex(8)}"

    client = get_redis()
    pipe = client.pipeline()
    pipe.zremrangebyscore(_KEY, 0, cutoff)
    pipe.zcard(_KEY)
    pipe.zadd(_KEY, {member: now})
    pipe.expire(_KEY, _WINDOW_SECONDS)
    _, count, _, _ = await pipe.execute()
    if count >= limit:
        raise HTTPException(status_code=429, detail="demo_engine_rate_limited")
