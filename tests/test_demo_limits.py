"""Cluster-wide demo agent rate limiter. Bounds the total demo-flavored
agent calls per minute across all engine workers. Defends against a
misconfigured server forwarding more demo traffic than expected.

Tests use the local docker-compose Redis on localhost:6380 (mapped from the
server worktree's docker-compose).
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi import HTTPException


_LOCAL_REDIS_URL = "redis://localhost:6380/0"


@pytest.fixture(autouse=True)
def _force_local_redis(monkeypatch):
    monkeypatch.setenv("REDIS_URL", _LOCAL_REDIS_URL)
    # Reset module-level singleton
    import xyz.agents.demo_limits as limits
    limits._redis_client = None
    yield
    limits._redis_client = None


@pytest_asyncio.fixture(autouse=True)
async def _clean_redis(_force_local_redis):
    from xyz.agents.demo_limits import get_redis
    client = get_redis()
    await client.flushdb()
    yield
    try:
        await client.flushdb()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_cluster_wide_cap_blocks_at_threshold(monkeypatch):
    monkeypatch.setenv("DEMO_AGENT_RATE_LIMIT_PER_MINUTE", "3")
    from xyz.agents.demo_limits import enforce_demo_agent_rate_limit
    for _ in range(3):
        await enforce_demo_agent_rate_limit()
    with pytest.raises(HTTPException) as exc:
        await enforce_demo_agent_rate_limit()
    assert exc.value.status_code == 429


@pytest.mark.asyncio
async def test_same_second_burst_doesnt_collapse(monkeypatch):
    """Multiple calls in the same wall-clock second must NOT collapse to one
    sorted-set entry (the bug the server's rate limiter caught)."""
    monkeypatch.setenv("DEMO_AGENT_RATE_LIMIT_PER_MINUTE", "3")
    from xyz.agents.demo_limits import enforce_demo_agent_rate_limit
    # Burst 5 calls — first 3 should succeed, last 2 should 429
    outcomes: list[bool | int] = []
    for _ in range(5):
        try:
            await enforce_demo_agent_rate_limit()
            outcomes.append(False)
        except HTTPException as e:
            outcomes.append(e.status_code)
    assert outcomes == [False, False, False, 429, 429]


@pytest.mark.asyncio
async def test_window_resets_after_60_seconds(monkeypatch):
    monkeypatch.setenv("DEMO_AGENT_RATE_LIMIT_PER_MINUTE", "3")
    import xyz.agents.demo_limits as limits
    from xyz.agents.demo_limits import enforce_demo_agent_rate_limit

    fake_time = [1000]
    monkeypatch.setattr(limits.time, "time", lambda: fake_time[0])

    for _ in range(3):
        await enforce_demo_agent_rate_limit()

    # Advance just past the 60s window
    fake_time[0] = 1000 + 61
    # Should succeed — all prior entries are outside the window
    await enforce_demo_agent_rate_limit()


@pytest.mark.asyncio
async def test_default_limit_is_10(monkeypatch):
    """When DEMO_AGENT_RATE_LIMIT_PER_MINUTE is unset, default is 10."""
    monkeypatch.delenv("DEMO_AGENT_RATE_LIMIT_PER_MINUTE", raising=False)
    from xyz.agents.demo_limits import enforce_demo_agent_rate_limit
    for _ in range(10):
        await enforce_demo_agent_rate_limit()
    with pytest.raises(HTTPException) as exc:
        await enforce_demo_agent_rate_limit()
    assert exc.value.status_code == 429
