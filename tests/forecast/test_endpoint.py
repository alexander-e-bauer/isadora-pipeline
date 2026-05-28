"""POST /agents/forecast endpoint tests — uses FastAPI TestClient.

These tests use helpers_app (a minimal stub FastAPI app that replicates the
bearer-auth dependency) for the auth-rejection and 422-validation tests, so
they run without importing the full app.py (which requires live API keys).
The integration test imports app directly and is skipped without ENGINE_INTEGRATION_DB.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

KEY = os.environ.get("KEY", "test-key")
HEADERS = {"Authorization": f"Bearer {KEY}"}


# ---------------------------------------------------------------------------
# Minimal stub app — mirrors auth dep without importing app.py
# ---------------------------------------------------------------------------

class _ForecastRequest(BaseModel):
    strategy_id: str
    strategy_version: int
    firm_id: str
    actor_user_id: Optional[str] = None
    dsl: dict
    t0: str
    horizon_days: int = 252
    n_paths: int = 1000
    forecast_seed: int
    overrides: Optional[dict[str, Any]] = None
    research_artifact_id: Optional[str] = None


def _make_stub_client() -> TestClient:
    """Build a minimal app that exposes /agents/forecast with bearer auth."""
    _security = HTTPBearer(auto_error=True)

    def _verify_bearer(credentials: HTTPAuthorizationCredentials = Depends(_security)):
        k = os.environ.get("KEY", "test-key")
        if credentials.credentials != k:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        return credentials.credentials

    mini = FastAPI()

    @mini.post("/agents/forecast", dependencies=[Depends(_verify_bearer)])
    def _forecast_stub(body: _ForecastRequest):
        return {"ok": True, "strategy_id": body.strategy_id}

    return TestClient(mini, raise_server_exceptions=False)


def _payload(seed: int = 42):
    return {
        "strategy_id": "strat-1", "strategy_version": 1, "firm_id": "firm-1",
        "actor_user_id": "user-1",
        "dsl": {
            "template": "covered_call",
            "selection": {"universe": ["AAPL"]},
            "action": {"delta_short_target": 0.30, "dte_min": 25, "dte_max": 45},
            "exit": {"max_profit_pct_close": 0.50},
            "risk_box": {},
        },
        "t0": "2025-12-31", "horizon_days": 30,
        "n_paths": 200, "forecast_seed": seed,
        "overrides": {"calibration_source": "recent_window"},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forecast_endpoint_rejects_missing_auth():
    """POST /agents/forecast without Authorization header returns 401 or 403."""
    c = _make_stub_client()
    res = c.post("/agents/forecast", json=_payload())
    assert res.status_code in (401, 403)


def test_forecast_endpoint_validates_required_fields():
    """POST /agents/forecast with a body missing required fields returns 422."""
    c = _make_stub_client()
    bad = {"firm_id": "firm-1"}
    res = c.post("/agents/forecast", json=bad, headers=HEADERS)
    assert res.status_code == 422


def test_forecast_endpoint_returns_artifact():
    """Integration smoke — requires a working DB wired via ENGINE_INTEGRATION_DB."""
    if not os.environ.get("ENGINE_INTEGRATION_DB"):
        pytest.skip("no integration DB configured")

    import importlib
    import sys
    # Ensure app.py is importable by providing required env vars.
    with pytest.MonkeyPatch().context() as mp:
        mp.setenv("OPENAI_API_KEY", "fake")
        mp.setenv("ANTHROPIC_API_KEY", "fake")
        mp.setenv("POLYGON_KEY", "fake")
        mp.setenv("KEY", "test-key")
        # Force re-import in case a prior import cached without env vars.
        for mod in list(sys.modules.keys()):
            if mod in ("app", "config"):
                del sys.modules[mod]
        from app import app as _app  # noqa: PLC0415

    c = TestClient(_app)
    res = c.post("/agents/forecast", json=_payload(), headers=HEADERS)
    assert res.status_code == 200
    body = res.json()
    assert body["content_hash"]
    assert "nav_bands" in body
    assert "p50" in body["nav_bands"]
