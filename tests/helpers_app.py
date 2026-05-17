"""Test helper: build a minimal FastAPI TestClient for bearer-auth route tests.

This module creates an isolated FastAPI app that only mounts the
/agents/research endpoint (with the same bearer-auth dependency as the
main app) so that bearer-auth tests can run without importing the full
app.py (which would pull in Pinecone, psycopg2, OpenAI, etc.).
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.testclient import TestClient
from pydantic import BaseModel


def get_test_client() -> TestClient:
    """Build a minimal FastAPI app with the /agents/research route and return its TestClient.

    The KEY env var must be set before calling this; it is read at call time
    (not at import time) so tests can monkeypatch it via patch.dict.
    """
    _security = HTTPBearer(auto_error=True)

    def _verify_bearer(credentials: HTTPAuthorizationCredentials = Depends(_security)):
        key = os.environ.get("KEY", "")
        if credentials.credentials != key:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        return credentials.credentials

    class _ResearchRequest(BaseModel):
        firm_id: int
        account_id: int | None = None
        symbol: str | None = None
        brief: str | None = None
        actor_user_id: int | None = None

    mini_app = FastAPI()

    @mini_app.post("/agents/research", dependencies=[Depends(_verify_bearer)])
    def _research_stub(body: _ResearchRequest):
        # Stub: just echo back the request so we can test auth without a real agent
        return {"ok": True, "firm_id": body.firm_id}

    return TestClient(mini_app, raise_server_exceptions=False)
