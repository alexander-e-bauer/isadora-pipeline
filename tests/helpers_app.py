"""Test helper: build a minimal FastAPI TestClient for bearer-auth route tests.

This module creates an isolated FastAPI app that mirrors the bearer-auth
dependency from app.py but only mounts the routes needed for auth tests,
so the bearer-auth tests run without importing the full app.py (which
would pull in Pinecone, psycopg2, OpenAI, etc.).

Routes mounted:
  - POST /agents/research          (Task 4.1)
  - POST /agents/author            (Task 4.2)
  - POST /agents/validate-dsl      (Task 4.2)
"""
from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field


# Request schemas are declared at module scope (not inside get_test_client)
# because FastAPI introspects type annotations at route-registration time,
# and locally-defined classes inside a closure occasionally confuse the
# pydantic forward-ref resolver — leading to FastAPI treating the body as
# a query parameter and emitting 422 "Field required" errors.
class _ResearchRequest(BaseModel):
    firm_id: int
    account_id: int | None = None
    symbol: str | None = None
    brief: str | None = None
    actor_user_id: int | None = None


class _AuthorRequest(BaseModel):
    firm_id: int
    brief: str = Field(..., min_length=1)
    actor_user_id: int | None = None
    target_account_ids: list[int] = Field(default_factory=list)


class _ValidateDslRequest(BaseModel):
    dsl: dict[str, Any]


def get_test_client() -> TestClient:
    """Build a minimal FastAPI app and return its TestClient.

    The KEY env var must be set before calling this; it is read at call time
    (not at import time) so tests can monkeypatch it via patch.dict.
    """
    _security = HTTPBearer(auto_error=True)

    def _verify_bearer(credentials: HTTPAuthorizationCredentials = Depends(_security)):
        key = os.environ.get("KEY", "")
        if credentials.credentials != key:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        return credentials.credentials

    mini_app = FastAPI()

    @mini_app.post("/agents/research", dependencies=[Depends(_verify_bearer)])
    def _research_stub(body: _ResearchRequest):
        return {"ok": True, "firm_id": body.firm_id}

    @mini_app.post("/agents/author", dependencies=[Depends(_verify_bearer)])
    def _author_stub(body: _AuthorRequest):
        return {"ok": True, "firm_id": body.firm_id}

    @mini_app.post("/agents/validate-dsl", dependencies=[Depends(_verify_bearer)])
    def _validate_stub(body: _ValidateDslRequest):
        # Echo the validator's real result so route-only tests (no DB) can
        # still exercise valid/invalid response shapes.
        from xyz.dsl.validate import validate_dsl
        valid, errors = validate_dsl(body.dsl)
        return {"valid": valid, "errors": errors}

    return TestClient(mini_app, raise_server_exceptions=False)
