"""Engine-side request-id middleware — Task 5.2.

Mirror of ``server-fastapi-wt/app/middleware/request_id.py``.  Keeping
the two implementations symmetric (same ContextVar names, same header
constant, same UUID4 generator, same 128-char ceiling) means an
operator can reason about correlation IDs across both services with
one mental model.

When the server hops to engine, ``app/engine_client.py`` sets the
``X-Request-Id`` header.  This middleware reads it and stashes the same
value into ``request_id_var`` so any engine event row or log line emitted
during the request joins to the server-side chain by the same id.
"""
from __future__ import annotations

import contextvars
import uuid
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
firm_id_var: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "firm_id", default=None
)


HEADER_NAME = "X-Request-Id"


def _new_request_id() -> str:
    return uuid.uuid4().hex


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Engine-side counterpart of the server's RequestIdMiddleware.

    See the server module's docstring for the rationale.  The behaviour
    is byte-identical so engine ↔ server hops are symmetric.
    """

    _MAX_INBOUND_ID_LEN = 128

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        inbound = request.headers.get(HEADER_NAME)
        if inbound and len(inbound) <= self._MAX_INBOUND_ID_LEN and inbound.strip():
            request_id = inbound.strip()
        else:
            request_id = _new_request_id()

        request.state.request_id = request_id
        rid_token = request_id_var.set(request_id)
        firm_token = firm_id_var.set(None)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(rid_token)
            firm_id_var.reset(firm_token)
        response.headers[HEADER_NAME] = request_id
        return response


def get_request_id() -> str | None:
    return request_id_var.get()


def set_firm_id(firm_id: int | None) -> contextvars.Token[int | None]:
    return firm_id_var.set(firm_id)


def get_firm_id() -> int | None:
    return firm_id_var.get()
