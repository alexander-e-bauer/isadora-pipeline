"""Engine-side auth helpers.

is_demo_session reads the X-Demo-Session header forwarded by the server
when the originating firm is a demo. Subagents branch on this to downgrade
the model and apply cluster-wide rate limits.
"""
from __future__ import annotations

from fastapi import Request


def is_demo_session(request: Request) -> bool:
    """Return True iff the request has the X-Demo-Session header set to 'true'.

    Performs a case-insensitive scan of both the header name and value so the
    dep tolerates server-side casing variations (`X-Demo-Session: True`,
    `x-demo-session: true`, etc.) regardless of whether the ASGI layer has
    already lowercased the raw header bytes.
    """
    target = b"x-demo-session"
    for name, value in request.headers.raw:
        if name.lower() == target:
            return value.decode("latin-1").strip().lower() == "true"
    return False
