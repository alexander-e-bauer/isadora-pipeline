"""Engine-side observability helpers — Task 5.2.

Mirrors ``server-fastapi-wt/app/middleware/request_id.py`` and
``app/logging_config.py`` so a request ID set on the server hop flows
through to the engine and back, visible in Cloud Logging on both
processes (acceptance criterion 1).
"""
from xyz.observability.request_id import (
    HEADER_NAME,
    RequestIdMiddleware,
    firm_id_var,
    get_firm_id,
    get_request_id,
    request_id_var,
    set_firm_id,
)
from xyz.observability.logging_config import JsonFormatter, configure_logging

__all__ = [
    "HEADER_NAME",
    "JsonFormatter",
    "RequestIdMiddleware",
    "configure_logging",
    "firm_id_var",
    "get_firm_id",
    "get_request_id",
    "request_id_var",
    "set_firm_id",
]
