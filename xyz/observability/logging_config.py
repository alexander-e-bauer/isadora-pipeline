"""Engine-side structured JSON logging — Task 5.2.

Mirror of ``server-fastapi-wt/app/logging_config.py``.  Same schema, so
Cloud Logging queries written against ``jsonPayload.request_id`` etc.
match log lines from both processes.
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any

from xyz.observability.request_id import firm_id_var, request_id_var


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record.

    Documented schema:
      ts, level, logger, msg, request_id, firm_id, [trace].
    """

    _RESERVED_ATTRS = frozenset(
        {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "message", "module",
            "msecs", "msg", "name", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "thread", "threadName",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
            "firm_id": firm_id_var.get(),
        }

        if record.exc_info:
            payload["trace"] = "".join(
                traceback.format_exception(*record.exc_info)
            ).rstrip()

        for k, v in record.__dict__.items():
            if k in self._RESERVED_ATTRS:
                continue
            if k in payload:
                continue
            payload[k] = v

        return json.dumps(payload, default=str)


def configure_logging(level: int | str = logging.INFO) -> None:
    """Install the JSON formatter on root + uvicorn loggers.

    Idempotent: re-runs clear the previous handler so we don't double-log.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(name)
        for h in list(uv_logger.handlers):
            uv_logger.removeHandler(h)
        uv_logger.propagate = True
