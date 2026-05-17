"""Strategy DSL validator.

Public surface:
    validate_dsl(dsl: dict) -> tuple[bool, list[str]]

Behavior:
- Returns ``(True, [])`` when the DSL satisfies the JSON-Schema *and* is
  ``kind: "declarative"``.
- Returns ``(False, [errors...])`` otherwise.
- ``kind: "scripted"`` is rejected at this layer (NOT at the schema) with
  the literal error string ``"scripted_not_supported_in_v1"`` so that the
  caller can detect the v1 limitation distinctly from generic schema errors.
- Top-level ``selection`` / ``trigger`` / ``action`` / ``exit`` / ``risk_box``
  missing → returns the missing key in the errors list.

The validator collects ALL errors (not just the first) so the caller can
surface them in one shot to the advisor UI.
"""
from __future__ import annotations

from typing import Any

# Importing jsonschema at module load is intentional — the library is a
# hard dependency for any caller of validate_dsl, and lazy import would
# only mask a missing dep until call time (worse error surface).
import jsonschema
from jsonschema import Draft202012Validator

from xyz.dsl.schema import DSL_SCHEMA

# A single shared validator instance so we don't re-compile the schema on
# every call.  The Draft202012Validator is stateless wrt input documents.
_VALIDATOR = Draft202012Validator(DSL_SCHEMA)

_SCRIPTED_ERR = "scripted_not_supported_in_v1"


def validate_dsl(dsl: Any) -> tuple[bool, list[str]]:
    """Validate a Strategy DSL document.

    Parameters
    ----------
    dsl:
        Parsed JSON object (Python dict) representing a strategy template.

    Returns
    -------
    (valid, errors):
        ``valid`` is True iff the document satisfies the schema AND
        ``kind == "declarative"``.  ``errors`` is a list of human-readable
        error strings.  Empty when ``valid`` is True.
    """
    errors: list[str] = []

    # Guard: the input must be a dict at the top level.  jsonschema raises
    # a different shape of error for non-objects, and an empty/None payload
    # is the most common 'oops' from a bad client, so we short-circuit.
    if not isinstance(dsl, dict):
        return False, ["dsl must be a JSON object"]

    # Collect every schema violation.  We want all errors (not first) so
    # the UI can show the full diff.
    schema_errors = sorted(_VALIDATOR.iter_errors(dsl), key=lambda e: e.path)
    for err in schema_errors:
        errors.append(_format_error(err))

    # Explicit v1 rejection of scripted strategies.  We test even when the
    # schema fails because ``kind`` is the most-actionable signal for the
    # caller — they should not get buried under a wall of risk_box errors
    # if the real blocker is "this kind isn't supported".
    if isinstance(dsl.get("kind"), str) and dsl["kind"] == "scripted":
        errors.append(_SCRIPTED_ERR)

    return (len(errors) == 0), errors


def _format_error(err: jsonschema.ValidationError) -> str:
    """Render a jsonschema error as ``<path>: <message>``.

    Path defaults to ``<root>`` when the error is at the top of the doc
    (e.g. a missing required field).  We strip the full schema dump that
    jsonschema includes by default — it is noisy and not useful to the
    advisor UI.
    """
    path = ".".join(str(p) for p in err.path) or "<root>"
    return f"{path}: {err.message}"
