"""Strategy DSL — JSON-Schema-validated declarative strategy definitions.

The DSL is the canonical wire format for option-overlay strategy templates,
spec'd in `NORTH_STAR_SPEC.md` §8.  ``schema.py`` carries the JSON-Schema
definition; ``validate.py`` wraps it with a single ``validate_dsl`` helper.

v1 only supports ``kind: "declarative"`` — ``kind: "scripted"`` is reserved
for v2 and is rejected by the validator with the literal error string
``"scripted_not_supported_in_v1"``.
"""
from xyz.dsl.schema import DSL_SCHEMA
from xyz.dsl.validate import validate_dsl

__all__ = ["DSL_SCHEMA", "validate_dsl"]
