"""JSON-Schema definition for the Strategy DSL (v1, declarative only).

The DSL spec lives in ``NORTH_STAR_SPEC.md`` §8.  This file mirrors that
spec as a Draft 2020-12 JSON-Schema document.  Validation is intentionally
strict — typos are caught via ``additionalProperties: false`` on every
``risk_box`` sub-object so an author can't silently rename a key.

What is enforced
----------------
- Top-level required keys: ``kind``, ``name``, ``version``, ``author_user_id``,
  ``firm_id``, ``selection``, ``trigger``, ``action``, ``exit``, ``risk_box``.
- ``kind`` is ``"declarative"`` or ``"scripted"`` at the schema level;
  ``"scripted"`` is rejected at the validator layer (see ``validate.py``)
  because it requires a separate code-execution sandbox not present in v1.
- ``autonomy_requirement`` values constrained to ``L0..L5``.
- ``risk_box`` sub-objects (``position_sizing``, ``greek_limits``, etc.)
  forbid additionalProperties so typos in nested keys are caught.

What is intentionally loose
---------------------------
- ``selection``, ``trigger``, ``action``, ``adjustment``, ``exit`` —
  these are free-form objects at the DSL layer.  Their internal shape is
  validated by the engine's *interpreter*, not by JSON-Schema.  The spec
  treats their semantic content as the responsibility of subsequent agents
  (BACKTEST, PROPOSE) which know the option-mechanics primitives.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Risk-box sub-schemas — every key is nullable; typos are forbidden.
# ---------------------------------------------------------------------------

_POSITION_SIZING = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "max_pct_account_notional": {"type": ["number", "null"]},
        "max_pct_position_shares": {"type": ["number", "null"]},
        "max_contracts_per_underlying": {"type": ["integer", "null"]},
    },
}

_GREEK_LIMITS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "max_short_delta_contribution": {"type": ["number", "null"]},
        "max_short_gamma": {"type": ["number", "null"]},
        "max_vega_per_underlying": {"type": ["number", "null"]},
        "max_theta_decay_target": {"type": ["number", "null"]},
        "max_portfolio_beta_delta": {"type": ["number", "null"]},
        "max_portfolio_short_gamma": {"type": ["number", "null"]},
    },
}

_LOSS_LIMITS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "max_loss_per_trade_pct": {"type": ["number", "null"]},
        "max_drawdown_pct": {"type": ["number", "null"]},
        "daily_realized_loss_stop": {"type": ["number", "null"]},
        "max_unrealized_drawdown_pct": {"type": ["number", "null"]},
        "max_consecutive_losses": {"type": ["integer", "null"]},
    },
}

_CONCENTRATION = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "max_underlying_pct": {"type": ["number", "null"]},
        "max_sector_pct": {"type": ["number", "null"]},
        "max_correlation_cluster_pct": {"type": ["number", "null"]},
        "max_notional_per_expiration_date": {"type": ["number", "null"]},
    },
}

_TIME_WINDOWS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "earnings_blackout_days": {"type": ["integer", "null"]},
        "fomc_blackout_days": {"type": ["integer", "null"]},
        "fed_speakers_blackout": {"type": ["boolean", "null"]},
        "ex_div_blackout_days": {"type": ["integer", "null"]},
        "no_open_after_time": {"type": ["string", "null"]},
        "no_open_before_dte": {"type": ["integer", "null"]},
        "no_trade_first_x_minutes": {"type": ["integer", "null"]},
        "friday_afternoon_lockdown": {"type": ["boolean", "null"]},
    },
}

_REGIME_GATES = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "min_iv_rank": {"type": ["number", "null"]},
        "max_iv_rank": {"type": ["number", "null"]},
        "vix_ceiling": {"type": ["number", "null"]},
        "term_structure_state": {"type": ["string", "null"]},
        "min_iv_rv_spread": {"type": ["number", "null"]},
        "underlying_technical_extreme": {"type": ["string", "null"]},
    },
}

_LIQUIDITY_GATES = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "min_open_interest": {"type": ["integer", "null"]},
        "min_avg_daily_volume": {"type": ["integer", "null"]},
        "max_bid_ask_spread_pct": {"type": ["number", "null"]},
    },
}

_ACCOUNT_GATES = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "min_buying_power_buffer_pct": {"type": ["number", "null"]},
        "margin_utilization_ceiling": {"type": ["number", "null"]},
    },
}

_EXECUTION_MICROSTRUCTURE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "max_order_retries": {"type": ["integer", "null"]},
        "max_slippage_tolerance_pct": {"type": ["number", "null"]},
        "max_order_size_vs_adv_pct": {"type": ["number", "null"]},
    },
}

_ASSIGNMENT_PIN_RISK = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pin_risk_window_dte": {"type": ["integer", "null"]},
        "max_itm_prob_at_dte": {"type": ["number", "null"]},
        "dividend_yield_early_exercise_threat": {"type": ["number", "null"]},
    },
}

_TAX_LOT_AWARENESS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "lot_method": {"type": ["string", "null"], "enum": ["FIFO", "LIFO", "HIFO", None]},
        "wash_sale_lockout_days": {"type": ["integer", "null"]},
    },
}

_CLIENT_OVERRIDES = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "do_not_call_above_strike": {"type": ["number", "null"]},
        "do_not_sell_below_price": {"type": ["number", "null"]},
        "restricted_tickers": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

_CROSS_STRATEGY_AGGREGATION = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "combined_delta_cap": {"type": ["number", "null"]},
        "combined_vega_cap": {"type": ["number", "null"]},
        "combined_short_gamma_cap": {"type": ["number", "null"]},
    },
}

_RISK_BOX = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "position_sizing": _POSITION_SIZING,
        "greek_limits": _GREEK_LIMITS,
        "loss_limits": _LOSS_LIMITS,
        "concentration": _CONCENTRATION,
        "time_windows": _TIME_WINDOWS,
        "regime_gates": _REGIME_GATES,
        "liquidity_gates": _LIQUIDITY_GATES,
        "account_gates": _ACCOUNT_GATES,
        "execution_microstructure": _EXECUTION_MICROSTRUCTURE,
        "assignment_pin_risk": _ASSIGNMENT_PIN_RISK,
        "tax_lot_awareness": _TAX_LOT_AWARENESS,
        "client_overrides": _CLIENT_OVERRIDES,
        "cross_strategy_aggregation": _CROSS_STRATEGY_AGGREGATION,
    },
}

# ---------------------------------------------------------------------------
# Autonomy requirement — every action family maps to L0..L5.
# ---------------------------------------------------------------------------

_AUTONOMY_LEVEL = {
    "type": "string",
    "enum": ["L0", "L1", "L2", "L3", "L4", "L5"],
}

_AUTONOMY_REQUIREMENT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "OPEN": _AUTONOMY_LEVEL,
        "CLOSE": _AUTONOMY_LEVEL,
        "ROLL": _AUTONOMY_LEVEL,
        "HEDGE": _AUTONOMY_LEVEL,
        "PORTFOLIO": _AUTONOMY_LEVEL,
        "EVENT": _AUTONOMY_LEVEL,
        "STATE": _AUTONOMY_LEVEL,
    },
}

# ---------------------------------------------------------------------------
# Top-level DSL schema.
# ---------------------------------------------------------------------------

DSL_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://option-overlays.dev/schemas/strategy-dsl-v1.json",
    "title": "Strategy DSL v1",
    "type": "object",
    "required": [
        "kind",
        "name",
        "version",
        "author_user_id",
        "firm_id",
        "selection",
        "trigger",
        "action",
        "exit",
        "risk_box",
    ],
    "additionalProperties": True,  # rationale / template / etc. may attach
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["declarative", "scripted"],
        },
        "name": {"type": "string", "minLength": 1, "maxLength": 255},
        "version": {"type": "integer", "minimum": 1},
        "author_user_id": {"type": "integer"},
        "firm_id": {"type": "integer"},
        "template": {
            "type": "string",
            "enum": ["covered_call", "cash_secured_put", "collar"],
        },
        "selection": {"type": "object"},
        "trigger": {"type": "object"},
        "action": {"type": "object"},
        "adjustment": {"type": "object"},
        "exit": {"type": "object"},
        "risk_box": _RISK_BOX,
        "autonomy_requirement": _AUTONOMY_REQUIREMENT,
        "rationale": {"type": "string"},
    },
}
