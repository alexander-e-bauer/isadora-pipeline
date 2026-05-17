"""Day-by-day options backtest replay engine (Task 4.3).

Submodules
----------
``fill_model``  — pure function that computes fill price given bid/ask + a
                  slippage tolerance pulled from the DSL risk_box.
``metrics``     — pure functions computing total_return, cagr, sharpe,
                  max_drawdown, win_rate, etc. from a NAV series.
``engine``      — the day-by-day replay loop.  Produces a deterministic
                  ``BacktestResult`` dataclass containing metrics + a
                  content_hash that freezes the result for 17a-4 audit.

Design notes
------------
- v1 supports only the ``covered_call`` template.
- Daily-bar fills only — no intraday microstructure modeling.
- No Greeks recomputation; the EOD-recorded Greeks (when present) are used.
- Determinism is a hard requirement.  Floats are serialised at fixed
  precision (6 decimals) before hashing; iteration order is stable.
"""
from __future__ import annotations

from xyz.backtest.engine import BacktestResult, run_backtest
from xyz.backtest.fill_model import compute_fill
from xyz.backtest.metrics import (
    cagr,
    max_drawdown,
    sharpe,
    total_return,
)

__all__ = [
    "BacktestResult",
    "run_backtest",
    "compute_fill",
    "cagr",
    "max_drawdown",
    "sharpe",
    "total_return",
]
