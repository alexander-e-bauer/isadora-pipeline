"""ForecastDb — thin SQLAlchemy wrapper for the forecast modules.

Exposes the helper methods that ForecastAgent, Calibrator, and
build_t0_market_context call:

    get_daily_ohlcv_df         — resample 30m bars to daily OHLCV
    get_daily_log_returns      — log-returns over one or more windows
    get_atm_iv_series          — ATM implied-vol time series from option_historical_eod
    get_t0_iv_surface          — {(strike, dte): iv} dict at t0
    get_t0_iv_atm              — scalar ATM IV at t0 (median of surface)
    get_t0_realized_volatility — annualised realised vol over trailing 20 days
    get_t0_spot                — close price at or before t0
    get_market_emb_day_at      — most-recent MarketEmbDay row at/before t0
    get_market_emb_week_at     — most-recent MarketEmbWeek row at/before t0
    emit_event                 — write a hash-chained audit event

All queries use ``tenant_session()`` (context manager from
``xyz.tenant.db``) which wraps engine's existing SQLAlchemy pool.

Note on OptionHistoricalEod column name
----------------------------------------
The model at ``xyz.polygon_service.models.OptionHistoricalEod`` stores
implied volatility as ``implied_vol`` (not ``iv``).  All calls below
reference that column name explicitly.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


class ForecastDb:
    """Thin wrapper that exposes the helper methods the forecast modules call.

    The ``session_factory`` argument is accepted to match the plan's
    ``make_forecast_db(session_factory)`` factory call in app.py.  Internally
    all session lifecycle is managed via ``tenant_session()`` (commit/rollback
    context manager) rather than a raw ``Session`` so callers don't need to
    worry about transaction hygiene.
    """

    def __init__(self, session_factory=None):
        # session_factory accepted for forward-compat / testing but the
        # default open-session path uses tenant_session() directly.
        self._session_factory = session_factory

    def _session(self):
        """Return a context manager that yields a committed Session."""
        from xyz.tenant.db import tenant_session
        return tenant_session()

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def emit_event(self, *, kind, firm_id, actor_user_id, payload):
        from xyz.tenant.events import emit_event as _emit

        with self._session() as s:
            _emit(
                db=s,
                kind=kind,
                firm_id=firm_id,
                actor_user_id=actor_user_id,
                payload=payload,
            )

    # ------------------------------------------------------------------
    # OHLCV helpers
    # ------------------------------------------------------------------

    def get_daily_ohlcv_df(self, *, symbol: str, t0: date, days: int = 252) -> pd.DataFrame:
        """Return a daily OHLCV DataFrame for *symbol* ending at *t0*.

        Queries ``historical_data`` (30m bars stored as Unix timestamps),
        resamples to business-day bars, and returns the trailing *days* rows.
        """
        from xyz.finazon_service.sql_service import HistoricalData, Ticker

        # Look back 2× days to have enough 30m bars after resampling
        t0_ts = int(
            (t0 - timedelta(days=0)).strftime("%s") or 0
        )
        lookback_ts = int(
            (t0 - timedelta(days=days * 2)).strftime("%s") or 0
        )

        with self._session() as s:
            ticker = s.query(Ticker).filter_by(symbol=symbol).first()
            if not ticker:
                raise ValueError(f"unknown symbol: {symbol}")
            rows = (
                s.query(HistoricalData)
                .filter(HistoricalData.ticker_id == ticker.id)
                .filter(HistoricalData.timestamp >= lookback_ts)
                .filter(HistoricalData.timestamp <= t0_ts)
                .order_by(HistoricalData.timestamp.asc())
                .all()
            )

        if not rows:
            return pd.DataFrame(
                columns=["timestamp", "datetime", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame([{
            "timestamp": r.timestamp,
            "datetime": pd.to_datetime(r.timestamp, unit="s"),
            "open": r.open, "high": r.high, "low": r.low,
            "close": r.close, "volume": r.volume,
        } for r in rows])
        df.set_index("datetime", inplace=True)
        daily = df.resample("B").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        daily.reset_index(inplace=True)
        daily["timestamp"] = daily["datetime"].astype("int64") // 10 ** 9
        return daily.tail(days)

    def get_daily_log_returns(self, *, symbol: str, windows) -> np.ndarray:
        """Concatenate log-returns from each (start, end) window."""
        out: list[float] = []
        for w_start, w_end in windows:
            df = self.get_daily_ohlcv_df(
                symbol=symbol,
                t0=w_end,
                days=(w_end - w_start).days + 1,
            )
            closes = df["close"].to_numpy()
            if len(closes) > 1:
                out.extend(np.diff(np.log(closes + 1e-12)))
        return np.array(out)

    # ------------------------------------------------------------------
    # Options / IV helpers
    # ------------------------------------------------------------------

    def get_atm_iv_series(self, *, symbol: str, windows) -> np.ndarray:
        """Return a 1-D array of daily ATM implied-vol values across windows.

        For each day, picks the CALL contract whose delta is closest to 0.5
        as the ATM proxy.  Reads from ``option_historical_eod``; the IV
        column is ``implied_vol`` (not ``iv``).
        """
        from xyz.polygon_service.models import OptionHistoricalEod

        all_iv: list[float] = []
        with self._session() as s:
            for w_start, w_end in windows:
                rows = (
                    s.query(OptionHistoricalEod)
                    .filter(OptionHistoricalEod.underlying == symbol)
                    .filter(OptionHistoricalEod.date >= w_start)
                    .filter(OptionHistoricalEod.date <= w_end)
                    .filter(OptionHistoricalEod.option_type == "CALL")
                    .all()
                )
                # Group by date, keep the row whose delta is closest to 0.5
                by_date: dict[date, tuple[float, float]] = {}  # {date: (delta, iv)}
                for r in rows:
                    if r.delta is None or r.implied_vol is None:
                        continue
                    d = r.date
                    if d not in by_date or abs(float(r.delta) - 0.5) < abs(by_date[d][0] - 0.5):
                        by_date[d] = (float(r.delta), float(r.implied_vol))
                for d in sorted(by_date.keys()):
                    all_iv.append(by_date[d][1])
        return np.array(all_iv)

    def get_t0_iv_surface(self, *, symbol: str, t0: date) -> dict[tuple[float, int], float]:
        """Return {(strike, dte): implied_vol} for the full CALL chain at t0."""
        from xyz.polygon_service.models import OptionHistoricalEod

        with self._session() as s:
            rows = (
                s.query(OptionHistoricalEod)
                .filter(OptionHistoricalEod.underlying == symbol)
                .filter(OptionHistoricalEod.date == t0)
                .filter(OptionHistoricalEod.option_type == "CALL")
                .all()
            )
            surface: dict[tuple[float, int], float] = {}
            for r in rows:
                if r.implied_vol is None or r.strike is None or r.expiry is None:
                    continue
                dte = (r.expiry - t0).days
                if dte <= 0:
                    continue
                surface[(float(r.strike), int(dte))] = float(r.implied_vol)
        return surface

    def get_t0_iv_atm(self, *, symbol: str, t0: date) -> float:
        """Scalar ATM IV at t0 — median of the IV surface; 0.20 if no data."""
        surf = self.get_t0_iv_surface(symbol=symbol, t0=t0)
        if not surf:
            return 0.20
        return float(np.median(list(surf.values())))

    def get_t0_realized_volatility(self, *, symbol: str, t0: date) -> float:
        """Annualised realised volatility from trailing 20 daily closes."""
        df = self.get_daily_ohlcv_df(symbol=symbol, t0=t0, days=20)
        closes = df["close"].to_numpy()
        if len(closes) < 2:
            return 0.20
        log_ret = np.diff(np.log(closes + 1e-12))
        return float(np.std(log_ret, ddof=1) * np.sqrt(252))

    def get_t0_spot(self, *, symbol: str, t0: date) -> float:
        """Most-recent close price at or before t0."""
        df = self.get_daily_ohlcv_df(symbol=symbol, t0=t0, days=5)
        if df.empty:
            return 100.0
        return float(df["close"].iloc[-1])

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def get_market_emb_day_at(self, *, symbol: str, t0: date):
        """Return the most-recent MarketEmbDay row at or before t0, or None."""
        from xyz.finazon_service.sql_service import MarketEmbDay, Ticker

        with self._session() as s:
            ticker = s.query(Ticker).filter_by(symbol=symbol).first()
            if not ticker:
                return None
            row = (
                s.query(MarketEmbDay)
                .filter(MarketEmbDay.ticker_id == ticker.id)
                .filter(MarketEmbDay.period_start <= t0)
                .order_by(MarketEmbDay.period_start.desc())
                .first()
            )
            if row is None:
                return None
            return {
                "market_summary": row.market_summary,
                "trend_strength": row.trend_strength,
                "volatility_regime": row.volatility_regime,
                "momentum_phase": row.momentum_phase,
                "technical_signals": row.technical_signals or "",
                "risk_level": row.risk_level,
                "news_headlines": row.news_headlines,
                "news_flags": row.news_flags or "",
                "period_start": (
                    row.period_start.date()
                    if hasattr(row.period_start, "date")
                    else row.period_start
                ),
                "period_end": (
                    row.period_end.date()
                    if hasattr(row.period_end, "date")
                    else row.period_end
                ),
            }

    def get_market_emb_week_at(self, *, symbol: str, t0: date):
        """Return the most-recent MarketEmbWeek row at or before t0, or None."""
        from xyz.finazon_service.sql_service import MarketEmbWeek, Ticker

        with self._session() as s:
            ticker = s.query(Ticker).filter_by(symbol=symbol).first()
            if not ticker:
                return None
            row = (
                s.query(MarketEmbWeek)
                .filter(MarketEmbWeek.ticker_id == ticker.id)
                .filter(MarketEmbWeek.period_start <= t0)
                .order_by(MarketEmbWeek.period_start.desc())
                .first()
            )
            if row is None:
                return None
            return {
                "market_summary": row.market_summary,
                "news_flags": row.news_flags or "",
                "period_start": (
                    row.period_start.date()
                    if hasattr(row.period_start, "date")
                    else row.period_start
                ),
                "period_end": (
                    row.period_end.date()
                    if hasattr(row.period_end, "date")
                    else row.period_end
                ),
            }


def make_forecast_db(session_factory=None) -> ForecastDb:
    """Factory used by app.py to construct a ForecastDb for a request."""
    return ForecastDb(session_factory=session_factory)
