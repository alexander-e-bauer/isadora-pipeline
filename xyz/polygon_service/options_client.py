"""Polygon REST client for options data.

Covers three endpoints:
  - GET /v3/snapshot/options/{underlyingAsset}   → chain snapshot
  - GET /v3/quotes/{optionsTicker}               → last quote for one contract
  - GET /v2/aggs/ticker/{ticker}/range/1/{span}/{from}/{to} → historical bars

All HTTP is done via httpx (synchronous client) so the caller controls
threading.  No polygon-api-client SDK dependency.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

import httpx

from xyz.polygon_service.cache import ChainCache
from xyz.polygon_service.rate_limit import RateLimiter


# ---------------------------------------------------------------------------
# Return-type dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OptionContract:
    contract_ticker: str
    underlying: str
    expiry: date
    strike: float
    option_type: Literal["CALL", "PUT"]
    bid: float | None = None
    ask: float | None = None
    mid: float | None = None
    last: float | None = None
    volume: int | None = None
    open_interest: int | None = None
    implied_vol: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None


@dataclass
class ChainSnapshot:
    underlying: str
    underlying_price: float
    asof_at: datetime
    contracts: list[OptionContract] = field(default_factory=list)


@dataclass
class Quote:
    contract_ticker: str
    bid: float | None = None
    ask: float | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    timestamp: datetime | None = None


@dataclass
class Aggregate:
    contract_ticker: str
    timestamp: datetime      # bar open-time (ms → UTC datetime)
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(s: str) -> date:
    """Parse ISO-8601 date string."""
    return date.fromisoformat(s)


def _parse_contract(result: dict, underlying: str) -> OptionContract:
    """Map one Polygon v3 snapshot result into an OptionContract."""
    details = result.get("details", {})
    greeks = result.get("greeks", {}) or {}
    last_quote = result.get("last_quote", {}) or {}
    day = result.get("day", {}) or {}

    raw_type = details.get("contract_type", "").upper()
    option_type: Literal["CALL", "PUT"] = "CALL" if raw_type == "CALL" else "PUT"

    bid = last_quote.get("bid") or last_quote.get("P")
    ask = last_quote.get("ask") or last_quote.get("a") or last_quote.get("A")
    mid: float | None = None
    if bid is not None and ask is not None:
        mid = (float(bid) + float(ask)) / 2.0

    return OptionContract(
        contract_ticker=details.get("ticker", ""),
        underlying=underlying.upper(),
        expiry=_parse_date(details["expiration_date"]),
        strike=float(details["strike_price"]),
        option_type=option_type,
        bid=float(bid) if bid is not None else None,
        ask=float(ask) if ask is not None else None,
        mid=mid,
        last=float(day["close"]) if day.get("close") is not None else None,
        volume=int(day["volume"]) if day.get("volume") is not None else None,
        open_interest=int(result["open_interest"]) if result.get("open_interest") is not None else None,
        implied_vol=float(result["implied_volatility"]) if result.get("implied_volatility") is not None else None,
        delta=float(greeks["delta"]) if greeks.get("delta") is not None else None,
        gamma=float(greeks["gamma"]) if greeks.get("gamma") is not None else None,
        theta=float(greeks["theta"]) if greeks.get("theta") is not None else None,
        vega=float(greeks["vega"]) if greeks.get("vega") is not None else None,
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OptionsClient:
    """Synchronous Polygon REST client for options market data."""

    BASE_URL = "https://api.polygon.io"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        rate_limiter: RateLimiter | None = None,
        cache: ChainCache | None = None,
    ) -> None:
        self._api_key = api_key or os.environ["POLYGON_KEY"]
        self._http = httpx.Client(timeout=30.0)
        self._rl = rate_limiter or RateLimiter()
        self._cache = cache or ChainCache(ttl_seconds=300)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_chain(
        self,
        underlying: str,
        *,
        asof: datetime | None = None,
        expiration_gte: date | None = None,
        expiration_lte: date | None = None,
    ) -> ChainSnapshot:
        """Return a full chain snapshot from Polygon v3 snapshot endpoint.

        Results are cached for 5 minutes keyed on (underlying, minute, exp bounds).
        """
        cache_key = ChainCache.make_key(underlying, asof, expiration_gte, expiration_lte)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        params: dict = {"apiKey": self._api_key, "limit": 250}
        if expiration_gte:
            params["expiration_date.gte"] = expiration_gte.isoformat()
        if expiration_lte:
            params["expiration_date.lte"] = expiration_lte.isoformat()

        contracts: list[OptionContract] = []
        underlying_price: float = 0.0
        asof_at: datetime = asof or datetime.utcnow()
        next_url: str | None = f"{self.BASE_URL}/v3/snapshot/options/{underlying.upper()}"

        while next_url:
            self._rl.acquire()
            resp = self._http.get(next_url, params=params)
            resp.raise_for_status()
            body = resp.json()

            for result in body.get("results", []):
                contracts.append(_parse_contract(result, underlying))

            # underlying price is in the first result's underlying block
            if not underlying_price and body.get("results"):
                ub = body["results"][0].get("underlying_asset", {}) or {}
                if ub.get("price"):
                    underlying_price = float(ub["price"])

            # Polygon returns a cursor URL for pagination
            next_url = body.get("next_url")
            if next_url:
                # next_url already has cursor; still need apiKey
                params = {"apiKey": self._api_key}

        snapshot = ChainSnapshot(
            underlying=underlying.upper(),
            underlying_price=underlying_price,
            asof_at=asof_at,
            contracts=contracts,
        )
        self._cache.put(cache_key, snapshot)
        return snapshot

    def get_last_quote(self, contract_ticker: str) -> Quote:
        """Return the most recent quote for a single options contract."""
        self._rl.acquire()
        resp = self._http.get(
            f"{self.BASE_URL}/v3/quotes/{contract_ticker}",
            params={"apiKey": self._api_key, "limit": 1, "order": "desc", "sort": "timestamp"},
        )
        resp.raise_for_status()
        body = resp.json()

        results = body.get("results", [])
        if not results:
            return Quote(contract_ticker=contract_ticker)

        r = results[0]
        ts_ns = r.get("sip_timestamp") or r.get("participant_timestamp")
        ts: datetime | None = None
        if ts_ns is not None:
            ts = datetime.utcfromtimestamp(int(ts_ns) / 1e9)

        return Quote(
            contract_ticker=contract_ticker,
            bid=float(r["bid_price"]) if r.get("bid_price") is not None else None,
            ask=float(r["ask_price"]) if r.get("ask_price") is not None else None,
            bid_size=int(r["bid_size"]) if r.get("bid_size") is not None else None,
            ask_size=int(r["ask_size"]) if r.get("ask_size") is not None else None,
            timestamp=ts,
        )

    def get_historical_aggs(
        self,
        contract_ticker: str,
        from_date: date,
        to_date: date,
        timespan: str = "day",
    ) -> list[Aggregate]:
        """Return historical aggregate bars for a single options contract."""
        self._rl.acquire()
        resp = self._http.get(
            f"{self.BASE_URL}/v2/aggs/ticker/{contract_ticker}/range/1/{timespan}"
            f"/{from_date.isoformat()}/{to_date.isoformat()}",
            params={"apiKey": self._api_key, "adjusted": "false", "sort": "asc"},
        )
        resp.raise_for_status()
        body = resp.json()

        aggs: list[Aggregate] = []
        for bar in body.get("results", []):
            ts = datetime.utcfromtimestamp(bar["t"] / 1000.0)
            aggs.append(
                Aggregate(
                    contract_ticker=contract_ticker,
                    timestamp=ts,
                    open=float(bar["o"]) if bar.get("o") is not None else None,
                    high=float(bar["h"]) if bar.get("h") is not None else None,
                    low=float(bar["l"]) if bar.get("l") is not None else None,
                    close=float(bar["c"]) if bar.get("c") is not None else None,
                    volume=float(bar["v"]) if bar.get("v") is not None else None,
                )
            )
        return aggs

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "OptionsClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
