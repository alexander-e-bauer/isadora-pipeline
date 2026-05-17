"""Anthropic Claude API wrapper with prompt caching.

This thin wrapper provides:
  - Single chokepoint for API key + model selection (testable via DI).
  - Centralised prompt-caching policy (cache_control on stable system blocks).
  - One place to swap providers without touching agent code.

Import is deferred at the class level so that the module can be imported
even if the anthropic SDK is not installed — the ImportError is only raised
when AnthropicClient is instantiated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "claude-sonnet-4-6"


@dataclass
class CachedSystemBlock:
    """A block of system-prompt text that should be prompt-cached.

    The Anthropic SDK only supports the ``ephemeral`` cache type today, so
    every block is emitted with ``cache_control={"type": "ephemeral"}``.
    A TTL field is intentionally omitted to avoid mis-signaling capability
    the API does not yet expose.
    """

    text: str


class AnthropicClient:
    """Thin wrapper around anthropic.Anthropic.

    Why a wrapper:
      - Single chokepoint for the API key + model selection (testable).
      - Centralised prompt-caching policy (cache_control on system blocks).
      - One place to swap to a different provider later without touching agents.

    Thread-safety: the lazy ``self._client`` initialization is **not**
    thread-safe.  Today this is fine because routes instantiate one
    AnthropicClient per request.  If this is ever promoted to an
    app-level singleton shared across the FastAPI threadpool, wrap the
    init block in a ``threading.Lock``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "anthropic SDK is required to use AnthropicClient. "
                "Install it with: pip install 'anthropic>=0.40'"
            ) from exc

        self._anthropic = _anthropic
        self._api_key = api_key
        self._model = model
        self._client: Any = None  # constructed on first complete() call

    def complete(
        self,
        *,
        system: list[CachedSystemBlock] | str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.2,
        tools: list[dict] | None = None,
    ) -> Any:
        """Send a message to Claude. System blocks marked with cache_control.

        Parameters
        ----------
        system:
            Either a plain string (no caching) or a list of CachedSystemBlock
            objects (each block is marked with cache_control=ephemeral).
        messages:
            OpenAI-style chat message dicts (role + content).
        max_tokens:
            Maximum completion tokens (default 4096).
        temperature:
            Sampling temperature (default 0.2 — lower for structured output).
        tools:
            Optional list of tool schemas for Claude tool-use.

        Returns
        -------
        anthropic.types.Message
            The raw SDK Message object.
        """
        if self._client is None:
            # Lazy: defer env-var read + SDK client construction until the
            # first actual API call.  Keeps construction free of side effects
            # so app startup never fails on a missing ANTHROPIC_API_KEY when
            # the route is never hit (e.g. unrelated requests, test imports).
            self._client = self._anthropic.Anthropic(
                api_key=self._api_key or os.environ["ANTHROPIC_API_KEY"]
            )

        if isinstance(system, str):
            system_param: Any = system
        else:
            system_param = [
                {
                    "type": "text",
                    "text": block.text,
                    "cache_control": {"type": "ephemeral"},
                }
                for block in system
            ]

        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_param,
            messages=messages,
        )
        if tools:
            kwargs["tools"] = tools

        return self._client.messages.create(**kwargs)
