"""Citation dataclass — a pointer to a source backing a claim in a research artifact."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Citation:
    """A pointer to a source backing a claim in the research artifact.

    kind:
      - 'url'      → public web source (news headline, SEC filing)
      - 'db_row'   → reference to a row in our DB (e.g. computed_metrics:12345)
      - 'polygon'  → a Polygon API resource (e.g. snapshot:AAPL:2026-05-16T15:30Z)
    """

    kind: Literal["url", "db_row", "polygon"]
    source: str             # URL, table:id, or Polygon resource path
    excerpt: str | None = None   # the snippet supporting the claim

    def to_dict(self) -> dict:
        return {"kind": self.kind, "source": self.source, "excerpt": self.excerpt}
