"""Diagnostics: the distance array, the threshold and where the cuts fall."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .sentences import Span

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


@dataclass
class Analysis:
    """Everything the chunker decided before it assembled chunks.

    ``distances[i]`` is the cosine distance between sentence ``i`` and ``i + 1``. A cut
    after sentence ``i`` happens when ``values[i] > threshold`` (``values`` equals
    ``distances`` except for the ``gradient`` type). ``fragile`` lists indices whose value
    is within ``eps`` of the threshold: a tiny numeric difference could flip those cuts.
    """

    sentences: list[str]
    spans: Optional[list[Span]]
    distances: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    threshold: Optional[float] = None
    breakpoints: list[int] = field(default_factory=list)
    fragile: list[int] = field(default_factory=list)
    threshold_type: str = "percentile"
    amount: Optional[float] = None
    embed_calls: int = 0

    @property
    def chunk_count(self) -> int:
        """Number of chunks the cuts alone would give (before size options)."""
        return len(self.breakpoints) + 1 if self.sentences else 0


def analyze(text: str, embeddings: "Embeddings", *, eps: float = 1e-9, **options: Any) -> Analysis:
    """Analyse ``text`` with the same options as :class:`semsplit.SemanticChunker`."""
    from .chunker import SemanticChunker

    return SemanticChunker(embeddings, **options).analyze(text, eps=eps)
