"""Shared test helpers (no network, no keys)."""

from __future__ import annotations

from pathlib import Path

from langchain_core.embeddings import Embeddings

from semsplit import HashEmbeddings

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "examples" / "data"

WORDS = ["harbor", "budget", "garden", "engine", "museum", "orchard",
         "tunnel", "glacier", "market", "bakery", "library", "vineyard"]


def data(name: str) -> str:
    return (DATA / name).read_text(encoding="utf-8")


def make_doc(sentences: int, per_topic: int = 9) -> str:
    """A deterministic document whose topic changes every ``per_topic`` sentences."""
    out = []
    for i in range(sentences):
        w = WORDS[(i // per_topic) % len(WORDS)]
        out.append(f"The {w} committee reviewed {w} plans and the {w} budget for item {i}.")
    return " ".join(out)


class CountingEmbeddings(Embeddings):
    """HashEmbeddings that records every embed_documents call."""

    def __init__(self) -> None:
        self.inner = HashEmbeddings()
        self.batches: list[int] = []

    @property
    def calls(self) -> int:
        return len(self.batches)

    @property
    def texts(self) -> int:
        return sum(self.batches)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(len(texts))
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)
