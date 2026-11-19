"""Embedding helpers: a deterministic offline mock, a replay cache and an Ollama client."""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from typing import Sequence

from langchain_core.embeddings import Embeddings

from .errors import EmbeddingBackendError, InvalidConfigError

_CJK = "぀-ヿ㐀-䶿一-鿿"
# CJK runs become character bigrams (no spaces to split on); other words stay whole.
_TOKEN = re.compile(rf"[{_CJK}]+|(?:(?![{_CJK}])\w)+")
_CJK_RUN = re.compile(rf"[{_CJK}]+")


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if _CJK_RUN.fullmatch(tok):
            out.extend([tok] if len(tok) == 1 else [tok[i : i + 2] for i in range(len(tok) - 1)])
        else:
            out.append(tok)
    return out


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-words embedding for offline practice, tests and CI.

    Every token is hashed (sha256) into one of ``dim`` buckets and the vector is
    L2-normalised. Sentences that share words end up close together, so topic
    boundaries do show up. It matches **word overlap, not meaning**: synonyms and
    paraphrases are not recognised. Use a real embedding model for real work.
    """

    def __init__(self, dim: int = 256) -> None:
        if not isinstance(dim, int) or dim < 1:
            raise InvalidConfigError(f"dim must be a positive integer, got {dim!r}.", key="err.lib.dim", value=dim)
        self.dim = dim

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            vec[int(hashlib.sha256(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)


class ReplayEmbeddings(Embeddings):
    """Embeds every distinct text once and replays the stored vector afterwards.

    Parity checks hand the same instance to the old and the new implementation,
    so both see identical vectors even when the backend is not deterministic.
    """

    def __init__(self, inner: Embeddings) -> None:
        self.inner = inner
        self.cache: dict[str, list[float]] = {}
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            self.calls += 1
            for text, vec in zip(missing, self.inner.embed_documents(missing)):
                self.cache[text] = list(vec)
        return [self.cache[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)


class OllamaEmbeddings(Embeddings):
    """Minimal client for a local Ollama server (``POST /api/embed``), standard library only."""

    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: float = 300.0) -> None:
        if not model:
            raise InvalidConfigError(
                "Ollama model name is empty. Use --embeddings ollama:<model>.", key="err.lib.ollama_model"
            )
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": list(texts)}).encode()
        request = urllib.request.Request(
            f"{self.host}/api/embed", data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:  # noqa: S310 (user-chosen host)
                payload = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise EmbeddingBackendError(
                f"Could not get embeddings from Ollama at {self.host} ({exc}). "
                "Start the server (ollama serve), pull the model (ollama pull "
                f"{self.model}) or set OLLAMA_HOST.",
                key="err.lib.ollama_unreachable", host=self.host, reason=exc, model=self.model,
            ) from exc
        vectors = payload.get("embeddings") if isinstance(payload, dict) else None
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingBackendError(
                f"Unexpected reply from Ollama at {self.host}: expected {len(texts)} embeddings.",
                key="err.lib.ollama_reply", host=self.host, count=len(texts),
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def embedding_from_spec(spec: str, env: "dict[str, str] | None" = None) -> Embeddings:
    """Build an embedding from a CLI spec: ``hash`` or ``ollama:<model>``."""
    import os

    environ = os.environ if env is None else env
    if spec == "hash":
        return HashEmbeddings()
    if spec.startswith("ollama:"):
        return OllamaEmbeddings(spec[len("ollama:") :], host=environ.get("OLLAMA_HOST", "http://localhost:11434"))
    raise InvalidConfigError(
        f"Unknown embeddings spec {spec!r}. Use 'hash' or 'ollama:<model>'.",
        key="err.lib.embeddings_spec", spec=spec,
    )


def batched(items: Sequence[str], size: int | None) -> list[list[str]]:
    """Split ``items`` into lists of at most ``size`` (one list when ``size`` is None)."""
    if size is None:
        return [list(items)]
    return [list(items[i : i + size]) for i in range(0, len(items), size)]
