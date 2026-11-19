"""Parity checks: does semsplit produce the same chunks as the original implementation?

Both implementations receive the *same* embedding vectors (each distinct text is embedded
once and replayed), so any difference is a difference in the splitting code, not in a
non-deterministic embedding backend.
"""

from __future__ import annotations

import json
import platform
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
from langchain_core.embeddings import Embeddings

from .chunker import SemanticChunker
from .embeddings import HashEmbeddings, ReplayEmbeddings
from .errors import InvalidConfigError, LegacyNotInstalledError

GOLDEN_FORMAT = 1
LEGACY_VERSION = "0.4.2"

#: Configurations compared when the caller does not pass any (compat arguments only).
DEFAULT_CONFIGS: list[dict[str, Any]] = [
    {"breakpoint_threshold_type": "percentile"},
    {"breakpoint_threshold_type": "percentile", "breakpoint_threshold_amount": 50},
    {"breakpoint_threshold_type": "standard_deviation"},
    {"breakpoint_threshold_type": "interquartile"},
    {"breakpoint_threshold_type": "gradient"},
    {"number_of_chunks": 3},
    {"buffer_size": 0, "breakpoint_threshold_amount": 50},
    {"buffer_size": 2, "breakpoint_threshold_amount": 50},
]


@dataclass
class CaseResult:
    """Outcome of one (document, configuration) comparison."""

    doc: str
    config: dict[str, Any]
    equal: bool
    legacy_chunks: list[str]
    new_chunks: list[str]
    legacy_start_index: list[int]
    new_start_index: list[int]
    diff_boundaries: dict[str, list[int]] = field(default_factory=dict)
    first_mismatch: Optional[dict[str, int]] = None
    fragile: list[int] = field(default_factory=list)


@dataclass
class ParityReport:
    """All cases of a parity run."""

    against: str
    cases: list[CaseResult]

    @property
    def all_equal(self) -> bool:
        return all(c.equal for c in self.cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "against": self.against,
            "all_equal": self.all_equal,
            "cases": [asdict(c) for c in self.cases],
        }


def legacy_chunker_class() -> Any:
    """Import the archived ``SemanticChunker`` lazily; raise a helpful error when missing."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the package prints its own sunset notice
            from langchain_experimental.text_splitter import SemanticChunker as Legacy
    except ImportError as exc:
        raise LegacyNotInstalledError() from exc
    return Legacy


def _boundaries(chunks: Sequence[str]) -> list[int]:
    out, total = [], 0
    for chunk in chunks[:-1]:
        total += len(chunk)
        out.append(total)
    return out


def _compare(
    doc: str, config: dict[str, Any], expected: tuple[list[str], list[int]],
    actual: tuple[list[str], list[int]], fragile: list[int],
) -> CaseResult:
    exp_chunks, exp_starts = expected
    new_chunks, new_starts = actual
    equal = exp_chunks == new_chunks and exp_starts == new_starts
    diff: dict[str, list[int]] = {}
    mismatch: Optional[dict[str, int]] = None
    if not equal:
        eb, nb = set(_boundaries(exp_chunks)), set(_boundaries(new_chunks))
        diff = {"only_legacy": sorted(eb - nb), "only_new": sorted(nb - eb)}
        for i in range(max(len(exp_chunks), len(new_chunks))):
            a = exp_chunks[i] if i < len(exp_chunks) else None
            b = new_chunks[i] if i < len(new_chunks) else None
            if a != b:
                offset = 0
                if a is not None and b is not None:
                    offset = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                mismatch = {"chunk_index": i, "char_offset": offset}
                break
        if mismatch is None:  # same chunks, different start_index values
            mismatch = {"chunk_index": next(i for i, (x, y) in enumerate(zip(exp_starts, new_starts)) if x != y), "char_offset": 0}
    return CaseResult(
        doc=doc, config=config, equal=equal,
        legacy_chunks=exp_chunks, new_chunks=new_chunks,
        legacy_start_index=exp_starts, new_start_index=new_starts,
        diff_boundaries=diff, first_mismatch=mismatch, fragile=fragile,
    )


def _run(cls: Any, embeddings: Embeddings, text: str, config: Mapping[str, Any]) -> tuple[list[str], list[int]]:
    docs = cls(embeddings, add_start_index=True, **config).create_documents([text])
    return [d.page_content for d in docs], [d.metadata["start_index"] for d in docs]


def load_golden(path: Union[str, Path]) -> dict[str, Any]:
    """Read and validate a golden file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InvalidConfigError(
            f"Cannot read golden file {path}: {exc}. Record one with --record.",
            key="err.lib.golden_read", path=path, reason=exc,
        ) from exc
    if not isinstance(data, dict) or data.get("format") != GOLDEN_FORMAT or not isinstance(data.get("cases"), list):
        raise InvalidConfigError(
            f"{path} is not a semsplit golden file (format {GOLDEN_FORMAT}).",
            key="err.lib.golden_format", path=path, format=GOLDEN_FORMAT,
        )
    return data


def check_parity(
    texts: "Mapping[str, str] | Sequence[str]",
    embeddings: Optional[Embeddings] = None,
    configs: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    against: str = "legacy",
    eps: float = 1e-9,
) -> ParityReport:
    """Compare semsplit with the original implementation or with a golden file.

    Args:
        texts: documents by name (or a list, named ``doc0``, ``doc1`` ...). Ignored for golden files.
        embeddings: embedding to use (default ``HashEmbeddings``).
        configs: ``SemanticChunker`` keyword sets to compare (default :data:`DEFAULT_CONFIGS`).
        against: ``"legacy"`` (needs ``semsplit[legacy]``) or the path of a golden file.
        eps: a cut whose value is within ``eps`` of the threshold is reported as fragile.
    """
    if against == "legacy":
        legacy = legacy_chunker_class()
        named = dict(texts) if isinstance(texts, Mapping) else {f"doc{i}": t for i, t in enumerate(texts)}
        replay = ReplayEmbeddings(embeddings or HashEmbeddings())
        cases = []
        for name, text in named.items():
            for config in (configs if configs is not None else DEFAULT_CONFIGS):
                cfg = dict(config)
                expected = _run(legacy, replay, text, cfg)
                actual = _run(SemanticChunker, replay, text, cfg)
                fragile = SemanticChunker(replay, **cfg).analyze(text, eps=eps).fragile
                cases.append(_compare(name, cfg, expected, actual, fragile))
        return ParityReport(against="legacy", cases=cases)

    golden = load_golden(against)
    replay = ReplayEmbeddings(embeddings or HashEmbeddings())
    cases = []
    for case in golden["cases"]:
        cfg = dict(case["config"])
        expected = (list(case["chunks"]), list(case["start_index"]))
        actual = _run(SemanticChunker, replay, case["text"], cfg)
        fragile = SemanticChunker(replay, **cfg).analyze(case["text"], eps=eps).fragile
        cases.append(_compare(str(case.get("id", "case")), cfg, expected, actual, fragile))
    return ParityReport(against=f"golden:{against}", cases=cases)


def record_golden(
    texts: "Mapping[str, str] | Sequence[str]",
    path: Union[str, Path],
    embeddings: Optional[Embeddings] = None,
    configs: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    embeddings_name: str = "hash",
    source: str = "legacy",
) -> dict[str, Any]:
    """Run ``source`` (``"legacy"`` or ``"semsplit"``) and store its output as a golden file."""
    if source not in ("legacy", "semsplit"):
        raise InvalidConfigError("source must be 'legacy' or 'semsplit'.", key="err.lib.source")
    cls = legacy_chunker_class() if source == "legacy" else SemanticChunker
    named = dict(texts) if isinstance(texts, Mapping) else {f"doc{i}": t for i, t in enumerate(texts)}
    replay = ReplayEmbeddings(embeddings or HashEmbeddings())
    cases = []
    for name, text in named.items():
        for n, config in enumerate(configs if configs is not None else DEFAULT_CONFIGS):
            cfg = dict(config)
            chunks, starts = _run(cls, replay, text, cfg)
            cases.append({"id": f"{name}:{n}", "text": text, "config": cfg, "chunks": chunks, "start_index": starts})
    data = {
        "format": GOLDEN_FORMAT,
        "generator": {
            "source": source,
            "langchain-experimental": LEGACY_VERSION if source == "legacy" else None,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "embeddings": embeddings_name,
        "cases": cases,
    }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return data
