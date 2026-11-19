"""The SemanticChunker: split text where neighbouring sentence embeddings drift apart."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Optional, Sequence

import numpy as np
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings

from .analysis import Analysis
from .embeddings import batched
from .errors import InvalidConfigError
from .sentences import DEFAULT_SENTENCE_REGEX, PRESETS, SentenceSplit, Span, split_spans
from .sizing import split_long_text
from .thresholds import (
    BREAKPOINT_DEFAULTS,
    BreakpointThresholdType,
    breakpoint_values,
    cosine_distances,
    fragile_indices,
    threshold_from_chunk_count,
)

TextMode = Literal["legacy", "original"]

# Distances closer than this to the largest one count as a tie when re-cutting (max_chunk_size).
TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class _Piece:
    text: str
    start: Optional[int]  # offset in the source text; None in legacy mode


class SemanticChunker(BaseDocumentTransformer):
    """Split text into chunks at points where the embedding distance between
    neighbouring sentences is unusually large.

    Compatible with ``langchain_experimental.text_splitter.SemanticChunker`` 0.4.2: with the
    default arguments the output (and the legacy ``start_index`` values) is identical.
    The idea comes from Greg Kamradt's "5 Levels Of Text Splitting" notebook.

    Args:
        embeddings: any LangChain ``Embeddings``.
        buffer_size: neighbours on each side that are added to a sentence before embedding.
        add_start_index: put ``start_index`` in document metadata.
        breakpoint_threshold_type: ``percentile``, ``standard_deviation``, ``interquartile``
            or ``gradient``.
        breakpoint_threshold_amount: threshold parameter (default depends on the type).
        number_of_chunks: aim for this many chunks (overrides the threshold type).
        sentence_split_regex: custom regex; when it differs from the default it wins over
            ``sentence_split``.
        min_chunk_size: smaller pieces are merged into the next one.
        max_chunk_size: opt-in hard limit (in ``length_function`` units). Over-long chunks are
            re-cut at the largest distance inside them, without extra embedding calls.
        length_function: size measure for ``min_chunk_size``/``max_chunk_size`` (default ``len``).
        text_mode: ``"legacy"`` joins sentences with one space (old behaviour); ``"original"``
            returns exact slices of the source (newlines kept, exact ``start_index``, no empty
            chunks, ``min_chunk_size`` also applied to the last chunk). ``None`` picks
            ``"original"`` when a non-default ``sentence_split`` is set, else ``"legacy"``.
        sentence_split: ``"default"``, ``"cjk"`` or a callable ``text -> [(start, end), ...]``.
        embed_batch_size: call ``embed_documents`` with at most this many texts at a time.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        buffer_size: int = 1,
        add_start_index: bool = False,
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        number_of_chunks: Optional[int] = None,
        sentence_split_regex: str = DEFAULT_SENTENCE_REGEX,
        min_chunk_size: Optional[int] = None,
        *,
        max_chunk_size: Optional[int] = None,
        length_function: Callable[[str], int] = len,
        text_mode: Optional[TextMode] = None,
        sentence_split: SentenceSplit = "default",
        embed_batch_size: Optional[int] = None,
    ) -> None:
        if breakpoint_threshold_type not in BREAKPOINT_DEFAULTS:
            raise InvalidConfigError(
                f"Unknown breakpoint_threshold_type {breakpoint_threshold_type!r}. "
                f"Use one of: {', '.join(BREAKPOINT_DEFAULTS)}.",
                key="err.lib.threshold_type", value=breakpoint_threshold_type,
                choices=", ".join(BREAKPOINT_DEFAULTS),
            )
        if not isinstance(buffer_size, int) or isinstance(buffer_size, bool) or buffer_size < 0:
            raise InvalidConfigError(
                f"buffer_size must be an integer >= 0, got {buffer_size!r}.",
                key="err.lib.buffer_size", value=buffer_size,
            )
        for name, value in (
            ("number_of_chunks", number_of_chunks),
            ("min_chunk_size", min_chunk_size),
            ("max_chunk_size", max_chunk_size),
            ("embed_batch_size", embed_batch_size),
        ):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise InvalidConfigError(
                    f"{name} must be a positive integer or None, got {value!r}.",
                    key="err.lib.positive_int", name=name, value=value,
                )
        if min_chunk_size is not None and max_chunk_size is not None and max_chunk_size < min_chunk_size:
            raise InvalidConfigError(
                f"max_chunk_size ({max_chunk_size}) must not be smaller than "
                f"min_chunk_size ({min_chunk_size}).",
                key="err.lib.max_lt_min", max=max_chunk_size, min=min_chunk_size,
            )
        if not callable(length_function):
            raise InvalidConfigError("length_function must be a callable str -> int.", key="err.lib.length_function")
        if isinstance(sentence_split, str):
            if sentence_split not in PRESETS:
                raise InvalidConfigError(
                    f"Unknown sentence_split {sentence_split!r}. Use 'default', 'cjk' or a callable.",
                    key="err.lib.sentence_split_unknown", value=sentence_split,
                )
        elif not callable(sentence_split):
            raise InvalidConfigError(
                "sentence_split must be 'default', 'cjk' or a callable.", key="err.lib.sentence_split_type"
            )
        if text_mode not in (None, "legacy", "original"):
            raise InvalidConfigError(
                f"text_mode must be 'legacy' or 'original', got {text_mode!r}.",
                key="err.lib.text_mode", value=text_mode,
            )
        try:
            re.compile(sentence_split_regex)
        except re.error as exc:
            raise InvalidConfigError(
                f"sentence_split_regex is not a valid regex: {exc}.", key="err.lib.regex", reason=exc
            ) from exc

        self._add_start_index = add_start_index
        self.embeddings = embeddings
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.number_of_chunks = number_of_chunks
        self.sentence_split_regex = sentence_split_regex
        self.breakpoint_threshold_amount = (
            BREAKPOINT_DEFAULTS[breakpoint_threshold_type]
            if breakpoint_threshold_amount is None
            else breakpoint_threshold_amount
        )
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.length_function = length_function
        self.sentence_split = sentence_split
        self.embed_batch_size = embed_batch_size
        self.text_mode: TextMode = text_mode or ("original" if sentence_split != "default" else "legacy")

    # ---- sentences -------------------------------------------------------------------

    def _get_single_sentences_list(self, text: str) -> list[str]:
        """Legacy hook (kept for subclasses): split with ``sentence_split_regex``."""
        return re.split(self.sentence_split_regex, text)

    def _splitter(self) -> SentenceSplit:
        if self.sentence_split_regex != DEFAULT_SENTENCE_REGEX:
            return self.sentence_split_regex
        return self.sentence_split

    def _sentences(self, text: str) -> tuple[list[str], Optional[list[Span]]]:
        if self.text_mode == "original":
            spans = split_spans(text, self._splitter())
            return [text[s:e] for s, e in spans], spans
        if self.sentence_split == "default" or self.sentence_split_regex != DEFAULT_SENTENCE_REGEX:
            return self._get_single_sentences_list(text), None
        spans = split_spans(text, self.sentence_split)
        return [text[s:e] for s, e in spans], None

    # ---- embeddings and thresholds ----------------------------------------------------

    def _windows(self, sentences: Sequence[str]) -> list[str]:
        k = self.buffer_size
        out = []
        for i in range(len(sentences)):
            before = "".join(s + " " for s in sentences[max(0, i - k) : i])
            after = "".join(" " + s for s in sentences[i + 1 : i + 1 + k])
            out.append(before + sentences[i] + after)
        return out

    def _embed(self, windows: list[str]) -> tuple[list[Any], int]:
        vectors: list[Any] = []
        batches = batched(windows, self.embed_batch_size)
        for batch in batches:
            got = self.embeddings.embed_documents(batch)
            if len(got) != len(batch):
                raise InvalidConfigError(
                    f"embed_documents returned {len(got)} vectors for {len(batch)} texts.",
                    key="err.lib.embed_count", got=len(got), want=len(batch),
                )
            vectors.extend(got)
        return vectors, len(batches)

    def analyze(self, text: str, eps: float = 1e-9) -> Analysis:
        """Compute sentences, distances, threshold, cuts and fragile cuts for ``text``."""
        sentences, spans = self._sentences(text)
        n = len(sentences)
        info = Analysis(
            sentences=sentences,
            spans=spans,
            threshold_type=self.breakpoint_threshold_type,
            amount=self.breakpoint_threshold_amount,
        )
        # One sentence cannot be compared; gradient needs more than two sentences.
        if n <= 1 or (self.breakpoint_threshold_type == "gradient" and n == 2):
            return info
        vectors, info.embed_calls = self._embed(self._windows(sentences))
        distances = cosine_distances(vectors)
        if self.number_of_chunks is not None:
            threshold = threshold_from_chunk_count(distances, self.number_of_chunks)
            values: Any = distances
        else:
            threshold, values = breakpoint_values(
                distances, self.breakpoint_threshold_type, self.breakpoint_threshold_amount
            )
        info.distances = [float(d) for d in distances]
        info.values = [float(v) for v in values]
        info.threshold = threshold
        info.breakpoints = [i for i, v in enumerate(values) if v > threshold]
        info.fragile = fragile_indices(values, threshold, eps)
        return info

    # ---- chunk assembly ---------------------------------------------------------------

    def _group(self, info: Analysis, text: str, i0: int, i1: int) -> _Piece:
        if info.spans is not None:
            start, end = info.spans[i0][0], info.spans[i1][1]
            return _Piece(text[start:end], start)
        return _Piece(" ".join(info.sentences[i0 : i1 + 1]), None)

    def _groups(self, info: Analysis, text: str) -> list[tuple[int, int]]:
        n = len(info.sentences)
        if n == 0:
            return []
        if not info.distances:  # nothing to compare: sentences stand alone
            return [(i, i) for i in range(n)]
        groups: list[tuple[int, int]] = []
        start = 0
        for index in info.breakpoints:
            if self.min_chunk_size is not None:
                if self.length_function(self._group(info, text, start, index).text) < self.min_chunk_size:
                    continue  # too small: keep growing into the next piece
            groups.append((start, index))
            start = index + 1
        if start < n:
            groups.append((start, n - 1))
        if (
            self.text_mode == "original"
            and self.min_chunk_size is not None
            and len(groups) > 1
            and self.length_function(self._group(info, text, *groups[-1]).text) < self.min_chunk_size
        ):
            groups[-2:] = [(groups[-2][0], groups[-1][1])]  # fold the short tail into its neighbour
        return groups

    @staticmethod
    def _recut_position(info: Analysis, i0: int, i1: int) -> int:
        """Sentence index to cut after when the range ``i0..i1`` is longer than the limit.

        The cut goes where the distance to the next sentence is largest. When several
        positions share that largest distance (repeated boilerplate gives exact ties), take
        the one nearest the middle of the range by characters: both halves then shrink
        evenly. ``np.argmax`` alone would always pick the first tie and peel off one sentence
        per step, which makes long repetitive documents quadratic and leaves one-sentence
        chunks behind.
        """
        inside = np.asarray(info.distances[i0:i1], dtype=float)
        # 같은 최대 거리(부동소수 오차 1e-12 이내)를 가진 후보 위치들
        tied = np.flatnonzero(inside >= inside.max() - TIE_TOLERANCE)
        if len(tied) == 1:
            return i0 + int(tied[0])
        # 후보 k 에서 자르면 왼쪽 조각은 문장 i0..i0+k 까지 — 그 글자 수가 범위 절반에 가까운 후보를 고른다
        ends = np.cumsum([len(s) + 1 for s in info.sentences[i0 : i1 + 1]])
        half = ends[-1] / 2
        best = tied[int(np.argmin(np.abs(ends[tied] - half)))]
        return i0 + int(best)

    def _enforce_max(self, info: Analysis, text: str, groups: list[tuple[int, int]]) -> list[_Piece]:
        limit = self.max_chunk_size
        out: list[_Piece] = []
        stack = list(reversed(groups))
        while stack:
            i0, i1 = stack.pop()
            piece = self._group(info, text, i0, i1)
            if limit is None or self.length_function(piece.text) <= limit:
                out.append(piece)
            elif i1 > i0:
                cut = self._recut_position(info, i0, i1)  # biggest drift inside the range
                stack.append((cut + 1, i1))
                stack.append((i0, cut))
            else:  # a single sentence is too long on its own
                for s, e in split_long_text(piece.text, limit, self.length_function):
                    out.append(_Piece(piece.text[s:e], None if piece.start is None else piece.start + s))
        return out

    def _split(self, text: str) -> list[_Piece]:
        info = self.analyze(text)
        return self._enforce_max(info, text, self._groups(info, text))

    # ---- public API (same names as the original) ---------------------------------------

    def split_text(self, text: str) -> list[str]:
        """Split ``text`` into chunks."""
        return [p.text for p in self._split(text)]

    def create_documents(
        self, texts: list[str], metadatas: Optional[list[dict]] = None
    ) -> list[Document]:
        """Split every text and wrap the chunks in ``Document`` objects."""
        _metadatas = metadatas or [{}] * len(texts)
        documents: list[Document] = []
        for i, text in enumerate(texts):
            running = 0  # legacy start_index: running sum of chunk lengths
            for piece in self._split(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    metadata["start_index"] = running if piece.start is None else piece.start
                documents.append(Document(page_content=piece.text, metadata=metadata))
                running += len(piece.text)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """Split documents, copying their metadata onto every chunk."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        """``BaseDocumentTransformer`` entry point."""
        return self.split_documents(list(documents))
