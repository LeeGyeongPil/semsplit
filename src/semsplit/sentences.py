"""Sentence splitting that reports offsets into the original text."""

from __future__ import annotations

import re
from typing import Callable, Union

from .errors import InvalidConfigError

DEFAULT_SENTENCE_REGEX = r"(?<=[.?!])\s+"

# Cut after full-width stops even without a space, keep closing quotes with the sentence,
# and after ASCII .?! only when whitespace follows (so "3.14" and "e.g.x" stay whole).
_CLOSERS = "」』）”’"
CJK_SENTENCE_REGEX = (
    rf"(?<=[。！？｡])(?![{_CLOSERS}])\s*"
    rf"|(?<=[。！？｡][{_CLOSERS}])\s*"
    r"|(?<=[.?!])\s+"
)

Span = tuple[int, int]
SentenceSplitter = Callable[[str], "list[Span]"]
SentenceSplit = Union[str, SentenceSplitter]

PRESETS = {"default": DEFAULT_SENTENCE_REGEX, "cjk": CJK_SENTENCE_REGEX}


def regex_spans(text: str, pattern: str) -> list[Span]:
    """Spans of the pieces between regex matches, trimmed, with blank pieces dropped."""
    matches = list(re.finditer(pattern, text))
    starts = [0] + [m.end() for m in matches]
    ends = [m.start() for m in matches] + [len(text)]
    spans: list[Span] = []
    for start, end in zip(starts, ends):
        span = _trim(text, start, end)
        if span is not None:
            spans.append(span)
    return spans


def _trim(text: str, start: int, end: int) -> "Span | None":
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def validate_spans(text: str, spans: "list[Span]") -> list[Span]:
    """Check callback output: ordered, non-overlapping ``(start, end)`` inside the text."""
    cleaned: list[Span] = []
    prev_end = 0
    for item in spans:
        try:
            start, end = item
        except (TypeError, ValueError):
            raise InvalidConfigError(
                f"sentence_split callback must return (start, end) pairs, got {item!r}.",
                key="err.lib.span_pair", item=item,
            ) from None
        if not (isinstance(start, int) and isinstance(end, int)) or not (prev_end <= start <= end <= len(text)):
            raise InvalidConfigError(
                f"sentence_split callback returned an invalid span {item!r}: spans must be "
                f"ordered, non-overlapping and inside the text (length {len(text)}).",
                key="err.lib.span_invalid", item=item, length=len(text),
            )
        prev_end = end
        trimmed = _trim(text, start, end)
        if trimmed is not None:
            cleaned.append(trimmed)
    return cleaned


def split_spans(text: str, how: SentenceSplit) -> list[Span]:
    """Split with a preset name (``default``/``cjk``), a regex string or a callback."""
    if callable(how):
        return validate_spans(text, list(how(text)))
    return regex_spans(text, PRESETS.get(how, how))
