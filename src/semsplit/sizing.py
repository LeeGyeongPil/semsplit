"""Helpers for ``max_chunk_size``: cut one over-long text into pieces that fit."""

from __future__ import annotations

from typing import Callable


def fit_prefix(text: str, pos: int, limit: int, length: Callable[[str], int]) -> int:
    """Largest ``k`` such that ``length(text[pos:pos + k]) <= limit`` (at least 1).

    Grows the window exponentially, then bisects, so only chunk-sized slices are measured.
    """
    n = len(text) - pos
    hi = 1
    while hi < n and length(text[pos : pos + hi]) <= limit:
        hi *= 2
    hi = min(hi, n)
    lo = 0  # length(text[pos:pos+lo]) <= limit is assumed to hold for lo = 0
    if length(text[pos : pos + hi]) <= limit:
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if length(text[pos : pos + mid]) <= limit:
            lo = mid
        else:
            hi = mid
    return max(lo, 1)


def split_long_text(text: str, limit: int, length: Callable[[str], int]) -> list[tuple[int, int]]:
    """Offsets ``(start, end)`` of pieces of one over-long text.

    Pieces end at the last whitespace that still fits, or at the character limit when a
    word alone is too long. Surrounding whitespace is trimmed, so every piece is a slice.
    """
    parts: list[tuple[int, int]] = []
    pos = 0
    total = len(text)
    while pos < total:
        while pos < total and text[pos].isspace():
            pos += 1
        if pos >= total:
            break
        k = fit_prefix(text, pos, limit, length)
        end = pos + k
        if end < total and not text[end].isspace():  # mid-word: back off to the last space
            window = text[pos:end]
            cut = max(window.rfind(" "), window.rfind("\n"), window.rfind("\t"))
            if cut > 0:
                end = pos + cut
        stop = end
        while stop > pos and text[stop - 1].isspace():
            stop -= 1
        if stop > pos:
            parts.append((pos, stop))
        pos = end
    return parts
