"""Cosine distances, breakpoint thresholds and fragile-boundary detection.

The numeric path mirrors the original float64 NumPy implementation
(zero vectors count as similarity 0) so the default output stays identical.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from .errors import InvalidConfigError

BreakpointThresholdType = Literal["percentile", "standard_deviation", "interquartile", "gradient"]

BREAKPOINT_DEFAULTS: dict[str, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
    "gradient": 95,
}


def cosine_distances(vectors: Sequence[Sequence[float]]) -> list[float]:
    """Distance ``1 - cos`` between each vector and the next (length ``n - 1``)."""
    out: list[float] = []
    for i in range(len(vectors) - 1):
        x = np.array([vectors[i]])
        y = np.array([vectors[i + 1]])
        if x.shape[1] != y.shape[1]:
            raise InvalidConfigError("Embedding vectors have different lengths.", key="err.lib.vector_len")
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.dot(x, y.T) / np.outer(np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1))
        sim[np.isnan(sim) | np.isinf(sim)] = 0.0
        out.append(1 - sim[0][0])
    return out


def breakpoint_values(
    distances: list[float], kind: str, amount: float
) -> tuple[float, "list[float] | np.ndarray"]:
    """Return ``(threshold, values compared against it)`` for one threshold type."""
    if kind == "percentile":
        return float(np.percentile(distances, amount)), distances
    if kind == "standard_deviation":
        return float(np.mean(distances) + amount * np.std(distances)), distances
    if kind == "interquartile":
        q1, q3 = np.percentile(distances, [25, 75])
        return float(np.mean(distances) + amount * (q3 - q1)), distances
    if kind == "gradient":
        grad = np.gradient(distances, range(0, len(distances)))
        return float(np.percentile(grad, amount)), grad
    raise InvalidConfigError(
        f"Unknown breakpoint_threshold_type {kind!r}. "
        f"Use one of: {', '.join(BREAKPOINT_DEFAULTS)}.",
        key="err.lib.threshold_type", value=kind, choices=", ".join(BREAKPOINT_DEFAULTS),
    )


def threshold_from_chunk_count(distances: list[float], number_of_chunks: int) -> float:
    """Invert the percentile rule: pick the percentile that yields about ``number_of_chunks``."""
    x1, y1 = len(distances), 0.0
    x2, y2 = 1.0, 100.0
    x = max(min(number_of_chunks, x1), x2)
    y = y2 if x2 == x1 else y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)
    y = min(max(y, 0), 100)
    return float(np.percentile(distances, y))


def fragile_indices(values: "Sequence[float] | np.ndarray", threshold: float, eps: float) -> list[int]:
    """Indices whose value lies within ``eps`` of the threshold (cut decision could flip)."""
    return [i for i, v in enumerate(values) if abs(float(v) - threshold) <= eps]
