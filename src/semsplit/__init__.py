"""semsplit: drop-in successor of langchain-experimental's SemanticChunker."""

from .analysis import Analysis, analyze
from .chunker import SemanticChunker
from .embeddings import HashEmbeddings, OllamaEmbeddings, ReplayEmbeddings
from .errors import (
    EmbeddingBackendError,
    InvalidConfigError,
    LegacyNotInstalledError,
    SemsplitError,
)
from .parity import ParityReport, check_parity, record_golden
from .thresholds import BreakpointThresholdType

__version__ = "0.1.0"

__all__ = [
    "Analysis",
    "BreakpointThresholdType",
    "EmbeddingBackendError",
    "HashEmbeddings",
    "InvalidConfigError",
    "LegacyNotInstalledError",
    "OllamaEmbeddings",
    "ParityReport",
    "ReplayEmbeddings",
    "SemanticChunker",
    "SemsplitError",
    "analyze",
    "check_parity",
    "record_golden",
    "__version__",
]
