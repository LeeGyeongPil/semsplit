"""Exception types raised by semsplit."""

from __future__ import annotations


class SemsplitError(Exception):
    """Base class of every error semsplit raises on purpose.

    ``str(error)`` is always the English message. ``key`` and ``params`` let a front end
    (the CLI) look the same message up in another language; they are optional.
    """

    def __init__(self, message: str = "", *, key: "str | None" = None, **params: object) -> None:
        super().__init__(message)
        self.key = key
        self.params = params


class InvalidConfigError(SemsplitError, ValueError):
    """A constructor argument or sentence splitter result is not usable."""


class LegacyNotInstalledError(SemsplitError, ImportError):
    """The archived langchain-experimental package is needed but not installed."""

    def __init__(self) -> None:
        super().__init__(
            key="err.legacy_missing",
            message=
            "langchain-experimental 0.4.2 is not installed, so there is nothing to compare "
            "against. Install it with: pip install 'semsplit[legacy]' "
            "(or: uv run --extra legacy semsplit parity ...). "
            "Without it you can still compare against a recorded file: semsplit parity --golden FILE."
        )


class EmbeddingBackendError(SemsplitError, RuntimeError):
    """An embedding backend (for example a local Ollama server) could not be reached."""
