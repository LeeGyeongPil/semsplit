"""CLI language selection and message tables (English and Korean)."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib import resources
from typing import Callable, Mapping, Optional

SUPPORTED = ("en", "ko")
ENV_VAR = "SEMSPLIT_LANG"


def _normalise(value: Optional[str]) -> Optional[str]:
    """Map ``ko``, ``ko_KR.UTF-8``, ``en_US`` ... to a supported language, else None."""
    if not value:
        return None
    head = value.strip().lower().replace("-", "_").split(".")[0].split("_")[0]
    return head if head in SUPPORTED else None


def resolve_lang(cli_value: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> str:
    """Pick the screen language.

    Order: ``--lang`` value, then ``SEMSPLIT_LANG``, then the first non-empty of
    ``LC_ALL``, ``LC_MESSAGES``, ``LANG``. Anything that is not Korean ends up English.
    """
    environ = os.environ if env is None else env
    chosen = _normalise(cli_value)
    if chosen:
        return chosen
    chosen = _normalise(environ.get(ENV_VAR))
    if chosen:
        return chosen
    for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
        raw = environ.get(name)
        if raw:  # the first locale variable that is set decides
            return "ko" if raw.strip().lower().startswith("ko") else "en"
    return "en"


@lru_cache(maxsize=None)
def load_messages(lang: str) -> dict[str, str]:
    """Load the message table of one language from the packaged JSON file."""
    if lang not in SUPPORTED:
        raise ValueError(f"Unsupported language {lang!r}; use one of {SUPPORTED}.")
    text = resources.files("semsplit").joinpath("locales", f"{lang}.json").read_text(encoding="utf-8")
    return json.loads(text)


def translator(lang: str) -> Callable[..., str]:
    """Return ``t(key, **values)`` for ``lang`` (falls back to English for a missing key)."""
    table = load_messages(lang)
    fallback = load_messages("en")

    def t(key: str, **values: object) -> str:
        template = table.get(key, fallback.get(key, key))
        return template.format(**values) if values else template

    return t
