"""Regenerate tests/golden/semantic_chunker_0.4.2.json from the archived original.

    uv run --extra legacy python scripts/make_golden.py

The file stores, for each case, the text, the constructor arguments and what
langchain-experimental 0.4.2 returned (chunks and start_index values) when it was fed
semsplit.HashEmbeddings. The golden test replays the cases against semsplit with no
legacy package installed. Run this only when HashEmbeddings itself changes on purpose.
"""

from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

from semsplit.embeddings import HashEmbeddings
from semsplit.parity import GOLDEN_FORMAT, _run, legacy_chunker_class

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "examples" / "data"
OUT = ROOT / "tests" / "golden" / "semantic_chunker_0.4.2.json"


def long_document(sentences: int = 60) -> str:
    """A deterministic multi-topic document built from templates."""
    topics = [
        ("harbor", ["The harbor opens at dawn.", "Fishing boats leave the harbor early.", "Gulls follow the boats out of the harbor."]),
        ("budget", ["The council debated the annual budget.", "Budget cuts reduced the library hours.", "Taxpayers asked how the budget is spent."]),
        ("garden", ["Tomatoes grow best in a sunny garden.", "Water the garden every morning.", "Compost keeps garden soil rich."]),
    ]
    out = []
    for i in range(sentences):
        out.append(topics[(i // 6) % 3][1][i % 3])
    return " ".join(out)


def cases() -> list[tuple[str, str, dict]]:
    en = (DATA / "three_topics.en.txt").read_text(encoding="utf-8")
    ko = (DATA / "ko.txt").read_text(encoding="utf-8")
    ja = (DATA / "ja.txt").read_text(encoding="utf-8")
    ml = (DATA / "multiline.en.txt").read_text(encoding="utf-8")
    methods = ["percentile", "standard_deviation", "interquartile", "gradient"]
    out: list[tuple[str, str, dict]] = []
    for name, text in (("en", en), ("ko", ko), ("ja", ja)):
        for m in methods:
            out.append((f"{name}-{m}", text, {"breakpoint_threshold_type": m}))
    out += [
        ("en-percentile-50", en, {"breakpoint_threshold_amount": 50}),
        ("ko-percentile-50", ko, {"breakpoint_threshold_amount": 50}),
        ("ja-custom-regex", ja, {"sentence_split_regex": "(?<=[。！？])", "breakpoint_threshold_amount": 50}),
        ("en-number-of-chunks-3", en, {"number_of_chunks": 3}),
        ("en-min-chunk-size-200", en, {"breakpoint_threshold_amount": 50, "min_chunk_size": 200}),
        ("en-buffer-0", en, {"buffer_size": 0, "breakpoint_threshold_amount": 50}),
        ("en-buffer-2", en, {"buffer_size": 2, "breakpoint_threshold_amount": 50}),
        ("en-std-1", en, {"breakpoint_threshold_type": "standard_deviation", "breakpoint_threshold_amount": 1}),
        ("en-iqr-0.5", en, {"breakpoint_threshold_type": "interquartile", "breakpoint_threshold_amount": 0.5}),
        ("trailing-space", "One. Two. Three. ", {"breakpoint_threshold_amount": 50}),
        ("empty-string", "", {}),
        ("one-sentence", "Just one sentence here.", {}),
        ("gradient-two-sentences", "First sentence. Second sentence.", {"breakpoint_threshold_type": "gradient"}),
        ("two-sentences-percentile", "Cats purr. Stocks fell.", {"breakpoint_threshold_amount": 50}),
        ("multiline-newlines", ml, {"breakpoint_threshold_amount": 50}),
        ("long-document", long_document(), {"breakpoint_threshold_amount": 80}),
        ("long-document-min-size", long_document(), {"breakpoint_threshold_amount": 80, "min_chunk_size": 300}),
    ]
    return out


def main() -> int:
    legacy = legacy_chunker_class()
    emb = HashEmbeddings()
    records = []
    for case_id, text, config in cases():
        chunks, starts = _run(legacy, emb, text, config)
        records.append({"id": case_id, "text": text, "config": config, "chunks": chunks, "start_index": starts})
    data = {
        "format": GOLDEN_FORMAT,
        "generator": {
            "source": "legacy",
            "langchain-experimental": version("langchain-experimental"),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "embeddings": "hash",
        "cases": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} cases to {OUT} (langchain-experimental {data['generator']['langchain-experimental']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
