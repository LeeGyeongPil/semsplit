"""AC-1: default (compat) mode reproduces langchain-experimental 0.4.2 byte for byte."""

import json

import pytest

from helpers import ROOT
from semsplit import HashEmbeddings, SemanticChunker, check_parity

GOLDEN = ROOT / "tests" / "golden" / "semantic_chunker_0.4.2.json"
CASES = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]


def test_golden_case_count():
    assert len(CASES) >= 20
    docs = {c["id"].split("-")[0] for c in CASES}
    assert {"en", "ko", "ja"} <= docs  # three languages are covered


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_split_text_and_start_index_match_golden(case):
    chunker = SemanticChunker(HashEmbeddings(), add_start_index=True, **case["config"])
    assert chunker.split_text(case["text"]) == case["chunks"]
    docs = chunker.create_documents([case["text"]])
    assert [d.page_content for d in docs] == case["chunks"]
    assert [d.metadata["start_index"] for d in docs] == case["start_index"]


def test_golden_generator_is_recorded():
    meta = json.loads(GOLDEN.read_text(encoding="utf-8"))["generator"]
    assert meta["langchain-experimental"] == "0.4.2"


def test_check_parity_against_golden_file_is_equal():
    report = check_parity([], against=str(GOLDEN))
    assert report.all_equal
    assert len(report.cases) == len(CASES)
    assert all(isinstance(c.fragile, list) for c in report.cases)
