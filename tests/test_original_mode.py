"""AC-5: text_mode="original" returns exact slices with exact offsets."""

import pytest

from helpers import data, make_doc
from semsplit import HashEmbeddings, SemanticChunker


def docs_for(text, **kw):
    kw.setdefault("breakpoint_threshold_amount", 50)
    chunker = SemanticChunker(HashEmbeddings(), text_mode="original", add_start_index=True, **kw)
    return chunker.create_documents([text])


def assert_slices(text, docs):
    for d in docs:
        s = d.metadata["start_index"]
        assert text[s : s + len(d.page_content)] == d.page_content
        assert d.page_content.strip() == d.page_content
        assert d.page_content != ""


@pytest.mark.parametrize("name", ["three_topics.en.txt", "multiline.en.txt", "ko.txt"])
def test_every_chunk_is_a_slice_of_the_source(name):
    text = data(name)
    docs = docs_for(text)
    assert len(docs) >= 2
    assert_slices(text, docs)


def test_newlines_are_preserved_inside_chunks():
    text = data("multiline.en.txt")
    docs = docs_for(text, breakpoint_threshold_amount=95)
    assert any("\n" in d.page_content for d in docs)
    assert_slices(text, docs)


def test_legacy_mode_start_index_drifts_but_original_does_not():
    text = data("multiline.en.txt")
    legacy = SemanticChunker(HashEmbeddings(), add_start_index=True, breakpoint_threshold_amount=50)
    drifted = [
        d for d in legacy.create_documents([text])
        if text[d.metadata["start_index"] : d.metadata["start_index"] + len(d.page_content)] != d.page_content
    ]
    assert drifted  # the documented quirk of the original
    assert_slices(text, docs_for(text))


def test_trailing_whitespace_and_empty_text_give_no_empty_chunks():
    docs = docs_for("One. Two. Three. ")
    assert [d.page_content for d in docs] and all(d.page_content for d in docs)
    assert docs_for("") == []
    assert docs_for("   \n ") == []


def test_last_chunk_obeys_min_chunk_size():
    text = data("three_topics.en.txt")
    docs = docs_for(text, min_chunk_size=200)
    assert len(text) > 200
    assert all(len(d.page_content) >= 200 for d in docs)
    assert_slices(text, docs)
    legacy_last = SemanticChunker(HashEmbeddings(), breakpoint_threshold_amount=50, min_chunk_size=200).split_text(text)
    assert len(legacy_last[-1]) < 200  # what the original does


def test_short_document_below_min_size_is_kept_whole():
    docs = docs_for("Tiny text. Very small.", min_chunk_size=500)
    assert len(docs) == 1


def test_start_indexes_increase_and_cover_long_document():
    text = make_doc(120)
    docs = docs_for(text, breakpoint_threshold_amount=90)
    starts = [d.metadata["start_index"] for d in docs]
    assert starts == sorted(starts) and starts[0] == 0
    assert_slices(text, docs)


def test_metadata_is_copied_per_chunk():
    chunker = SemanticChunker(HashEmbeddings(), text_mode="original", add_start_index=True, breakpoint_threshold_amount=50)
    from langchain_core.documents import Document

    out = chunker.split_documents([Document(page_content=data("three_topics.en.txt"), metadata={"src": "a"})])
    assert all(d.metadata["src"] == "a" for d in out)
    out[0].metadata["src"] = "changed"
    assert out[1].metadata["src"] == "a"
