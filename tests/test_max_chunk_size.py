"""AC-4: opt-in hard size limit without extra embedding calls."""

import time

import numpy as np
import pytest

from helpers import CountingEmbeddings, data, make_doc
from semsplit import HashEmbeddings, InvalidConfigError, SemanticChunker

LONG = make_doc(450)


def test_long_document_is_really_long():
    assert len(LONG) >= 15_000


def test_chunks_respect_char_limit():
    baseline = SemanticChunker(HashEmbeddings()).split_text(LONG)
    assert max(map(len, baseline)) > 2000  # the limit below actually matters
    limited = SemanticChunker(HashEmbeddings(), max_chunk_size=2000).split_text(LONG)
    assert max(map(len, limited)) <= 2000
    # Re-cut only at sentence boundaries: joining gives back the same text.
    assert " ".join(limited) == " ".join(baseline)


def test_no_extra_embedding_calls():
    plain, capped = CountingEmbeddings(), CountingEmbeddings()
    SemanticChunker(plain).split_text(LONG)
    SemanticChunker(capped, max_chunk_size=500).split_text(LONG)
    assert capped.batches == plain.batches


def test_default_none_changes_nothing():
    a = SemanticChunker(HashEmbeddings()).split_text(LONG)
    b = SemanticChunker(HashEmbeddings(), max_chunk_size=None).split_text(LONG)
    assert a == b


def test_recut_happens_at_largest_distance_inside_the_chunk():
    text = data("three_topics.en.txt")
    chunker = SemanticChunker(HashEmbeddings())  # default 95: two chunks, the second is 215 chars
    first, second = chunker.split_text(text)
    limit = len(second) - 1
    info = chunker.analyze(text)
    start = info.breakpoints[-1] + 1 if info.breakpoints else 0
    n = len(info.sentences)
    expected_cut = start + int(np.argmax(info.distances[start : n - 1]))
    got = SemanticChunker(HashEmbeddings(), max_chunk_size=limit).split_text(text)
    expected_head = " ".join(info.sentences[start : expected_cut + 1])
    assert got[0] == first
    assert got[1] == expected_head
    assert " ".join(got[1:]) == second


def test_word_count_length_function():
    def words(s: str) -> int:
        return len(s.split())

    chunks = SemanticChunker(HashEmbeddings(), max_chunk_size=30, length_function=words).split_text(LONG)
    assert max(map(words, chunks)) <= 30
    assert len(chunks) > len(SemanticChunker(HashEmbeddings()).split_text(LONG))


def test_single_sentence_longer_than_limit_is_cut_at_spaces():
    long_sentence = " ".join(f"word{i}" for i in range(200)) + "."
    text = "Short one. " + long_sentence + " Another short one."
    chunks = SemanticChunker(HashEmbeddings(), max_chunk_size=100).split_text(text)
    assert max(map(len, chunks)) <= 100
    assert " ".join(chunks).split() == text.split()  # nothing lost, words intact


def test_unbroken_text_is_cut_at_the_character_limit():
    chunks = SemanticChunker(HashEmbeddings(), max_chunk_size=100).split_text("x" * 1000)
    assert [len(c) for c in chunks] == [100] * 10


def test_one_sentence_text_in_original_mode_keeps_offsets():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa."
    docs = SemanticChunker(
        HashEmbeddings(), max_chunk_size=20, text_mode="original", add_start_index=True
    ).create_documents([text])
    assert all(len(d.page_content) <= 20 for d in docs)
    for d in docs:
        s = d.metadata["start_index"]
        assert text[s : s + len(d.page_content)] == d.page_content


def test_limit_smaller_than_min_is_rejected():
    with pytest.raises(InvalidConfigError, match="max_chunk_size"):
        SemanticChunker(HashEmbeddings(), min_chunk_size=500, max_chunk_size=100)


@pytest.mark.parametrize("bad", [0, -5, 2.5, "10"])
def test_bad_limit_is_rejected(bad):
    with pytest.raises(InvalidConfigError, match="max_chunk_size"):
        SemanticChunker(HashEmbeddings(), max_chunk_size=bad)


# ---- repeated (boilerplate) text: tied distances -----------------------------------------

REPEATED = "Cats sleep. Stocks fell. Rain falls. " * 2000  # 6,000 sentences, 74,000 chars


def test_repeated_text_is_cut_into_full_size_pieces_not_single_sentences():
    # Every third distance ties for the largest. Taking the first tie each time used to peel
    # off one sentence per step: quadratic time and thousands of one-sentence chunks.
    chunker = SemanticChunker(HashEmbeddings(), breakpoint_threshold_amount=50, max_chunk_size=2000)
    started = time.perf_counter()
    chunks = chunker.split_text(REPEATED)
    elapsed = time.perf_counter() - started
    assert max(map(len, chunks)) <= 2000
    assert " ".join(chunks).split() == REPEATED.split()  # nothing lost or reordered
    sizes = sorted(map(len, chunks))
    assert sizes[len(sizes) // 2] >= 500  # the median chunk holds many sentences
    assert len(chunks) <= len(REPEATED) // 500
    assert elapsed < 30  # was quadratic; now well under a second on a laptop


def test_repeated_text_in_original_mode_keeps_slices():
    chunker = SemanticChunker(
        HashEmbeddings(), breakpoint_threshold_amount=50, max_chunk_size=1500,
        text_mode="original", add_start_index=True,
    )
    docs = chunker.create_documents([REPEATED])
    assert max(len(d.page_content) for d in docs) <= 1500
    assert sorted(len(d.page_content) for d in docs)[len(docs) // 2] >= 400
    for d in docs:
        s = d.metadata["start_index"]
        assert REPEATED[s : s + len(d.page_content)] == d.page_content


def test_tie_break_picks_the_tie_nearest_the_middle():
    from semsplit.analysis import Analysis

    info = Analysis(sentences=["aaaa"] * 9, spans=None, threshold_type="percentile", amount=95)
    info.distances = [0.5, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5, 0.1]  # ties at 0, 2, 4, 6
    assert SemanticChunker._recut_position(info, 0, 8) == 4  # middle of 9 equal sentences
    info.distances = [0.5, 0.1, 0.9, 0.1, 0.5, 0.1, 0.5, 0.1]  # a unique maximum still wins
    assert SemanticChunker._recut_position(info, 0, 8) == 2
