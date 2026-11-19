"""Boundaries, validation and helper modules."""

import subprocess
import sys

import numpy as np
import pytest
from langchain_core.documents import Document

from helpers import ROOT, CountingEmbeddings, data
from semsplit import (
    HashEmbeddings,
    InvalidConfigError,
    OllamaEmbeddings,
    ReplayEmbeddings,
    SemanticChunker,
    analyze,
)
from semsplit.embeddings import embedding_from_spec
from semsplit.sizing import split_long_text
from semsplit.thresholds import cosine_distances, threshold_from_chunk_count


def test_quickstart_runs_and_prints_chunks_line():
    out = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "quickstart.py")], capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    assert any(line.startswith("chunks: ") for line in out.stdout.splitlines())


def test_hash_embeddings_are_deterministic_and_normalised():
    emb = HashEmbeddings()
    a, b = emb.embed_documents(["Cats purr.", "Cats purr."])
    assert a == b and abs(np.linalg.norm(a) - 1) < 1e-9
    assert emb.embed_query("Cats purr.") == a
    assert emb.embed_documents(["猫は寝ます。"]) != emb.embed_documents(["株式市場"])
    assert emb.embed_documents([""])[0] == [0.0] * 256  # empty text is a zero vector, not an error


def test_zero_vectors_count_as_distance_one():
    assert cosine_distances([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]) == [1.0, 0.0]


def test_number_of_chunks_larger_than_sentences_is_clamped():
    text = data("three_topics.en.txt")
    chunks = SemanticChunker(HashEmbeddings(), number_of_chunks=500).split_text(text)
    assert 1 <= len(chunks) <= 9
    assert threshold_from_chunk_count([0.1, 0.2], 1) == pytest.approx(0.2)  # fewest chunks: top percentile


@pytest.mark.parametrize("kind", ["percentile", "standard_deviation", "interquartile", "gradient"])
def test_every_method_handles_tiny_inputs(kind):
    chunker = SemanticChunker(HashEmbeddings(), breakpoint_threshold_type=kind)
    for text in ["", "One.", "One. Two.", "One. Two. Three."]:
        assert isinstance(chunker.split_text(text), list)


def test_legacy_empty_string_keeps_old_result():
    assert SemanticChunker(HashEmbeddings()).split_text("") == [""]


def test_invalid_threshold_type_message_lists_choices():
    with pytest.raises(InvalidConfigError, match="percentile, standard_deviation, interquartile, gradient"):
        SemanticChunker(HashEmbeddings(), breakpoint_threshold_type="median")  # type: ignore[arg-type]
    assert issubclass(InvalidConfigError, ValueError)


@pytest.mark.parametrize("kw", [{"buffer_size": -1}, {"number_of_chunks": 0}, {"min_chunk_size": -3}, {"text_mode": "x"}, {"sentence_split_regex": "("}, {"length_function": 5}])
def test_bad_constructor_arguments(kw):
    with pytest.raises(InvalidConfigError):
        SemanticChunker(HashEmbeddings(), **kw)


def test_analyze_reports_threshold_cuts_and_fragile():
    info = analyze(data("three_topics.en.txt"), HashEmbeddings(), breakpoint_threshold_amount=50)
    assert len(info.sentences) == 9 and len(info.distances) == 8
    assert info.threshold is not None and info.chunk_count == len(info.breakpoints) + 1
    assert all(info.values[i] > info.threshold for i in info.breakpoints)
    wide = analyze(data("three_topics.en.txt"), HashEmbeddings(), breakpoint_threshold_amount=50, eps=10)
    assert wide.fragile == list(range(8))


def test_analyze_and_split_agree():
    text = data("ko.txt")
    chunker = SemanticChunker(HashEmbeddings(), breakpoint_threshold_amount=50)
    assert chunker.analyze(text).chunk_count == len(chunker.split_text(text))


def test_subclass_hook_get_single_sentences_list_is_honoured():
    class ByLine(SemanticChunker):
        def _get_single_sentences_list(self, text):
            return text.split("\n")

    out = ByLine(HashEmbeddings(), number_of_chunks=2).split_text("cats purr\ncats nap\nstocks fell\nstocks rose")
    assert len(out) == 2


def test_transform_and_split_documents_copy_metadata():
    chunker = SemanticChunker(HashEmbeddings(), add_start_index=True, breakpoint_threshold_amount=50)
    docs = [Document(page_content=data("three_topics.en.txt"), metadata={"id": 1}), Document(page_content=data("ko.txt"), metadata={"id": 2})]
    out = chunker.transform_documents(docs)
    assert {d.metadata["id"] for d in out} == {1, 2}
    assert all("start_index" in d.metadata for d in out)


def test_replay_embeddings_embed_each_text_once():
    inner = CountingEmbeddings()
    replay = ReplayEmbeddings(inner)
    SemanticChunker(replay).split_text(data("three_topics.en.txt"))
    first = inner.texts
    SemanticChunker(replay).split_text(data("three_topics.en.txt"))
    assert inner.texts == first  # second run served from the cache


def test_embedding_spec_parsing(monkeypatch):
    assert isinstance(embedding_from_spec("hash"), HashEmbeddings)
    emb = embedding_from_spec("ollama:nomic-embed-text", env={"OLLAMA_HOST": "http://example:1/"})
    assert isinstance(emb, OllamaEmbeddings) and emb.host == "http://example:1"
    with pytest.raises(InvalidConfigError, match="ollama:<model>"):
        embedding_from_spec("openai")
    with pytest.raises(InvalidConfigError):
        embedding_from_spec("ollama:")


def test_split_long_text_edge_cases():
    assert split_long_text("", 10, len) == []
    assert split_long_text("   ", 10, len) == []
    text = "aaaa bbbb cccc"
    parts = [text[s:e] for s, e in split_long_text(text, 9, len)]
    assert parts == ["aaaa bbbb", "cccc"]
    word = "abcdef"
    assert [word[s:e] for s, e in split_long_text(word, 4, len)] == ["abcd", "ef"]


def test_large_document_is_fast_enough():
    from helpers import make_doc

    chunks = SemanticChunker(HashEmbeddings(), max_chunk_size=1000, text_mode="original", embed_batch_size=100).split_text(make_doc(3000))
    assert max(map(len, chunks)) <= 1000
