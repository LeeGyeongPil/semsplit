"""AC-6: CJK sentence splitting and custom splitters."""

import pytest

from semsplit import HashEmbeddings, InvalidConfigError, SemanticChunker
from semsplit.sentences import regex_spans, split_spans, CJK_SENTENCE_REGEX

JA = "猫は小さな動物です。猫は暖かい場所で寝ます。今日、株式市場が下落しました。"


def test_default_keeps_old_behaviour_one_chunk():
    chunker = SemanticChunker(HashEmbeddings())
    assert chunker.split_text(JA) == [JA]


def test_cjk_splits_japanese_without_inventing_spaces():
    chunker = SemanticChunker(HashEmbeddings(), sentence_split="cjk", breakpoint_threshold_amount=50)
    chunks = chunker.split_text(JA)
    assert len(chunks) >= 2
    assert all(chunks)
    assert all(" " not in c for c in chunks)
    assert "".join(chunks) == JA


def test_cjk_spans_cover_each_sentence():
    spans = regex_spans(JA, CJK_SENTENCE_REGEX)
    assert [JA[s:e] for s, e in spans] == ["猫は小さな動物です。", "猫は暖かい場所で寝ます。", "今日、株式市場が下落しました。"]


def test_cjk_keeps_closing_quotes_and_handles_mixed_text():
    text = "彼は言った。「行こう。」そして走った。Hello world. Next one! 完了？はい。"
    pieces = [text[s:e] for s, e in split_spans(text, "cjk")]
    assert "「行こう。」" in "".join(pieces)
    assert not any(p.startswith("」") for p in pieces)
    assert "Hello world." in pieces
    pi = "Pi is 3.14 ok. Next."
    assert [pi[s:e] for s, e in split_spans(pi, "cjk")] == ["Pi is 3.14 ok.", "Next."]


def test_callable_splitter_is_used():
    text = "cats purr softly\ncats nap daily\nstocks fell today\nstocks lost value"

    def by_line(t: str):
        spans, pos = [], 0
        for line in t.split("\n"):
            spans.append((pos, pos + len(line)))
            pos += len(line) + 1
        return spans

    chunker = SemanticChunker(HashEmbeddings(), sentence_split=by_line, number_of_chunks=2)
    assert chunker.analyze(text).sentences == ["cats purr softly", "cats nap daily", "stocks fell today", "stocks lost value"]
    chunks = chunker.split_text(text)
    assert len(chunks) == 2
    assert "\n".join(chunks) == text  # slices of the source, newlines kept


@pytest.mark.parametrize("bad", [[(0, 99)], [(5, 2)], [(0, 3), (2, 6)], ["x"], [(0.5, 2)]])
def test_bad_callback_output_gives_clear_error(bad):
    chunker = SemanticChunker(HashEmbeddings(), sentence_split=lambda t: bad)
    with pytest.raises(InvalidConfigError, match="sentence_split callback"):
        chunker.split_text("short text. more.")


def test_unknown_preset_is_rejected():
    with pytest.raises(InvalidConfigError, match="sentence_split"):
        SemanticChunker(HashEmbeddings(), sentence_split="klingon")


def test_explicit_regex_wins_over_preset():
    chunker = SemanticChunker(HashEmbeddings(), sentence_split="cjk", sentence_split_regex=r"(?<=;)\s*")
    assert chunker.analyze("a one; b two; c three").sentences == ["a one;", "b two;", "c three"]
