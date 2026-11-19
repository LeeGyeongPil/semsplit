"""AC-7: embed_batch_size splits the embedding calls without changing the result."""

import pytest

from helpers import CountingEmbeddings, make_doc
from semsplit import HashEmbeddings, InvalidConfigError, SemanticChunker

DOC = make_doc(450)


def test_450_sentences_in_8_batches_of_at_most_64():
    emb = CountingEmbeddings()
    SemanticChunker(emb, embed_batch_size=64).split_text(DOC)
    assert emb.calls == 8
    assert max(emb.batches) <= 64
    assert emb.texts == 450


def test_one_call_without_batching():
    emb = CountingEmbeddings()
    SemanticChunker(emb).split_text(DOC)
    assert emb.batches == [450]


def test_result_is_identical_to_unbatched():
    assert SemanticChunker(HashEmbeddings(), embed_batch_size=64).split_text(DOC) == SemanticChunker(
        HashEmbeddings()
    ).split_text(DOC)


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_invalid_batch_size(bad):
    with pytest.raises(InvalidConfigError, match="embed_batch_size"):
        SemanticChunker(HashEmbeddings(), embed_batch_size=bad)


def test_wrong_number_of_vectors_is_reported():
    class Broken(HashEmbeddings):
        def embed_documents(self, texts):
            return super().embed_documents(texts)[:-1]

    with pytest.raises(InvalidConfigError, match="vectors"):
        SemanticChunker(Broken()).split_text(DOC)
