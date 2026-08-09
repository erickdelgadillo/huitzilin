import numpy as np
import pytest

from huitzilin.embeddings import SentenceTransformerEmbedder
from huitzilin.models import Article


class FakeEncoder:
    """Small deterministic encoder used to test Huitzilin without ML downloads."""

    def __init__(self):
        self.last_sentences = None
        self.last_kwargs = None

    def encode(self, sentences, **kwargs):
        self.last_sentences = sentences
        self.last_kwargs = kwargs

        return np.array(
            [
                [float(index + 1), 0.5, 0.25]
                for index, _ in enumerate(sentences)
            ],
            dtype=np.float32,
        )


def test_encode_returns_float32_matrix():
    model = FakeEncoder()
    embedder = SentenceTransformerEmbedder(model=model)

    vectors = embedder.encode(["first", "second"])

    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32

    assert model.last_kwargs["convert_to_numpy"] is True
    assert model.last_kwargs["normalize_embeddings"] is True
    assert model.last_kwargs["show_progress_bar"] is False


def test_encode_query_returns_single_vector():
    embedder = SentenceTransformerEmbedder(model=FakeEncoder())

    vector = embedder.encode_query("ammonia oxidation")

    assert vector.shape == (3,)


def test_encode_articles_combines_title_and_abstract():
    model = FakeEncoder()
    embedder = SentenceTransformerEmbedder(model=model)

    article = Article(
        pubmed_id="123",
        title="Marine ammonia oxidation",
        abstract="Archaea contribute to nitrification.",
    )

    vectors = embedder.encode_articles([article])

    assert vectors.shape == (1, 3)

    assert model.last_sentences == [
        "Marine ammonia oxidation\n\n"
        "Archaea contribute to nitrification."
    ]


def test_empty_inputs_are_rejected():
    embedder = SentenceTransformerEmbedder(model=FakeEncoder())

    with pytest.raises(ValueError):
        embedder.encode([])

    with pytest.raises(ValueError):
        embedder.encode_query("   ")

    with pytest.raises(ValueError):
        embedder.encode_articles([])
