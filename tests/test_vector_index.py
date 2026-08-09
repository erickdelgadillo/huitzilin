import numpy as np
import pytest

from huitzilin.models import Article
from huitzilin.vector_index import FaissVectorIndex


def make_articles():
    return [
        Article(
            pubmed_id="1",
            title="Marine ammonia oxidation",
            abstract="Archaeal nitrification.",
        ),
        Article(
            pubmed_id="2",
            title="Nitrogen cycling",
            abstract="Microbial nitrogen transformations.",
        ),
        Article(
            pubmed_id="3",
            title="Terrestrial photosynthesis",
            abstract="Plant carbon fixation.",
        ),
    ]


def test_vector_search_returns_best_match():
    articles = make_articles()

    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    index = FaissVectorIndex(dimension=3)
    index.add(articles, vectors)

    results = index.search(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=3,
    )

    assert len(results) == 3
    assert results[0].pubmed_id == "1"
    assert results[0].score == pytest.approx(1.0)
    assert all(result.source == "dense" for result in results)


def test_top_k_is_limited_to_index_size():
    articles = make_articles()

    vectors = np.eye(3, dtype=np.float32)

    index = FaissVectorIndex(dimension=3)
    index.add(articles, vectors)

    results = index.search(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=100,
    )

    assert len(results) == 3
    assert len({result.pubmed_id for result in results}) == 3


def test_vector_dimension_is_validated():
    index = FaissVectorIndex(dimension=3)

    article = Article(
        pubmed_id="1",
        title="Test",
        abstract="Test",
    )

    with pytest.raises(ValueError):
        index.add(
            [article],
            np.array([[1.0, 2.0]], dtype=np.float32),
        )
