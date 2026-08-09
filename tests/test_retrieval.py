import pytest

from huitzilin.models import SearchHit
from huitzilin.retrieval import reciprocal_rank_fusion


def hit(pubmed_id: str, score: float, source: str) -> SearchHit:
    return SearchHit(
        pubmed_id=pubmed_id,
        title=f"Article {pubmed_id}",
        score=score,
        source=source,
    )


def test_rrf_rewards_articles_found_by_both_rankings():
    lexical = [
        hit("A", 20.0, "bm25"),
        hit("B", 10.0, "bm25"),
        hit("C", 5.0, "bm25"),
    ]

    dense = [
        hit("B", 0.9, "dense"),
        hit("A", 0.8, "dense"),
        hit("D", 0.7, "dense"),
    ]

    results = reciprocal_rank_fusion(
        [lexical, dense],
        top_k=4,
    )

    assert results[0].pubmed_id in {"A", "B"}
    assert results[1].pubmed_id in {"A", "B"}

    assert results[0].score == pytest.approx(
        results[1].score
    )

    assert all(
        result.source == "hybrid"
        for result in results
    )


def test_rrf_handles_results_unique_to_one_backend():
    lexical = [
        hit("A", 10.0, "bm25"),
    ]

    dense = [
        hit("B", 0.9, "dense"),
    ]

    results = reciprocal_rank_fusion(
        [lexical, dense],
        top_k=10,
    )

    assert {result.pubmed_id for result in results} == {
        "A",
        "B",
    }


def test_rrf_validates_arguments():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], top_k=0)

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], k=-1)
