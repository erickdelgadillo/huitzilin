"""Regression tests for search indexing and hybrid ranking."""
import pytest

from huitzilin.database import upsert_articles
from huitzilin.lexical import search_lexical
from huitzilin.models import Article, SearchHit
from huitzilin.retrieval import reciprocal_rank_fusion


def test_lexical_search_on_new_database_returns_no_matches(tmp_path):
    assert search_lexical("nitrification", tmp_path / "new.db") == []


def test_lexical_index_handles_upserts(tmp_path):
    db = tmp_path / "articles.db"
    upsert_articles([Article("1", "marine archaea", "nitrification")], db)
    assert [hit.pubmed_id for hit in search_lexical("archaea", db)] == ["1"]

    upsert_articles([Article("1", "marine bacteria", "denitrification")], db)
    assert search_lexical("archaea", db) == []
    assert [hit.pubmed_id for hit in search_lexical("bacteria", db)] == ["1"]


def test_lexical_query_rejects_punctuation_only(tmp_path):
    with pytest.raises(ValueError, match="no searchable terms"):
        search_lexical("!?", tmp_path / "papers.db")


def test_rrf_does_not_double_count_duplicate_ids():
    first = SearchHit("1", "Paper", 3.0, "bm25")
    repeated = SearchHit("1", "Paper", 2.0, "bm25")
    results = reciprocal_rank_fusion([[first, repeated]], k=60)
    assert len(results) == 1
    assert results[0].score == pytest.approx(1 / 61)
