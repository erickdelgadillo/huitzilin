
import pytest

from huitzilin.database import upsert_articles
from huitzilin.models import Article, SearchHit
from huitzilin.retrieval import HybridRetriever


class FakeEmbedder:
    def encode_query(self, query):
        return [1.0, 0.0]


class FakeVectorIndex:
    def search(self, query_vector, top_k=20):
        return [
            SearchHit(
                pubmed_id="2",
                title="Marine microbial interactions",
                score=0.95,
                source="dense",
            ),
            SearchHit(
                pubmed_id="3",
                title="Ocean microbial communities",
                score=0.80,
                source="dense",
            ),
        ][:top_k]


@pytest.fixture
def database(tmp_path):
    db = tmp_path / "test.db"

    upsert_articles(
        [
            Article(
                pubmed_id="1",
                title="Marine archaea and ammonia oxidation",
                abstract="Marine archaea oxidize ammonia.",
            ),
            Article(
                pubmed_id="2",
                title="Marine microbial interactions",
                abstract="Interactions among marine microorganisms.",
            ),
        ],
        db,
    )

    return db


def test_lexical_mode(database):
    retriever = HybridRetriever(database)

    results = retriever.search(
        "ammonia",
        mode="lexical",
    )

    assert len(results) == 1
    assert results[0].pubmed_id == "1"
    assert results[0].source == "bm25"


def test_dense_mode(database):
    retriever = HybridRetriever(
        database,
        embedder=FakeEmbedder(),
        vector_index=FakeVectorIndex(),
    )

    results = retriever.search(
        "microbial interactions",
        mode="dense",
    )

    assert results[0].pubmed_id == "2"
    assert results[0].source == "dense"


def test_hybrid_mode(database):
    retriever = HybridRetriever(
        database,
        embedder=FakeEmbedder(),
        vector_index=FakeVectorIndex(),
    )

    results = retriever.search(
        "marine microbial interactions",
        mode="hybrid",
    )

    assert results
    assert all(hit.source == "hybrid" for hit in results)
    assert len({hit.pubmed_id for hit in results}) == len(results)


def test_missing_dense_dependencies(database):
    retriever = HybridRetriever(database)

    with pytest.raises(RuntimeError):
        retriever.search("marine archaea", mode="hybrid")


@pytest.mark.parametrize(
    "query,mode,top_k",
    [
        ("", "lexical", 10),
        ("marine", "invalid", 10),
        ("marine", "lexical", 0),
        ("marine", "lexical", -1),
        ("marine", "lexical", True),
    ],
)
def test_invalid_inputs(database, query, mode, top_k):
    retriever = HybridRetriever(database)

    with pytest.raises(ValueError):
        retriever.search(query, mode=mode, top_k=top_k)


def test_empty_results(database):
    retriever = HybridRetriever(database)

    results = retriever.search(
        "xyznonexistentterm",
        mode="lexical",
    )

    assert results == []


def test_result_limit(database):
    retriever = HybridRetriever(
        database,
        embedder=FakeEmbedder(),
        vector_index=FakeVectorIndex(),
    )

    results = retriever.search(
        "marine",
        mode="hybrid",
        top_k=1,
    )

    assert len(results) == 1


def test_candidate_k_smaller_than_top_k(database):
    retriever = HybridRetriever(
        database,
        embedder=FakeEmbedder(),
        vector_index=FakeVectorIndex(),
    )

    with pytest.raises(ValueError, match="candidate_k"):
        retriever.search(
            "marine",
            mode="hybrid",
            top_k=20,
            candidate_k=10,
        )


@pytest.mark.parametrize("candidate_k", [0, -1, True, 1.5])
def test_invalid_candidate_k(database, candidate_k):
    retriever = HybridRetriever(database)

    with pytest.raises(ValueError, match="candidate_k"):
        retriever.search(
            "marine",
            mode="lexical",
            candidate_k=candidate_k,
        )


def test_both_engines_empty(tmp_path):
    class EmptyIndex:
        def search(self, query_vector, top_k=20):
            return []

    db = tmp_path / "empty.db"

    retriever = HybridRetriever(
        db,
        embedder=FakeEmbedder(),
        vector_index=EmptyIndex(),
    )

    results = retriever.search(
        "marine archaea",
        mode="hybrid",
    )

    assert results == []


def test_dense_empty_lexical_has_results(database):
    class EmptyIndex:
        def search(self, query_vector, top_k=20):
            return []

    retriever = HybridRetriever(
        database,
        embedder=FakeEmbedder(),
        vector_index=EmptyIndex(),
    )

    results = retriever.search(
        "ammonia",
        mode="hybrid",
    )

    assert len(results) == 1
    assert results[0].pubmed_id == "1"


def test_lexical_empty_dense_has_results(database):
    retriever = HybridRetriever(
        database,
        embedder=FakeEmbedder(),
        vector_index=FakeVectorIndex(),
    )

    results = retriever.search(
        "xyznonexistentterm",
        mode="hybrid",
    )

    assert [hit.pubmed_id for hit in results] == ["2", "3"]


def test_hybrid_order_is_reproducible(database):
    retriever = HybridRetriever(
        database,
        embedder=FakeEmbedder(),
        vector_index=FakeVectorIndex(),
    )

    first = retriever.search(
        "marine microbial interactions",
        mode="hybrid",
    )

    second = retriever.search(
        "marine microbial interactions",
        mode="hybrid",
    )

    assert first == second


def test_lexical_does_not_use_embedder(database):
    class FailingEmbedder:
        def encode_query(self, query):
            raise AssertionError(
                "The embedder must not be called in lexical mode."
            )

    retriever = HybridRetriever(
        database,
        embedder=FailingEmbedder(),
    )

    results = retriever.search(
        "ammonia",
        mode="lexical",
    )

    assert results[0].pubmed_id == "1"


def test_candidate_k_forwarded_to_dense_index(database):
    class RecordingIndex:
        def __init__(self):
            self.requested_k = None

        def search(self, query_vector, top_k=20):
            self.requested_k = top_k
            return []

    index = RecordingIndex()

    retriever = HybridRetriever(
        database,
        embedder=FakeEmbedder(),
        vector_index=index,
    )

    retriever.search(
        "marine",
        mode="hybrid",
        top_k=5,
        candidate_k=42,
    )

    assert index.requested_k == 42
