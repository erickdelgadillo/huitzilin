from huitzilin.database import upsert_articles
from huitzilin.lexical import search_lexical
from huitzilin.models import Article


def test_lexical_search_prioritizes_exact_term(tmp_path):
    database = tmp_path / "test.db"

    articles = [
        Article(
            pubmed_id="1",
            title="amoA diversity in marine archaea",
            abstract="Study of ammonia oxidizing archaea.",
        ),
        Article(
            pubmed_id="2",
            title="Marine microbial ecology",
            abstract="General study of microbial communities.",
        ),
        Article(
            pubmed_id="3",
            title="Nitrogen cycling in coastal waters",
            abstract="The amoA gene was used to study ammonia oxidation.",
        ),
    ]

    upsert_articles(articles, database)

    results = search_lexical(
        "amoA",
        database,
        top_k=10,
    )

    assert len(results) == 2
    assert results[0].pubmed_id == "1"
    assert all(result.source == "bm25" for result in results)
