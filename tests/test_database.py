import sqlite3

from huitzilin.database import upsert_articles
from huitzilin.models import Article


def test_upsert_articles(tmp_path):
    database = tmp_path / "test.db"

    article = Article(
        pubmed_id="12345678",
        title="Test article",
        abstract="Test abstract",
        authors=("Doe Jane", "Smith John"),
        doi="10.1234/example",
        journal="Test Journal",
        year=2026,
        publication_types=("Journal Article",),
        mesh_terms=("Archaea",),
        keywords=("marine",),
    )

    upsert_articles([article], database)

    with sqlite3.connect(database) as connection:
        row = connection.execute(
            """
            SELECT pubmed_id, title, year, doi
            FROM articles
            WHERE pubmed_id = ?
            """,
            ("12345678",),
        ).fetchone()

    assert row == (
        "12345678",
        "Test article",
        2026,
        "10.1234/example",
    )
