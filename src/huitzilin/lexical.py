import re
import sqlite3
from pathlib import Path

from .database import initialize_database
from .models import SearchHit


FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts
USING fts5(
    title,
    abstract,
    mesh_terms,
    keywords,
    content='articles',
    content_rowid='rowid',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS articles_fts_insert
AFTER INSERT ON articles
BEGIN
    INSERT INTO articles_fts(
        rowid,
        title,
        abstract,
        mesh_terms,
        keywords
    )
    VALUES (
        new.rowid,
        new.title,
        new.abstract,
        new.mesh_terms,
        new.keywords
    );
END;

CREATE TRIGGER IF NOT EXISTS articles_fts_delete
AFTER DELETE ON articles
BEGIN
    INSERT INTO articles_fts(
        articles_fts,
        rowid,
        title,
        abstract,
        mesh_terms,
        keywords
    )
    VALUES (
        'delete',
        old.rowid,
        old.title,
        old.abstract,
        old.mesh_terms,
        old.keywords
    );
END;

CREATE TRIGGER IF NOT EXISTS articles_fts_update
AFTER UPDATE ON articles
BEGIN
    INSERT INTO articles_fts(
        articles_fts,
        rowid,
        title,
        abstract,
        mesh_terms,
        keywords
    )
    VALUES (
        'delete',
        old.rowid,
        old.title,
        old.abstract,
        old.mesh_terms,
        old.keywords
    );

    INSERT INTO articles_fts(
        rowid,
        title,
        abstract,
        mesh_terms,
        keywords
    )
    VALUES (
        new.rowid,
        new.title,
        new.abstract,
        new.mesh_terms,
        new.keywords
    );
END;
"""


def initialize_lexical_index(database: str | Path) -> None:
    """Create the FTS5 index and synchronize existing articles."""

    database = Path(database)
    initialize_database(database)

    with sqlite3.connect(database) as connection:
        existed = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'articles_fts'
            """
        ).fetchone()

        connection.executescript(FTS_SCHEMA)

        if not existed:
            connection.execute(
                "INSERT INTO articles_fts(articles_fts) VALUES ('rebuild')"
            )


def _prepare_query(query: str) -> str:
    """Convert free text into a safe broad FTS5 query."""

    tokens = re.findall(r"\w+", query, flags=re.UNICODE)

    if not tokens:
        raise ValueError("Search query contains no searchable terms.")

    return " OR ".join(f'"{token}"' for token in tokens)


def search_lexical(
    query: str,
    database: str | Path,
    top_k: int = 20,
) -> list[SearchHit]:
    """Rank articles using SQLite FTS5 BM25."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    database = Path(database)
    initialize_lexical_index(database)

    fts_query = _prepare_query(query)

    sql = """
    SELECT
        articles.pubmed_id,
        articles.title,
        bm25(
            articles_fts,
            6.0,
            2.0,
            1.5,
            1.5
        ) AS bm25_score
    FROM articles_fts
    JOIN articles
      ON articles.rowid = articles_fts.rowid
    WHERE articles_fts MATCH ?
    ORDER BY bm25_score
    LIMIT ?
    """

    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            sql,
            (fts_query, top_k),
        ).fetchall()

    return [
        SearchHit(
            pubmed_id=pubmed_id,
            title=title,
            score=-float(score),
            source="bm25",
        )
        for pubmed_id, title, score in rows
    ]
