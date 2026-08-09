import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from .models import Article


SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    pubmed_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    authors TEXT NOT NULL,
    doi TEXT,
    journal TEXT,
    year INTEGER,
    publication_types TEXT NOT NULL,
    mesh_terms TEXT NOT NULL,
    keywords TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    source TEXT NOT NULL
);
"""


def initialize_database(database: str | Path) -> None:
    """Create the Huitzilin database and its core tables."""

    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database) as connection:
        connection.execute(SCHEMA)


def upsert_articles(
    articles: Iterable[Article],
    database: str | Path,
) -> None:
    """Insert new articles or update existing PubMed records."""

    database = Path(database)
    initialize_database(database)

    query = """
    INSERT INTO articles (
        pubmed_id,
        title,
        abstract,
        authors,
        doi,
        journal,
        year,
        publication_types,
        mesh_terms,
        keywords,
        retrieved_at,
        source
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

    ON CONFLICT(pubmed_id) DO UPDATE SET
        title = excluded.title,
        abstract = excluded.abstract,
        authors = excluded.authors,
        doi = excluded.doi,
        journal = excluded.journal,
        year = excluded.year,
        publication_types = excluded.publication_types,
        mesh_terms = excluded.mesh_terms,
        keywords = excluded.keywords,
        retrieved_at = excluded.retrieved_at,
        source = excluded.source;
    """

    rows = [
        (
            article.pubmed_id,
            article.title,
            article.abstract,
            json.dumps(article.authors, ensure_ascii=False),
            article.doi,
            article.journal,
            article.year,
            json.dumps(article.publication_types, ensure_ascii=False),
            json.dumps(article.mesh_terms, ensure_ascii=False),
            json.dumps(article.keywords, ensure_ascii=False),
            article.retrieved_at,
            article.source,
        )
        for article in articles
    ]

    with sqlite3.connect(database) as connection:
        connection.executemany(query, rows)
