from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class Article:
    """Canonical representation of a scientific article in Huitzilin."""

    pubmed_id: str
    title: str
    abstract: str

    authors: tuple[str, ...] = ()
    doi: str | None = None
    journal: str | None = None
    year: int | None = None

    publication_types: tuple[str, ...] = ()
    mesh_terms: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()

    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = "pubmed"
