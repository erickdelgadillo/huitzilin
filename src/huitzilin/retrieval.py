from collections.abc import Sequence

from .models import SearchHit


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]],
    top_k: int = 20,
    k: int = 60,
) -> list[SearchHit]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if k < 0:
        raise ValueError("k must be non-negative.")

    scores: dict[str, float] = {}
    hits: dict[str, SearchHit] = {}

    for ranking in rankings:
        seen: set[str] = set()

        for rank, hit in enumerate(ranking, start=1):
            if hit.pubmed_id in seen:
                continue

            seen.add(hit.pubmed_id)

            scores[hit.pubmed_id] = (
                scores.get(hit.pubmed_id, 0.0)
                + 1.0 / (k + rank)
            )

            hits.setdefault(hit.pubmed_id, hit)

    ordered_ids = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [
        SearchHit(
            pubmed_id=pubmed_id,
            title=hits[pubmed_id].title,
            score=scores[pubmed_id],
            source="hybrid",
        )
        for pubmed_id in ordered_ids[:top_k]
    ]
