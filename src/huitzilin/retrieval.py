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


class HybridRetriever:
    """Unified lexical, dense and hybrid scientific retrieval."""

    VALID_MODES = {"lexical", "dense", "hybrid"}

    def __init__(
        self,
        database,
        embedder=None,
        vector_index=None,
        rrf_k: int = 60,
    ):
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 0:
            raise ValueError("rrf_k must be a non-negative integer.")

        self.database = database
        self.embedder = embedder
        self.vector_index = vector_index
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        candidate_k: int = 100,
    ) -> list[SearchHit]:
        """Retrieve articles and return the selected ranking."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode: {mode}. "
                f"Choose from {sorted(self.VALID_MODES)}."
            )

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if (
            isinstance(candidate_k, bool)
            or not isinstance(candidate_k, int)
            or candidate_k <= 0
        ):
            raise ValueError("candidate_k must be a positive integer.")

        if mode == "hybrid" and candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k."
            )

        if mode in {"dense", "hybrid"}:
            if self.embedder is None or self.vector_index is None:
                raise RuntimeError(
                    "Dense retrieval requires both an embedder "
                    "and a vector index."
                )

        lexical_results = []
        dense_results = []

        if mode in {"lexical", "hybrid"}:
            from .lexical import search_lexical

            lexical_results = search_lexical(
                query,
                self.database,
                top_k=candidate_k if mode == "hybrid" else top_k,
            )

        if mode in {"dense", "hybrid"}:
            query_vector = self.embedder.encode_query(query)

            dense_results = self.vector_index.search(
                query_vector,
                top_k=candidate_k if mode == "hybrid" else top_k,
            )

        if mode == "lexical":
            return lexical_results

        if mode == "dense":
            return dense_results

        return reciprocal_rank_fusion(
            [lexical_results, dense_results],
            top_k=top_k,
            k=self.rrf_k,
        )
