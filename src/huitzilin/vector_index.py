from collections.abc import Sequence

import faiss
import numpy as np

from .models import Article, SearchHit


class FaissVectorIndex:
    """Exact dense vector search using cosine similarity."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero.")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self._articles: list[Article] = []

    @property
    def size(self) -> int:
        """Number of vectors stored in the index."""

        return self.index.ntotal

    def _prepare_vectors(
        self,
        vectors: np.ndarray,
    ) -> np.ndarray:
        """Validate and L2-normalize vectors for cosine search."""

        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D matrix.")

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vectors with dimension {self.dimension}, "
                f"got {vectors.shape[1]}."
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Vectors contain non-finite values.")

        norms = np.linalg.norm(vectors, axis=1)

        if np.any(norms == 0):
            raise ValueError("Zero-length vectors cannot be indexed.")

        vectors = np.ascontiguousarray(vectors.copy())

        faiss.normalize_L2(vectors)

        return vectors

    def add(
        self,
        articles: Sequence[Article],
        vectors: np.ndarray,
    ) -> None:
        """Add article vectors to the FAISS index."""

        if not articles:
            raise ValueError("At least one article is required.")

        vectors = self._prepare_vectors(vectors)

        if len(articles) != vectors.shape[0]:
            raise ValueError(
                "Number of articles must match number of vectors."
            )

        existing_ids = {
            article.pubmed_id
            for article in self._articles
        }

        new_ids = [
            article.pubmed_id
            for article in articles
        ]

        if len(new_ids) != len(set(new_ids)):
            raise ValueError("Duplicate PubMed IDs in input.")

        if existing_ids.intersection(new_ids):
            raise ValueError(
                "One or more articles already exist in the index."
            )

        self.index.add(vectors)
        self._articles.extend(articles)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
    ) -> list[SearchHit]:
        """Search indexed articles using cosine similarity."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.size == 0:
            return []

        query = self._prepare_vectors(query_vector)

        if query.shape[0] != 1:
            raise ValueError("Search expects exactly one query vector.")

        effective_k = min(top_k, self.size)

        scores, indices = self.index.search(
            query,
            effective_k,
        )

        results: list[SearchHit] = []

        for score, index_position in zip(
            scores[0],
            indices[0],
        ):
            if index_position < 0:
                continue

            article = self._articles[index_position]

            results.append(
                SearchHit(
                    pubmed_id=article.pubmed_id,
                    title=article.title,
                    score=float(score),
                    source="dense",
                )
            )

        return results
