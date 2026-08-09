from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .models import Article


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Encoder(Protocol):
    """Minimal interface required from an embedding model."""

    def encode(self, sentences, **kwargs):
        ...


class SentenceTransformerEmbedder:
    """Dense embedding backend based on Sentence Transformers."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        model: Encoder | None = None,
    ) -> None:
        self.model_name = model_name
        self._model = model

    @property
    def model(self) -> Encoder:
        """Load the embedding model lazily."""

        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

        return self._model

    def encode(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """Encode texts as normalized dense vectors."""

        if not texts:
            raise ValueError("At least one text is required.")

        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        vectors = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        return vectors

    def encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        """Encode one search query."""

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        return self.encode([query])[0]

    def encode_articles(
        self,
        articles: Sequence[Article],
    ) -> np.ndarray:
        """Encode article title and abstract together."""

        if not articles:
            raise ValueError("At least one article is required.")

        texts = [
            f"{article.title}\n\n{article.abstract}".strip()
            for article in articles
        ]

        return self.encode(texts)
