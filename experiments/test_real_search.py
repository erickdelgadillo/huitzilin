
from pathlib import Path
from tempfile import TemporaryDirectory

from huitzilin.pubmed import PubMedClient
from huitzilin.database import upsert_articles
from huitzilin.embeddings import SentenceTransformerEmbedder
from huitzilin.vector_index import FaissVectorIndex
from huitzilin.retrieval import HybridRetriever

TARGET_PMID = "41835132"

QUERY = (
    "How do marine bacteria and phytoplankton "
    "respond to phosphorus limitation?"
)

PUBMED_QUERY = (
    '("phosphorus limitation" OR "phosphorus deficiency") '
    'AND (marine OR phytoplankton OR bacteria)'
)

print("1. Searching PubMed...")
client = PubMedClient()

ids = client.search(PUBMED_QUERY, max_results=100)

print(f"PubMed returned {len(ids)} identifiers.")
print(f"Target found naturally: {TARGET_PMID in ids}")

# Include the control explicitly if PubMed's first
# 100 results did not contain it.
# This guarantees inclusion, NOT successful discovery.
if TARGET_PMID not in ids:
    ids.append(TARGET_PMID)

print("2. Downloading articles...")
articles = client.fetch(ids)

articles = list({
    article.pubmed_id: article
    for article in articles
}.values())

assert any(
    article.pubmed_id == TARGET_PMID
    for article in articles
), "Positive control was not retrieved."

print(f"Downloaded {len(articles)} articles.")

with TemporaryDirectory() as temp:
    db = Path(temp) / "huitzilin_test.db"

    print("3. Creating SQLite database...")
    upsert_articles(articles, db)

    print("4. Generating embeddings...")
    embedder = SentenceTransformerEmbedder()
    vectors = embedder.encode_articles(articles)

    print("5. Building FAISS index...")
    index = FaissVectorIndex(
        dimension=vectors.shape[1]
    )
    index.add(articles, vectors)

    retriever = HybridRetriever(
        database=db,
        embedder=embedder,
        vector_index=index,
    )

    print("6. Comparing retrieval methods...")

    for mode in ("lexical", "dense", "hybrid"):
        results = retriever.search(
            QUERY,
            mode=mode,
            top_k=10,
            candidate_k=100,
        )

        print(f"\n{'=' * 60}")
        print(f"MODE: {mode.upper()}")
        print(f"{'=' * 60}")

        rank = None

        for position, hit in enumerate(results, 1):
            is_control = hit.pubmed_id == TARGET_PMID

            if is_control:
                rank = position

            marker = "  <-- POSITIVE CONTROL" if is_control else ""

            print(
                f"{position}. [{hit.pubmed_id}] "
                f"{hit.score:.5f}{marker}"
            )
            print(f"   {hit.title}")

        print(
            f"\nPositive control rank: "
            f"{rank if rank is not None else 'Outside top 10'}"
        )

print("\nExperiment completed.")
