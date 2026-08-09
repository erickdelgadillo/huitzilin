# Huitzilin

**Hybrid scientific literature retrieval for PubMed.**

Huitzilin is a Python project for discovering and ranking scientific literature using a combination of lexical and semantic information retrieval.

It retrieves scientific articles from PubMed, stores structured metadata locally, and ranks papers using:

* **BM25 lexical retrieval** with SQLite FTS5
* **Dense semantic retrieval** with Sentence Transformers
* **FAISS vector search** using cosine similarity
* **Reciprocal Rank Fusion (RRF)** for hybrid ranking

The project is currently under active development.

> **Huitzilin** is the Nahuatl word for *hummingbird*. The name reflects the idea of moving quickly across the scientific literature to retrieve relevant knowledge.

---

## Why Huitzilin?

Scientific literature searches often require two complementary kinds of retrieval.

Lexical search is particularly strong when the query contains exact scientific terminology such as:

* gene names
* protein names
* taxa
* chemical compounds
* abbreviations
* specialized terminology

Semantic retrieval can recover conceptually related papers even when the wording differs from the original query.

Huitzilin combines both approaches rather than relying exclusively on keyword matching or vector similarity.

```text
                      HUITZILIN

                    scientific query
                           │
                           ▼
                     PubMed / NCBI
                           │
                           ▼
                    SQLite database
                  metadata + abstracts
                     │           │
                     ▼           ▼
                SQLite FTS5    Embeddings
                  BM25           Dense
                     │           │
                     │           ▼
                     │          FAISS
                     │           │
                     └─────┬─────┘
                           ▼
                 Reciprocal Rank Fusion
                           │
                           ▼
                    Hybrid ranking
```

---

## Current features

### PubMed retrieval

Huitzilin uses the NCBI Entrez API to:

* search PubMed
* retrieve records in batches
* parse titles and abstracts
* extract authors
* extract DOI
* extract journal and publication year
* extract publication types
* extract MeSH terms
* extract article keywords

Retrieved records are represented internally using a canonical `Article` model.

### Persistent scientific metadata

Article metadata is stored in SQLite.

Records are identified by PubMed ID and inserted using an UPSERT strategy, allowing existing records to be updated safely.

### Lexical retrieval

Huitzilin uses **SQLite FTS5** for full-text indexing of:

* title
* abstract
* MeSH terms
* keywords

Results are ranked with **BM25**, with additional weight assigned to article titles.

This retrieval layer is particularly useful for exact scientific identifiers and terminology.

### Semantic embeddings

Dense embeddings are currently generated using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding backend is configurable and designed so that alternative scientific embedding models can be evaluated later.

Article embeddings are generated from:

```text
title + abstract
```

Vectors are L2-normalized before similarity search.

### FAISS vector retrieval

Dense vectors are indexed with:

```text
faiss.IndexFlatIP
```

Because the vectors are normalized, inner product corresponds to cosine similarity.

The vector index validates:

* vector dimensionality
* finite values
* zero-length vectors
* duplicate PubMed IDs
* requested result counts

### Hybrid retrieval

Lexical and dense rankings are combined using **Reciprocal Rank Fusion (RRF)**.

This avoids directly combining BM25 and cosine scores, which exist on different numerical scales.

Conceptually:

```text
BM25 ranking ──────┐
                   │
                   ├──► RRF ───► Hybrid ranking
                   │
Dense ranking ─────┘
```

---

## Installation

Huitzilin requires **Python 3.10 or newer**.

Clone the repository:

```bash
git clone https://github.com/erickdelgadillo/huitzilin.git
cd huitzilin
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the core package and development dependencies:

```bash
python -m pip install -e '.[dev]'
```

For semantic retrieval with Sentence Transformers and FAISS:

```bash
python -m pip install -e '.[dev,semantic]'
```

---

## NCBI configuration

PubMed access requires an email address.

Set it as an environment variable:

```bash
export NCBI_EMAIL="your_email@example.com"
```

An NCBI API key can optionally be supplied:

```bash
export NCBI_API_KEY="your_api_key"
```

Credentials are read from environment variables and should **never be committed to the repository**.

---

## Example: retrieve articles from PubMed

```python
from huitzilin.pubmed import PubMedClient
from huitzilin.database import upsert_articles

client = PubMedClient()

ids = client.search(
    "marine archaea",
    max_results=100,
)

articles = client.fetch(ids)

upsert_articles(
    articles,
    "data/huitzilin.db",
)
```

---

## Example: lexical search

```python
from huitzilin.lexical import search_lexical

results = search_lexical(
    "ammonia oxidation",
    "data/huitzilin.db",
    top_k=10,
)

for result in results:
    print(
        result.score,
        result.pubmed_id,
        result.title,
    )
```

---

## Example: dense retrieval

```python
from huitzilin.embeddings import SentenceTransformerEmbedder
from huitzilin.vector_index import FaissVectorIndex

embedder = SentenceTransformerEmbedder()

article_vectors = embedder.encode_articles(articles)

index = FaissVectorIndex(
    dimension=article_vectors.shape[1]
)

index.add(
    articles,
    article_vectors,
)

query_vector = embedder.encode_query(
    "microorganisms using ammonia oxidation "
    "to support carbon fixation"
)

results = index.search(
    query_vector,
    top_k=10,
)
```

---

## Example: hybrid ranking

```python
from huitzilin.retrieval import reciprocal_rank_fusion

hybrid_results = reciprocal_rank_fusion(
    [
        lexical_results,
        dense_results,
    ],
    top_k=10,
)
```

RRF rewards papers that rank highly across multiple independent retrieval systems without requiring their raw scores to be directly comparable.

---

## Testing

Run the complete test suite with:

```bash
pytest
```

The current tests cover:

* SQLite persistence
* BM25 lexical retrieval
* embedding interfaces
* query and article embeddings
* FAISS vector search
* vector validation
* result-count safety
* Reciprocal Rank Fusion

---

## Project structure

```text
src/huitzilin/
├── __init__.py
├── cli.py
├── database.py
├── embeddings.py
├── export.py
├── lexical.py
├── models.py
├── pubmed.py
├── query_builder.py
├── reranker.py
├── retrieval.py
└── vector_index.py

tests/
├── test_database.py
├── test_embeddings.py
├── test_lexical.py
├── test_retrieval.py
└── test_vector_index.py
```

Some modules are currently placeholders for functionality planned in later development stages.

---

## Roadmap

### Retrieval engine

* [x] PubMed search and retrieval
* [x] SQLite persistence
* [x] FTS5 full-text index
* [x] BM25 ranking
* [x] Sentence Transformer embeddings
* [x] FAISS vector search
* [x] Reciprocal Rank Fusion
* [ ] Unified `HybridRetriever`
* [ ] Persistent vector index
* [ ] Query expansion and scientific query builder

### Scientific ranking

* [ ] Evaluate scientific embedding models
* [ ] Compare MiniLM with SPECTER2 or equivalent scientific models
* [ ] Cross-encoder reranking
* [ ] Query-aware scientific metadata weighting

### Evaluation

* [ ] Curated relevance dataset
* [ ] Precision@K
* [ ] Recall@K
* [ ] Mean Reciprocal Rank
* [ ] NDCG
* [ ] Comparative BM25 / dense / hybrid evaluation

### Interface

* [ ] Command-line interface
* [ ] Search result export
* [ ] Query history
* [ ] Database statistics
* [ ] Optional API interface

### Future directions

* [ ] PubMed Central open-access full text
* [ ] Europe PMC integration
* [ ] Structured scientific information extraction
* [ ] Optional generative layer over retrieved evidence

---

## Design principles

Huitzilin is being developed around several principles:

**Reproducibility**
Retrieval and ranking should be inspectable and repeatable.

**Modularity**
PubMed retrieval, persistence, lexical search, embeddings, vector search, fusion and reranking are separate components.

**Scientific terminology matters**
Exact lexical signals should not be discarded simply because semantic models are available.

**Retrieval before generation**
The project focuses first on building a measurable information-retrieval system rather than adding a generative model prematurely.

**Evaluation over intuition**
Future model and ranking decisions will be compared using explicit relevance judgments and information-retrieval metrics.

---

## Development status

Huitzilin is an experimental scientific software project under active development.

The current implementation provides the core building blocks for hybrid literature retrieval. The next development stage will integrate these components behind a unified retrieval interface and expand evaluation to substantially larger PubMed candidate sets.
