import os
import sqlite3

import numpy as np
from Bio import Entrez
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# NCBI configuration
# ============================================================

NCBI_EMAIL = os.getenv("NCBI_EMAIL")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")

if not NCBI_EMAIL:
    raise RuntimeError(
        "NCBI_EMAIL is not defined. "
        "Set it with: export NCBI_EMAIL='your_email@example.com'"
    )

Entrez.email = NCBI_EMAIL

if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY


# ============================================================
# Query handling
# ============================================================

def read_query_file(query_file="query.txt"):
    """
    Read topic and question from query.txt.

    Accepted format:

        topic: marine archaea
        question: What articles discuss amoA?

    Also accepts:

        marine archaea
        What articles discuss amoA?
    """

    with open(query_file, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if len(lines) < 2:
        raise ValueError(
            "query.txt must contain at least two non-empty lines: "
            "a topic and a question."
        )

    topic = lines[0]
    question = lines[1]

    if topic.lower().startswith("topic:"):
        topic = topic.split(":", 1)[1].strip()

    if question.lower().startswith("question:"):
        question = question.split(":", 1)[1].strip()

    return topic, question


# ============================================================
# PubMed search
# ============================================================

def search_articles(query, max_results=1000):
    """
    Search PubMed and return a list of PubMed IDs.
    """

    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results
    )

    try:
        record = Entrez.read(handle)
    finally:
        handle.close()

    return record["IdList"]


# ============================================================
# PubMed article parsing
# ============================================================

def format_authors(authors):
    """
    Convert PubMed AuthorList into a readable string.
    """

    author_names = []

    for author in authors:

        if "CollectiveName" in author:
            author_names.append(str(author["CollectiveName"]))
            continue

        last_name = str(author.get("LastName", "")).strip()
        fore_name = str(author.get("ForeName", "")).strip()

        full_name = f"{last_name} {fore_name}".strip()

        if full_name:
            author_names.append(full_name)

    return ", ".join(author_names) if author_names else "Unknown"


def normalize_abstract(abstract):
    """
    PubMed abstracts can contain several labelled sections.
    Convert all possible forms into one single string.
    """

    if isinstance(abstract, (list, tuple)):
        return " ".join(str(section) for section in abstract)

    if abstract:
        return str(abstract)

    return "No abstract available"


def extract_doi(pubmed_article):
    """
    Extract the DOI specifically from PubMed ArticleIdList.
    """

    article_ids = (
        pubmed_article
        .get("PubmedData", {})
        .get("ArticleIdList", [])
    )

    for article_id in article_ids:

        attributes = getattr(article_id, "attributes", {})

        if attributes.get("IdType") == "doi":
            return str(article_id)

    return "No DOI available"


def fetch_article_details(article_ids):
    """
    Retrieve metadata and abstracts for PubMed IDs.
    """

    articles = []

    for pubmed_id in article_ids:

        handle = Entrez.efetch(
            db="pubmed",
            id=pubmed_id,
            rettype="xml",
            retmode="text"
        )

        try:
            records = Entrez.read(handle)["PubmedArticle"]
        finally:
            handle.close()

        if not records:
            print(f"⚠️ No record returned for PMID {pubmed_id}")
            continue

        pubmed_article = records[0]

        medline = pubmed_article["MedlineCitation"]
        article_data = medline["Article"]

        title = str(
            article_data.get(
                "ArticleTitle",
                "Unknown"
            )
        )

        authors = article_data.get(
            "AuthorList",
            []
        )

        author_names = format_authors(authors)

        abstract_raw = (
            article_data
            .get("Abstract", {})
            .get("AbstractText", [])
        )

        abstract = normalize_abstract(abstract_raw)

        doi = extract_doi(pubmed_article)

        articles.append(
            {
                "pubmed_id": str(pubmed_id),
                "title": title,
                "authors": author_names,
                "abstract": abstract,
                "doi": doi,
            }
        )

    return articles


# ============================================================
# SQLite storage
# ============================================================

def store_in_sqlite(articles, database="colibot.db"):
    """
    Store retrieved articles in SQLite.
    """

    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pubmed_id TEXT UNIQUE,
            title TEXT,
            authors TEXT,
            abstract TEXT,
            doi TEXT
        )
        """
    )

    for article in articles:

        cursor.execute(
            """
            INSERT INTO articles (
                pubmed_id,
                title,
                authors,
                abstract,
                doi
            )
            VALUES (?, ?, ?, ?, ?)

            ON CONFLICT(pubmed_id) DO UPDATE SET
                title = excluded.title,
                authors = excluded.authors,
                abstract = excluded.abstract,
                doi = excluded.doi
            """,
            (
                article["pubmed_id"],
                article["title"],
                article["authors"],
                article["abstract"],
                article["doi"],
            )
        )

    conn.commit()
    conn.close()


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Read search configuration
    # --------------------------------------------------------

    topic, question = read_query_file("query.txt")

    print(f"📥 Extracting articles based on query: {topic}")
    print(f"❓ Semantic question: {question}")

    # --------------------------------------------------------
    # Search PubMed
    # --------------------------------------------------------

    ids = search_articles(
        topic,
        max_results=10
    )

    print(f"✅ Found {len(ids)} article IDs.")

    if not ids:
        print("⚠️ PubMed returned no articles.")
        return

    # --------------------------------------------------------
    # Retrieve metadata and abstracts
    # --------------------------------------------------------

    articles = fetch_article_details(ids)

    print(f"✅ Retrieved {len(articles)} articles.")

    if not articles:
        print("⚠️ No valid articles were retrieved.")
        return

    # --------------------------------------------------------
    # Load semantic embedding model
    # --------------------------------------------------------

    print("🧠 Loading sentence-transformer model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # --------------------------------------------------------
    # Embed the scientific question
    # --------------------------------------------------------

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    # Shape should now be:
    #
    # (1, 384)

    # --------------------------------------------------------
    # Embed all abstracts IN A SINGLE BATCH
    # --------------------------------------------------------

    abstract_texts = [
        article["abstract"]
        for article in articles
    ]

    article_embeddings = model.encode(
        abstract_texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    # Shape should now be:
    #
    # (number_of_articles, 384)
    #
    # e.g.
    #
    # (10, 384)

    # --------------------------------------------------------
    # Semantic similarity
    # --------------------------------------------------------

    similarities = cosine_similarity(
        query_embedding,
        article_embeddings
    )[0]

    ranked_indices = np.argsort(
        similarities
    )[::-1]

    ranked_articles = [
        (
            articles[int(index)],
            float(similarities[index])
        )
        for index in ranked_indices
    ]

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print(f"🔍 Your question: {question}")
    print()
    print("✅ Search complete.")
    print("Top 5 relevant articles:")

    for rank, (article, score) in enumerate(
        ranked_articles[:5],
        start=1
    ):

        print()
        print("=" * 80)

        print(
            f"{rank}. {article['title']}"
        )

        print(
            f"Similarity: {score:.4f}"
        )

        print(
            f"PMID: {article['pubmed_id']}"
        )

        print(
            f"DOI: {article['doi']}"
        )

        print(
            f"Authors: {article['authors']}"
        )

        print()
        print(
            f"Abstract: {article['abstract'][:600]}..."
        )

    # --------------------------------------------------------
    # Save ranked articles
    # --------------------------------------------------------

    sorted_articles = [
        article
        for article, score in ranked_articles
    ]

    store_in_sqlite(
        sorted_articles
    )

    print()
    print("✅ Articles successfully saved to colibot.db.")


if __name__ == "__main__":
    main()