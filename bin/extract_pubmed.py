#!/usr/bin/env python3
import sqlite3
from Bio import Entrez
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configurar Entrez con un correo electrónico válido
Entrez.email = None
Entrez.api_key = None

# Function to search articles in PubMed
def search_articles(query, max_results=1000):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

# Function to fetch article details
def fetch_article_details(article_ids):
    articles = []
    for pubmed_id in article_ids:
        handle = Entrez.efetch(db="pubmed", id=pubmed_id, rettype="xml", retmode="text")
        record = Entrez.read(handle)["PubmedArticle"]
        handle.close()
        
        title = record[0]["MedlineCitation"]["Article"].get("ArticleTitle", "Unknown")
        authors = record[0]["MedlineCitation"]["Article"].get("AuthorList", [])
        author_names = ", ".join([a["LastName"] + " " + a.get("ForeName", "") for a in authors]) if authors else "Unknown"
        abstract = record[0]["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", "No abstract available")
        doi = record[0].get("PubmedData", {}).get("ArticleIdList", ["No DOI available"])[0]
        
        articles.append({
            "pubmed_id": pubmed_id,
            "title": title,
            "authors": author_names,
            "abstract": abstract,
            "doi": doi
        })
    return articles

# Function to store articles in SQLite
def store_in_sqlite(articles):
    conn = sqlite3.connect("colibot.db")
    cursor = conn.cursor()
    
    # Create table if it does not exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pubmed_id TEXT UNIQUE,
            title TEXT,
            authors TEXT,
            abstract TEXT,
            doi TEXT
        )
    ''')
    
    for article in articles:
        print(f"Inserting: {article}")
        cursor.execute('''INSERT OR IGNORE INTO articles (pubmed_id, title, authors, abstract, doi)
                          VALUES (?, ?, ?, ?, ?)''',
                       (str(article["pubmed_id"]), 
                        str(article["title"]), 
                        str(article["authors"]), 
                        str(article["abstract"]), 
                        str(article["doi"])) )
    
    conn.commit()
    conn.close()

# Main process
def main():
    # Read the query from the 'query.txt' file
    with open("query.txt", "r") as file:
        lines = file.readlines()
        topic = lines[0].strip()  # First line: topic of the search
        question = lines[1].strip()  # Second line: the specific question

    print(f"📥 Extracting articles based on query: {topic}")
    
    # Step 1: Search for articles
    ids = search_articles(topic, max_results=10)
    print(f"✅ Found {len(ids)} article IDs.")

    # Step 2: Fetch full article details
    articles = fetch_article_details(ids)
    print(f"✅ Retrieved {len(articles)} articles.")

    # Step 3: Use Sentence-Transformers to compute similarity between the query and articles
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 3.1: Get the embedding for the query (after reading query.txt)
    query_embedding = model.encode(question)
    query_embedding = query_embedding.reshape(1, -1)  # Make sure query_embedding is 2D

    # Step 3.2: Get the embeddings for the article abstracts
    article_embeddings = [model.encode(article['abstract']).reshape(1, -1) for article in articles]  # Make sure article_embeddings are 2D

    # Step 3.3: Calculate cosine similarity between the query and each article
    similarities = [cosine_similarity(query_embedding, article_embedding)[0][0] for article_embedding in article_embeddings]

    # Step 3.4: Sort articles by similarity score (highest first)
    sorted_articles = [articles[i] for i in np.argsort(similarities)[::-1]]

    # Show the most relevant articles
    print(f"\n🔍 Your question: {question}")
    print("✅ Search complete. Top 5 relevant articles are:")

    for article in sorted_articles[:5]:  # Show the top 5 most relevant articles
        print(f"\nTitle: {article['title']}")
        print(f"Authors: {article['authors']}")
        print(f"Abstract: {article['abstract'][:600]}...")

    # Step 4: Save articles to SQLite
    store_in_sqlite(sorted_articles)
    print("✅ Articles successfully saved to colibot.db.")

if __name__ == "__main__":
    main()

