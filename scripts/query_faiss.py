import sqlite3
import numpy as np
import faiss
import sys
from sentence_transformers import SentenceTransformer

# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and article IDs
index = faiss.read_index("colibot_faiss.index")
article_ids = np.load("colibot_article_ids.npy")

# Connect to SQLite database
conn = sqlite3.connect("colibot.db")
cursor = conn.cursor()

# Read the question from `query.txt`
with open("query.txt", "r") as file:
    for line in file:
        if line.startswith("question:"):
            user_query = line.split("question:")[1].strip()

print(f"🔍 Searching for articles related to: {user_query}")

# Function to search for similar articles
def search_articles(query, top_k=12):
    query_embedding = model.encode(query)
    query_embedding = np.expand_dims(query_embedding, axis=0)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        article_id = int(article_ids[idx])
        cursor.execute("SELECT title, authors, abstract FROM articles WHERE id = ?", (article_id,))
        result = cursor.fetchone()
        if result:
            title, authors, abstract = result
            results.append({"title": title, "authors": authors, "abstract": abstract})

    return results

# Main process
def main():
    results = search_articles(user_query)

    with open("../../../results.txt", "w") as f:
        f.write(f"🔍 Results for: {user_query}\n\n")
        for i, res in enumerate(results):
            f.write(f"{i+1}. 📰 {res['title']}\n")
            f.write(f"   ✍️  Authors: {res['authors']}\n")
            f.write(f"   📄 Abstract: {res['abstract'][:800]}...\n")
            f.write("--------------------------------------------------\n")

    print("\n✅ Results saved in 'results.txt'")

if __name__ == "__main__":
    main()

# Close database connection
conn.close()
