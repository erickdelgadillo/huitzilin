#!/usr/bin/env python3
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight but effective model

# Connect to SQLite database
conn = sqlite3.connect("colibot.db")
cursor = conn.cursor()

# Retrieve all article abstracts
cursor.execute("SELECT id, abstract FROM articles")
articles = cursor.fetchall()

# Check if we have any articles
if not articles:
    print("No articles found in the database. Exiting...")
    conn.close()
    exit()

# Prepare data for FAISS indexing
article_ids = []
embeddings = []

print("Generating embeddings...")
for article_id, abstract in articles:
    if isinstance(abstract, list):
        abstract = " ".join(abstract)  # Convert list to string if needed
    embedding = model.encode(abstract)  # Generate embedding
    article_ids.append(article_id)
    embeddings.append(embedding)

# Convert to numpy array
embeddings = np.array(embeddings, dtype=np.float32)

# Initialize FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for similarity search
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "colibot_faiss.index")

# Save mapping of article IDs
with open("colibot_article_ids.npy", "wb") as f:
    np.save(f, np.array(article_ids))

print("FAISS index created and saved successfully!")

# Close database connection
conn.close()
