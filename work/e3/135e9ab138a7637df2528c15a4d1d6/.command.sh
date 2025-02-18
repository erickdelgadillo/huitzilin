#!/bin/bash -ue
echo "🔍 Running FAISS query..."
python query_faiss.py query.txt > results.txt
echo "✅ Query completed. Check 'results.txt'."
