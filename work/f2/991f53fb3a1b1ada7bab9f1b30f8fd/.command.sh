#!/bin/bash -ue
echo "📥 Extracting articles based on query..."
python extract_pubmed.py query.txt
