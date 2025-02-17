nextflow.enable.dsl=2

params.query_file = file("data/input/query.txt")

process extract_pubmed {
    conda 'env/environment.yml'

    input:
    file query_file  // ✅ CORREGIDO

    output:
    file "colibot.db"  // ✅ CORREGIDO

    script:
    """
    echo "📥 Extracting articles based on query..."
    python bin/extract_pubmed.py ${query_file} colibot.db
    """
}

process generate_faiss {
    conda 'env/environment.yml'

   input:
   file colibot_db

    output:
    file "colibot_faiss.index",
    file "colibot_article_ids.npy"

    script:
    """
    echo "🔎 Generating FAISS index..."
    python bin/generate_faiss_embeddings.py colibot.db -i colibot_faiss.index -a colibot_article_ids.npy
    """
}

process query_faiss {
    conda 'env/environment.yml'

    input:
    tuple file(colibot_faiss_index), file(colibot_article_ids), file(colibot_db), file(query_file)  

    output:
    file "results.txt"

    script:
    """
    echo "🔍 Running FAISS query..."
    python bin/query_faiss.py ${query_file} -i ${colibot_faiss_index} -a ${colibot_article_ids} -d ${colibot_db} -o results.txt
    echo "✅ Query completed. Check 'results.txt'."
    """
}

workflow {
    query_ch = Channel.fromPath(params.query_file)

    db_ch = extract_pubmed(query_ch)
    faiss_ch = generate_faiss(db_ch)
    results_ch = query_faiss(faiss_ch, query_ch)

    results_ch.view()
}

