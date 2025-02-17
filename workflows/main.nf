nextflow.enable.dsl=2

// Definir parámetros
params.query_file = file("data/input/query.txt")

// Proceso extract_pubmed
process extract_pubmed {
    conda 'env/environment.yml'

    input:
    tuple val(query_name), file(query_file)  // ✅ Se usa 'file' en lugar de 'path'

    output:
    file "colibot.db"  // ✅ Se usa 'file' en lugar de 'path'

    script:
    """
    echo "📥 Extracting articles based on query..."
    python bin/extract_pubmed.py ${query_file} -o colibot.db
    """
}

// Proceso generate_faiss
process generate_faiss {
    conda 'env/environment.yml'

    input:
    file "colibot.db"

    output:
    file "colibot_faiss.index",
    file "colibot_article_ids.npy"

    script:
    """
    echo "🔎 Generating FAISS index..."
    python bin/generate_faiss_embeddings.py colibot.db -i colibot_faiss.index -a colibot_article_ids.npy
    """
}

// Proceso query_faiss
process query_faiss {
    conda 'env/environment.yml'

    input:
    file "colibot_faiss.index"
    file "colibot_article_ids.npy"
    file "colibot.db"
    file query_file

    output:
    file "results.txt"

    script:
    """
    echo "🔍 Running FAISS query..."
    python bin/query_faiss.py ${query_file} -i colibot_faiss.index -a colibot_article_ids.npy -d colibot.db -o results.txt
    echo "✅ Query completed. Check 'results.txt'."
    """
}

// Definir el flujo de trabajo
workflow {
    query_ch = Channel.fromPath(params.query_file)

    db_ch = extract_pubmed(query_ch)
    faiss_ch = generate_faiss(db_ch)
    results_ch = query_faiss(faiss_ch, query_ch)

    results_ch.view()
}

