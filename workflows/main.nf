nextflow.enable.dsl=2

// Definir parámetros
params.query_file = file("data/input/query.txt")

// Proceso extract_pubmed
process extract_pubmed {
    conda 'env/environment.yml'

    input:
    tuple val(query_name), file(query_file)  // 🔥 Cambia 'path' por 'file'

    output:
    file "colibot.db"  // 🔥 Asegúrate de que sea 'file' y no 'path'

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
    path "data/output/colibot.db"

    output:
    path "data/output/colibot_faiss.index",
    path "data/output/colibot_article_ids.npy"

    script:
    """
    echo "🔎 Generating FAISS index..."
    python bin/generate_faiss_embeddings.py data/output/colibot.db -i data/output/colibot_faiss.index -a data/output/colibot_article_ids.npy
    """
}

// Proceso query_faiss
process query_faiss {
    conda 'env/environment.yml'

    input:
    path "data/output/colibot_faiss.index"
    path "data/output/colibot_article_ids.npy"
    path "data/output/colibot.db"
    path query_file

    output:
    path "data/output/results.txt"

    script:
    """
    echo "🔍 Running FAISS query..."
    python bin/query_faiss.py ${query_file} -i data/output/colibot_faiss.index -a data/output/colibot_article_ids.npy -d data/output/colibot.db -o data/output/results.txt
    echo "✅ Query completed. Check 'data/output/results.txt'."
    """
}

// Definir el flujo de trabajo
workflow {
    query_ch = Channel.fromFilePairs([params.query_file])

    db_ch = extract_pubmed(query_ch)
    faiss_ch = generate_faiss(db_ch)
    results_ch = query_faiss(faiss_ch, query_ch)

    results_ch.view()
}

