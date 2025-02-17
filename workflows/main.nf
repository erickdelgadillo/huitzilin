nextflow.enable.dsl=2

// Definir parámetros de archivos
params.query_file = file("data/input/query.txt")

// Definir procesos
process extract_pubmed {
    conda 'env/environment.yml'

    input:
    path query_file

    output:
    path "colibot.db"

    script:
    '''
    echo "📥 Extracting articles based on query..."
    python bin/extract_pubmed.py ${query_file} > colibot.db
    '''
}

process generate_faiss {
    conda 'env/environment.yml'

    input:
    path "colibot.db"

    output:
    path "colibot_faiss.index"
    path "colibot_article_ids.npy"

    script:
    '''
    echo "🔎 Generating FAISS index..."
    python bin/generate_faiss_embeddings.py
    '''
}

process query_faiss {
    conda 'env/environment.yml'

    input:
    path "colibot_faiss.index"
    path "colibot_article_ids.npy"
    path "colibot.db"
    path query_file

    output:
    path "results.txt"

    script:
    '''
    echo "🔍 Running FAISS query..."
    python bin/query_faiss.py ${query_file} > results.txt
    echo "✅ Query completed. Check 'results.txt'."
    '''
}

// Definir el flujo de trabajo
workflow {
    query_ch = Channel.fromPath(params.query_file)  

    db_ch = extract_pubmed(query_ch)
    faiss_ch = generate_faiss(db_ch)
    results_ch = query_faiss(faiss_ch, query_ch)

    results_ch.view()
}

