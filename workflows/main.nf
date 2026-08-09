nextflow.enable.dsl=2

params.query_file = file("${projectDir}/../data/query.txt")
params.script_extract = file("${projectDir}/../bin/extract_pubmed.py")
params.script_generate = file("${projectDir}/../bin/generate_Faiss_embeddings.py")
params.script_query = file("${projectDir}/../bin/query_faiss.py")

process extract_pubmed {
    conda "${projectDir}/../environment/environment.yml"

    input:
    path(query_file)
    path(script_extract)  

    output:
    path "colibot.db"

    script:
    """
    echo "📥 Extracting articles based on query..."
    python extract_pubmed.py query.txt
    """
}

process generate_faiss {
    conda "${projectDir}/../environment/environment.yml"

    input:
    path("colibot.db")
    path(script_generate)

    output:
    path "colibot_faiss.index"
    path "colibot_article_ids.npy"

    script:
    """
    echo "🔎 Generating FAISS index..."
    python generate_Faiss_embeddings.py
    """
}

process query_faiss {
    publishDir "${projectDir}/../results", mode: 'copy'
    conda "${projectDir}/../environment/environment.yml"

    input:
    path("colibot_faiss.index")
    path("colibot_article_ids.npy")
    path("colibot.db")
    path(query_file)
    path(script_query)

    output:
    path "results.txt"

    script:
    """
    echo "🔍 Running FAISS query..."
    python query_faiss.py query.txt
    echo "✅ Query completed. Check 'results.txt'."
    """
}


workflow {
    db_ch = extract_pubmed(params.query_file, params.script_extract)
    faiss_ch = generate_faiss(db_ch, params.script_generate)

    query_faiss(
        faiss_ch[0], faiss_ch[1], db_ch, params.query_file, params.script_query
    )
}
