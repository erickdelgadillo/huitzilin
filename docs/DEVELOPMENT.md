# Development and quality checks

## Local environment

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pip check
python -m pytest
```

The `dev` extra includes NumPy and FAISS because existing tests import both.
The `semantic` extra adds Sentence Transformers for actual embedding inference;
the default test suite uses a deterministic fake encoder and **does not download a model**.

## CI

GitHub Actions runs the test suite on Python 3.10, 3.11 and 3.12.
The workflow is triggered by pull requests, pushes to `main`, and manual dispatch.
No NCBI API key, network access to PubMed, or downloaded embedding models are required by tests.

Before merging a pull request, ensure each matrix job has passed.
CI tests individual modules and regression scenarios; it does **not** yet validate
real PubMed connectivity, scientific relevance quality, or end-to-end semantic model performance.

## Future quality work

- Add a unified hybrid retriever and integration tests.
- Build an annotated scientific relevance benchmark for Precision@K, MRR and NDCG.
- Add a separately scheduled online smoke test with NCBI-compliant rate limits.
- Test reproducible vector-index persistence when implemented.
