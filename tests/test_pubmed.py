"""Offline PubMed client tests: no network requests or NCBI credentials."""
from unittest.mock import patch

import pytest

from huitzilin.pubmed import PubMedClient


def test_email_is_required(monkeypatch):
    monkeypatch.delenv("NCBI_EMAIL", raising=False)
    with pytest.raises(RuntimeError, match="NCBI_EMAIL"):
        PubMedClient()


def test_search_rejects_invalid_inputs():
    client = PubMedClient(email="testing@example.org")
    with pytest.raises(ValueError, match="empty"):
        client.search("  ")
    with pytest.raises(ValueError, match="max_results"):
        client.search("archaea", max_results=0)


def test_fetch_rejects_invalid_batch_size():
    client = PubMedClient(email="testing@example.org")
    with pytest.raises(ValueError, match="batch_size"):
        client.fetch(["1"], batch_size=0)


def test_fetch_empty_list_does_not_contact_ncbi():
    client = PubMedClient(email="testing@example.org")
    with patch("huitzilin.pubmed.Entrez.efetch") as fetch:
        assert client.fetch([]) == []
        fetch.assert_not_called()


def test_parse_structured_article_without_optional_fields():
    record = {
        "MedlineCitation": {
            "PMID": "1234",
            "Article": {
                "ArticleTitle": "Marine microbes",
                "Abstract": {"AbstractText": ["Archaea in the ocean."]},
                "Journal": {
                    "Title": "Test Journal",
                    "JournalIssue": {"PubDate": {"MedlineDate": "2022 Jan-Feb"}},
                },
            },
        },
    }
    article = PubMedClient._parse_article(record)
    assert article.pubmed_id == "1234"
    assert article.title == "Marine microbes"
    assert article.abstract == "Archaea in the ocean."
    assert article.year == 2022
    assert article.authors == ()
    assert article.doi is None
