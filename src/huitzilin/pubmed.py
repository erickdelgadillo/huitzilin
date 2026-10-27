import os
import re
from collections.abc import Sequence

from Bio import Entrez

from .models import Article


class PubMedClient:
    """Client for searching and retrieving scientific articles from PubMed."""

    def __init__(
        self,
        email: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.email = email or os.getenv("NCBI_EMAIL")
        self.api_key = api_key or os.getenv("NCBI_API_KEY")

        if not self.email:
            raise RuntimeError(
                "NCBI_EMAIL is not defined. "
                "Set it with: export NCBI_EMAIL='your_email@example.com'"
            )

        Entrez.email = self.email
        Entrez.api_key = self.api_key

    def search(
        self,
        query: str,
        max_results: int = 1000,
    ) -> list[str]:
        """Search PubMed and return matching PMIDs."""

        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        if max_results <= 0:
            raise ValueError("max_results must be greater than zero.")

        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
        )

        try:
            record = Entrez.read(handle)
        finally:
            handle.close()

        return [str(pmid) for pmid in record["IdList"]]

    def fetch(
        self,
        article_ids: Sequence[str],
        batch_size: int = 200,
    ) -> list[Article]:
        """Retrieve PubMed records in batches."""

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        articles: list[Article] = []

        for start in range(0, len(article_ids), batch_size):
            batch = article_ids[start:start + batch_size]

            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="xml",
                retmode="xml",
            )

            try:
                records = Entrez.read(handle)
            finally:
                handle.close()

            for record in records.get("PubmedArticle", []):
                articles.append(self._parse_article(record))

        return articles

    @staticmethod
    def _format_authors(authors) -> tuple[str, ...]:
        names: list[str] = []

        for author in authors:
            if "CollectiveName" in author:
                names.append(str(author["CollectiveName"]))
                continue

            last_name = str(author.get("LastName", "")).strip()
            fore_name = str(author.get("ForeName", "")).strip()

            full_name = f"{last_name} {fore_name}".strip()

            if full_name:
                names.append(full_name)

        return tuple(names)

    @staticmethod
    def _normalize_abstract(abstract_sections) -> str:
        if not abstract_sections:
            return ""

        sections: list[str] = []

        for section in abstract_sections:
            text = str(section).strip()

            if not text:
                continue

            attributes = getattr(section, "attributes", {})
            label = attributes.get("Label")

            if label:
                sections.append(f"{label}: {text}")
            else:
                sections.append(text)

        return " ".join(sections)

    @staticmethod
    def _extract_doi(pubmed_article) -> str | None:
        article_ids = (
            pubmed_article
            .get("PubmedData", {})
            .get("ArticleIdList", [])
        )

        for article_id in article_ids:
            attributes = getattr(article_id, "attributes", {})

            if attributes.get("IdType") == "doi":
                return str(article_id)

        return None

    @staticmethod
    def _extract_year(article_data) -> int | None:
        pub_date = (
            article_data
            .get("Journal", {})
            .get("JournalIssue", {})
            .get("PubDate", {})
        )

        if "Year" in pub_date:
            try:
                return int(str(pub_date["Year"]))
            except ValueError:
                pass

        medline_date = str(pub_date.get("MedlineDate", ""))

        match = re.search(r"\b(19|20)\d{2}\b", medline_date)

        if match:
            return int(match.group())

        return None

    @classmethod
    def _parse_article(cls, pubmed_article) -> Article:
        medline = pubmed_article["MedlineCitation"]
        article_data = medline["Article"]

        pubmed_id = str(medline.get("PMID", ""))

        title = str(
            article_data.get("ArticleTitle", "")
        ).strip()

        abstract_sections = (
            article_data
            .get("Abstract", {})
            .get("AbstractText", [])
        )

        journal = str(
            article_data
            .get("Journal", {})
            .get("Title", "")
        ).strip() or None

        publication_types = tuple(
            str(item)
            for item in article_data.get("PublicationTypeList", [])
        )

        mesh_terms = tuple(
            str(heading.get("DescriptorName", ""))
            for heading in medline.get("MeshHeadingList", [])
            if heading.get("DescriptorName")
        )

        keywords = tuple(
            str(keyword)
            for keyword_group in medline.get("KeywordList", [])
            for keyword in keyword_group
        )

        return Article(
            pubmed_id=pubmed_id,
            title=title,
            abstract=cls._normalize_abstract(abstract_sections),
            authors=cls._format_authors(
                article_data.get("AuthorList", [])
            ),
            doi=cls._extract_doi(pubmed_article),
            journal=journal,
            year=cls._extract_year(article_data),
            publication_types=publication_types,
            mesh_terms=mesh_terms,
            keywords=keywords,
        )
