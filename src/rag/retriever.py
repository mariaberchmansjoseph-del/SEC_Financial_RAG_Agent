"""
RAG Retriever — searches ChromaDB index for relevant chunks.
src/rag/retriever.py
"""

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)

VECTORSTORE = "data/vectorstore/sec_index"
COLLECTION  = "sec_filings"
EMBED_MODEL = "all-MiniLM-L6-v2"


def is_prose(text: str) -> bool:
    """
    Return True if text is readable prose, not a table.
    Financial tables have high digit ratios.
    """
    if not text or len(text) < 30:
        return False
    digits = sum(1 for c in text if c.isdigit())
    ratio  = digits / max(len(text), 1)
    return ratio < 0.20


class SECRetriever:
    """
    Retrieves relevant SEC filing chunks for a query.
    Supports filtering by company ticker, year, sector,
    and filing type.
    """

    def __init__(self):
        embed_fn        = SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        client          = chromadb.PersistentClient(
            path=VECTORSTORE
        )
        self.collection = client.get_collection(
            name               = COLLECTION,
            embedding_function = embed_fn
        )
        total = self.collection.count()
        print(f"✅ Retriever loaded: {total:,} chunks")

    def search(
        self,
        query:   str,
        top_k:   int = 5,
        ticker:  str = None,
        year:    str = None,
        sector:  str = None,
        form:    str = None,
    ) -> list:
        """
        Search for relevant chunks matching the query.

        Args:
            query:   Natural language question
            top_k:   Number of results to return
            ticker:  Filter by ticker e.g. "NVDA"
            year:    Filter by year e.g. "2025"
            sector:  Filter by sector e.g. "energy"
            form:    Filter by form type e.g. "10-K"

        Returns:
            List of result dicts sorted by relevance.
        """

        # ── BUILD FILTER ──────────────────────────────────────
        filters = []

        if ticker:
            filters.append(
                {"ticker": {"$eq": ticker.upper()}}
            )
        if year:
            filters.append(
                {"year": {"$eq": str(year)}}
            )
        if sector:
            filters.append(
                {"sector": {"$eq": sector.lower()}}
            )
        if form:
            filters.append(
                {"form_type": {"$eq": form.upper()}}
            )

        if len(filters) == 0:
            where = None
        elif len(filters) == 1:
            where = filters[0]
        else:
            where = {"$and": filters}

        # ── QUERY CHROMADB ────────────────────────────────────
        kwargs = {
            "query_texts": [query],
            "n_results":   top_k * 2,  # fetch extra to filter
            "include":     ["documents", "metadatas",
                            "distances"]
        }
        if where:
            kwargs["where"] = where

        try:
            raw = self.collection.query(**kwargs)
        except Exception as e:
            print(f"  Search error: {e}")
            return []

        docs      = raw["documents"][0]
        metas     = raw["metadatas"][0]
        distances = raw["distances"][0]

        # ── FORMAT AND FILTER ─────────────────────────────────
        results = []
        for doc, meta, dist in zip(docs, metas, distances):
            results.append({
                "text":        doc,
                "score":       round(1 - dist, 4),
                "company":     meta.get("company",     ""),
                "ticker":      meta.get("ticker",      ""),
                "sector":      meta.get("sector",      ""),
                "form_type":   meta.get("form_type",   ""),
                "filing_date": meta.get("filing_date", ""),
                "year":        meta.get("year",        ""),
                "section":     meta.get("section",     ""),
            })

        # Prefer prose over tables
        prose_results = [r for r in results if is_prose(r["text"])]

        # Fall back to all results if prose filter removes everything
        final = prose_results if prose_results else results

        # Return top_k after filtering
        return final[:top_k]

    def format_context(self, results: list) -> str:
        """
        Format retrieved chunks into a context string
        for the LLM prompt. Includes source attribution.
        """
        if not results:
            return "No relevant information found."

        parts = []
        for i, r in enumerate(results, 1):
            source = (
                f"{r['company']} "
                f"({r['form_type']} {r['year']}, "
                f"filed {r['filing_date']})"
            )
            parts.append(
                f"[Source {i}: {source}]\n{r['text']}"
            )

        return "\n\n---\n\n".join(parts)

    def search_and_format(
        self,
        query:   str,
        top_k:   int = 5,
        ticker:  str = None,
        year:    str = None,
        sector:  str = None,
        form:    str = None,
    ) -> tuple:
        """
        Convenience method: search and return both
        results list and formatted context string.
        """
        results = self.search(
            query  = query,
            top_k  = top_k,
            ticker = ticker,
            year   = year,
            sector = sector,
            form   = form,
        )
        context = self.format_context(results)
        return results, context


# ── TEST ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("SEC RETRIEVER TEST")
    print("=" * 60)
    print()

    retriever = SECRetriever()
    print()

    tests = [
        {
            "desc":   "NVIDIA AI chip risks",
            "query":  "risks related to AI chip demand "
                      "and supply chain",
            "ticker": "NVDA",
            "top_k":  3,
        },
        {
            "desc":   "Microsoft Azure cloud revenue",
            "query":  "Azure cloud revenue growth and "
                      "commercial cloud performance",
            "ticker": "MSFT",
            "form":   "10-K",
            "top_k":  3,
        },
        {
            "desc":   "Oil price impact on energy sector",
            "query":  "crude oil prices impact on revenue "
                      "and production",
            "sector": "energy",
            "top_k":  3,
        },
        {
            "desc":   "JPMorgan credit risk 2025",
            "query":  "credit risk management and loan "
                      "loss provisions",
            "ticker": "JPM",
            "year":   "2025",
            "top_k":  3,
        },
        {
            "desc":   "AI strategy across tech companies",
            "query":  "artificial intelligence strategy "
                      "investment and generative AI",
            "sector": "technology",
            "top_k":  5,
        },
    ]

    for test in tests:
        print(f"{'─' * 60}")
        print(f"Query: {test['desc']}")

        results = retriever.search(
            query  = test["query"],
            top_k  = test["top_k"],
            ticker = test.get("ticker"),
            year   = test.get("year"),
            sector = test.get("sector"),
            form   = test.get("form"),
        )

        if not results:
            print("  No results found.")
        else:
            for i, r in enumerate(results, 1):
                print(
                    f"  {i}. {r['company']} "
                    f"({r['form_type']} {r['year']}) "
                    f"[score: {r['score']}]"
                )
                print(f"     {r['text'][:150]}...")

        print()

    print("=" * 60)
    print("RETRIEVER TEST COMPLETE")
    print("=" * 60)