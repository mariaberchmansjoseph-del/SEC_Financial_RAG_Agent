"""
Index SEC filing chunks into ChromaDB with FAISS backend.
Uses sentence-transformers for free local embeddings.
Run: python scripts/index_documents.py
"""

import os
import json
import chromadb
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)

Path("data/vectorstore").mkdir(parents=True, exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────
CHUNKS_PATH   = "data/processed/chunks/all_chunks.jsonl"
VECTORSTORE   = "data/vectorstore/sec_index"
COLLECTION    = "sec_filings"
EMBED_MODEL   = "all-MiniLM-L6-v2"
# Free, fast, good quality for English financial text
# 384 dimensions, runs on CPU in seconds
BATCH_SIZE    = 100


def load_chunks() -> list:
    """Load all chunks from JSONL file."""
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_index(chunks: list):
    """Build ChromaDB index with FAISS backend."""

    print(f"Embedding model: {EMBED_MODEL}")
    print(f"Total chunks:    {len(chunks):,}")
    print()

    # Initialise embedding function
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # Initialise ChromaDB with persistent storage
    client = chromadb.PersistentClient(path=VECTORSTORE)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION)
        print("Deleted existing collection.")
    except Exception:
        pass

    # Create collection
    collection = client.create_collection(
        name               = COLLECTION,
        embedding_function = embed_fn,
        metadata           = {"hnsw:space": "cosine"}
    )
    print(f"Created collection: {COLLECTION}")
    print()

    # Index in batches
    total_indexed = 0
    errors        = 0

    for i in tqdm(range(0, len(chunks), BATCH_SIZE),
                  desc="Indexing chunks"):
        batch = chunks[i:i + BATCH_SIZE]

        ids        = []
        documents  = []
        metadatas  = []

        for chunk in batch:
            text = chunk.get("text", "").strip()
            if not text or len(text) < 20:
                errors += 1
                continue

            ids.append(chunk["chunk_id"])
            documents.append(text)
            metadatas.append({
                "company":     str(chunk.get("company", "")),
                "ticker":      str(chunk.get("ticker", "")),
                "sector":      str(chunk.get("sector", "")),
                "form_type":   str(chunk.get("form_type", "")),
                "filing_date": str(chunk.get("filing_date", "")),
                "year":        str(chunk.get("year", "")),
                "section":     str(chunk.get("section", "")),
                "chunk_index": int(chunk.get("chunk_index", 0)),
                "token_count": int(chunk.get("token_count", 0)),
            })

        if ids:
            try:
                collection.add(
                    ids        = ids,
                    documents  = documents,
                    metadatas  = metadatas
                )
                total_indexed += len(ids)
            except Exception as e:
                errors += len(ids)
                print(f"\n  Batch error: {str(e)[:60]}")

    return collection, total_indexed, errors


def verify_index(collection):
    """Run test queries to verify the index works."""
    print("\nVerifying index with test queries...")
    print()

    test_queries = [
        {
            "query":  "What are NVIDIA risks related to AI chips?",
            "filter": {"ticker": "NVDA"},
            "desc":   "NVIDIA risk factors"
        },
        {
            "query":  "Microsoft Azure cloud revenue growth",
            "filter": {"ticker": "MSFT", "form_type": "10-K"},
            "desc":   "Microsoft annual report"
        },
        {
            "query":  "JPMorgan credit risk management",
            "filter": {"sector": "Financial Services"},
            "desc":   "Banking sector risk"
        },
        {
            "query":  "ExxonMobil oil production outlook",
            "filter": {"ticker": "XOM"},
            "desc":   "ExxonMobil energy outlook"
        },
        {
            "query":  "AI artificial intelligence investment strategy",
            "filter": None,
            "desc":   "AI mentions across all companies"
        },
    ]

    for test in test_queries:
        print(f"Query: '{test['desc']}'")

        kwargs = {
            "query_texts": [test["query"]],
            "n_results":   3,
            "include":     ["documents", "metadatas",
                            "distances"]
        }
        if test["filter"]:
            kwargs["where"] = test["filter"]

        try:
            results = collection.query(**kwargs)
            docs    = results["documents"][0]
            metas   = results["metadatas"][0]
            dists   = results["distances"][0]

            for j, (doc, meta, dist) in enumerate(
                zip(docs, metas, dists)
            ):
                print(f"  Result {j+1}: {meta['company']} "
                      f"({meta['form_type']} {meta['year']}) "
                      f"[score: {1-dist:.3f}]")
                print(f"    {doc[:120]}...")

        except Exception as e:
            print(f"  Error: {str(e)[:60]}")

        print()


def print_index_stats(collection):
    """Print statistics about the built index."""
    count = collection.count()

    print("="*55)
    print("INDEX STATISTICS")
    print("="*55)
    print(f"Total documents indexed: {count:,}")

    # Sample metadata
    sample = collection.get(limit=1000, include=["metadatas"])
    metas  = sample["metadatas"]

    if metas:
        df = pd.DataFrame(metas)

        print(f"\nBy company:")
        company_counts = df["company"].value_counts()
        for company, cnt in company_counts.items():
            print(f"  {company:<25} {cnt:>5,} chunks")

        print(f"\nBy form type:")
        form_counts = df["form_type"].value_counts()
        for form, cnt in form_counts.items():
            print(f"  {form:<10} {cnt:>5,} chunks")

        print(f"\nBy year:")
        year_counts = df["year"].value_counts().sort_index()
        for year, cnt in year_counts.items():
            print(f"  {year:<8} {cnt:>5,} chunks")

        print(f"\nBy section:")
        sec_counts = df["section"].value_counts()
        for sec, cnt in sec_counts.items():
            print(f"  {sec:<15} {cnt:>5,} chunks")


if __name__ == "__main__":
    print("="*55)
    print("SEC FILING INDEX BUILDER (ChromaDB + FAISS)")
    print("="*55)
    print()

    if not Path(CHUNKS_PATH).exists():
        print(f"❌ Chunks not found: {CHUNKS_PATH}")
        print("   Run: python scripts/chunk_documents.py first")
        exit()

    # Load chunks
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks):,} chunks")

    # Build index
    print("\nBuilding index...")
    print("(First run downloads embedding model ~90MB)")
    collection, indexed, errors = build_index(chunks)

    print(f"\n✅ Indexed: {indexed:,} chunks")
    if errors:
        print(f"⚠️  Errors: {errors}")

    # Print stats
    print_index_stats(collection)

    # Verify with test queries
    verify_index(collection)

    print("="*55)
    print("INDEX COMPLETE")
    print("="*55)
    print(f"Stored at: {VECTORSTORE}")
    print(f"\nNext: python src/rag/retriever.py")