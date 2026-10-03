"""
Chunk parsed SEC documents into RAG-ready segments.
Run: python scripts/chunk_documents.py
"""

import re
import json
import tiktoken
import pandas as pd
from pathlib import Path
from tqdm import tqdm

Path("data/processed/chunks").mkdir(parents=True, exist_ok=True)

CHUNK_SIZE   = 512
CHUNK_OVERLAP = 64
MIN_CHUNK    = 50

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def chunk_text(text: str) -> list:
    if not text or len(text.strip()) < 50:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences
                 if len(s.strip()) > 20]

    if not sentences:
        return []

    chunks       = []
    current      = []
    current_toks = 0

    for sentence in sentences:
        sent_toks = count_tokens(sentence)

        if sent_toks > CHUNK_SIZE:
            tokens = enc.encode(sentence)
            for i in range(0, len(tokens),
                           CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_tokens = tokens[i:i + CHUNK_SIZE]
                chunk_text   = enc.decode(chunk_tokens)
                if count_tokens(chunk_text) >= MIN_CHUNK:
                    chunks.append(chunk_text)
            continue

        if current_toks + sent_toks > CHUNK_SIZE and current:
            chunk = " ".join(current)
            if count_tokens(chunk) >= MIN_CHUNK:
                chunks.append(chunk)

            overlap_sents = []
            overlap_toks  = 0
            for s in reversed(current):
                st = count_tokens(s)
                if overlap_toks + st <= CHUNK_OVERLAP:
                    overlap_sents.insert(0, s)
                    overlap_toks += st
                else:
                    break

            current      = overlap_sents
            current_toks = overlap_toks

        current.append(sentence)
        current_toks += sent_toks

    if current:
        chunk = " ".join(current)
        if count_tokens(chunk) >= MIN_CHUNK:
            chunks.append(chunk)

    return chunks


def classify_section(text: str) -> str:
    t = text.lower()
    risk_words = ["risk", "risks", "uncertainty",
                  "adverse", "litigation", "failure",
                  "volatile", "competition", "regulatory"]
    mda_words  = ["revenue", "income", "margin",
                  "operating", "growth", "increased",
                  "decreased", "million", "billion",
                  "compared to prior", "results of operations"]

    risk_score = sum(1 for w in risk_words if w in t)
    mda_score  = sum(1 for w in mda_words  if w in t)

    if risk_score >= 3:
        return "risk_factors"
    elif mda_score >= 4:
        return "mda"
    return "general"


def chunk_all_documents():
    parsed_path = "data/processed/parsed_documents.csv"

    if not Path(parsed_path).exists():
        print(f"Not found: {parsed_path}")
        print("Run: python scripts/parse_documents.py first")
        return

    df = pd.read_csv(parsed_path, encoding="utf-8-sig")
    print(f"Documents to chunk: {len(df)}")
    print(f"Chunk size:  {CHUNK_SIZE} tokens")
    print(f"Overlap:     {CHUNK_OVERLAP} tokens")
    print()

    all_chunks = []
    chunk_id   = 0
    skipped    = 0

    for _, row in tqdm(df.iterrows(),
                       total=len(df),
                       desc="Chunking"):

        full_text = str(row.get("full_text", ""))

        if len(full_text) < 100:
            skipped += 1
            continue

        text_chunks = chunk_text(full_text)

        for i, chunk in enumerate(text_chunks):
            section = classify_section(chunk)

            all_chunks.append({
                "chunk_id":     f"chunk_{chunk_id:06d}",
                "company":      str(row.get("company", "")),
                "ticker":       str(row.get("ticker", "")),
                "sector":       str(row.get("sector", "")),
                "form_type":    str(row.get("form_type", "")),
                "filing_date":  str(row.get("filing_date", "")),
                "year":         str(row.get("year", "")),
                "section":      section,
                "chunk_index":  i,
                "total_chunks": len(text_chunks),
                "token_count":  count_tokens(chunk),
                "text":         chunk,
            })
            chunk_id += 1

    # Save CSV
    chunks_df = pd.DataFrame(all_chunks)
    csv_path  = "data/processed/chunks/all_chunks.csv"
    chunks_df.to_csv(csv_path, index=False,
                     encoding="utf-8-sig")

    # Save JSONL
    jsonl_path = "data/processed/chunks/all_chunks.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False)
                    + "\n")

    print(f"\n{'='*55}")
    print(f"CHUNKING COMPLETE")
    print(f"{'='*55}")
    print(f"Total chunks:       {len(all_chunks):,}")
    print(f"Skipped documents:  {skipped}")
    print(f"Avg per document:   "
          f"{len(all_chunks)/max(len(df)-skipped,1):.0f}")

    print(f"\nToken count stats:")
    print(chunks_df["token_count"].describe()
          .round(0).to_string())

    print(f"\nChunks by section:")
    print(chunks_df["section"].value_counts().to_string())

    print(f"\nChunks by company (top 10):")
    print(chunks_df.groupby("company")["chunk_id"]
          .count()
          .sort_values(ascending=False)
          .head(10)
          .to_string())

    print(f"\nChunks by form type:")
    print(chunks_df.groupby("form_type")["chunk_id"]
          .count().to_string())

    print(f"\nSaved:")
    print(f"  CSV:   {csv_path}")
    print(f"  JSONL: {jsonl_path}")
    print(f"\nNext: python scripts/index_documents.py")


if __name__ == "__main__":
    print("="*55)
    print("SEC FILING DOCUMENT CHUNKER")
    print("="*55)
    chunk_all_documents()