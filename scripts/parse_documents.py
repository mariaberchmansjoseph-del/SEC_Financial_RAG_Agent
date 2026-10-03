"""
Parse SEC iXBRL/HTML filings into clean readable text.
Modern SEC filings use inline XBRL — we extract
the human-readable text from within the XBRL tags.
Run: python scripts/parse_documents.py
"""

import re
import warnings
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

Path("data/processed").mkdir(parents=True, exist_ok=True)


def extract_text_from_ixbrl(content: bytes) -> str:
    """
    Extract human-readable text from inline XBRL filing.
    iXBRL embeds readable text inside XML tags.
    We strip the tags and keep the text content.
    """
    try:
        # Decode
        try:
            text_content = content.decode("utf-8", errors="ignore")
        except Exception:
            text_content = content.decode("latin-1",
                                          errors="ignore")

        # Parse with lxml
        soup = BeautifulSoup(text_content, "lxml")

        # Remove truly non-content elements
        for tag in soup(["script", "style", "meta",
                         "link", "head"]):
            tag.decompose()

        # For iXBRL: ix:header contains no readable content
        for tag in soup.find_all("ix:header"):
            tag.decompose()

        # ix:hidden contains hidden XBRL facts — remove
        for tag in soup.find_all("ix:hidden"):
            tag.decompose()

        # Get all text — BeautifulSoup strips XML tags
        # and returns the text content inside them
        full_text = soup.get_text(separator=" ", strip=True)

        # Clean up
        # Remove XBRL taxonomy references that slip through
        full_text = re.sub(
            r'https?://\S+', ' ', full_text
        )
        full_text = re.sub(
            r'[a-z]+:[A-Za-z]+Member\b', ' ', full_text
        )
        full_text = re.sub(
            r'iso4217:[A-Z]+', ' ', full_text
        )
        full_text = re.sub(
            r'xbrli?:[a-zA-Z]+', ' ', full_text
        )
        full_text = re.sub(
            r'us-gaap:[A-Za-z]+', ' ', full_text
        )
        full_text = re.sub(r'\s+', ' ', full_text)
        full_text = full_text.strip()

        return full_text

    except Exception as e:
        return ""


def score_text_quality(text: str) -> float:
    """
    Score text quality 0-1.
    Good text has high ratio of real words.
    """
    if not text or len(text) < 100:
        return 0.0

    words  = text.split()
    if not words:
        return 0.0

    # Count words with majority alphabetic characters
    good_words = sum(
        1 for w in words
        if sum(c.isalpha() for c in w) > len(w) * 0.5
    )

    return good_words / len(words)


def find_section_in_text(text: str,
                          section_name: str) -> str:
    """Extract a named section from filing text."""
    text_lower = text.lower()

    patterns = {
        "item_1a": [
            r"item\s*1a\.?\s*risk\s*factor",
            r"risk\s*factors\s*our\s*business",
            r"item\s*1a\s*\.\s*risk",
        ],
        "item_7": [
            r"item\s*7\.?\s*management",
            r"management.s\s*discussion\s*and\s*analysis",
            r"results\s*of\s*operations",
        ],
        "item_1": [
            r"item\s*1\.?\s*business\s*overview",
            r"item\s*1\.?\s*business\b",
        ],
    }

    for pattern in patterns.get(section_name, []):
        matches = list(re.finditer(
            pattern, text_lower
        ))

        for match in matches:
            start   = match.start()
            preview = text_lower[start:start + 50]

            # Skip table of contents entries
            # (text like "Item 1A. Risk Factors...13")
            if re.search(r'\.\s*\d+\s*$',
                         preview.strip()):
                continue

            content = text[start:start + 15000]
            content = re.sub(r'\s+', ' ', content).strip()

            if len(content) > 300:
                return content

    return ""


def parse_all_filings() -> pd.DataFrame:
    meta_path = "data/raw/metadata/filings_metadata.csv"
    if not Path(meta_path).exists():
        print(f"❌ Not found: {meta_path}")
        return pd.DataFrame()

    df = pd.read_csv(meta_path, encoding="utf-8-sig")
    print(f"Total filings: {len(df)}")
    print("Extracting text from iXBRL filings...")
    print()

    all_docs   = []
    failed     = 0
    low_quality = 0

    for _, row in tqdm(df.iterrows(),
                       total=len(df),
                       desc="Parsing"):
        file_path = Path(row["file_path"])

        if not file_path.exists():
            failed += 1
            continue

        content = file_path.read_bytes()
        if len(content) < 1000:
            failed += 1
            continue

        # Extract text from iXBRL
        full_text = extract_text_from_ixbrl(content)

        if not full_text:
            failed += 1
            continue

        # Score quality
        quality = score_text_quality(full_text)
        if quality < 0.4:
            low_quality += 1
            # Still include but flag it
            pass

        # Keep first 50,000 chars
        full_text_stored = full_text[:500000]

        # Find sections
        item_1a = find_section_in_text(full_text, "item_1a")
        item_7  = find_section_in_text(full_text, "item_7")
        item_1  = find_section_in_text(full_text, "item_1")

        all_docs.append({
            "company":     row["company"],
            "ticker":      row["ticker"],
            "sector":      row["sector"],
            "form_type":   row["form_type"],
            "filing_date": row["filing_date"],
            "year":        row["year"],
            "file_path":   str(file_path),
            "full_text":   full_text_stored,
            "word_count":  len(full_text.split()),
            "quality":     round(quality, 3),
            "item_1":      item_1,
            "item_1a":     item_1a,
            "item_7":      item_7,
            "has_item_1a": len(item_1a) > 300,
            "has_item_7":  len(item_7) > 300,
        })

    result_df = pd.DataFrame(all_docs)

    if result_df.empty:
        print("❌ No documents parsed.")
        print("   Check file paths in metadata CSV.")
        return result_df

    out_path = "data/processed/parsed_documents.csv"
    result_df.to_csv(out_path, index=False,
                     encoding="utf-8-sig")

    print(f"\n{'='*55}")
    print(f"PARSING COMPLETE")
    print(f"{'='*55}")
    print(f"Parsed:       {len(result_df)}")
    print(f"Failed:       {failed}")
    print(f"Low quality:  {low_quality}")

    print(f"\nWord count stats:")
    print(result_df["word_count"].describe()
          .round(0).to_string())

    print(f"\nQuality score stats:")
    print(result_df["quality"].describe()
          .round(3).to_string())

    print(f"\nSection coverage:")
    print(f"  Has Item 1A (Risk):  "
          f"{result_df['has_item_1a'].sum()} / {len(result_df)}")
    print(f"  Has Item 7  (MD&A):  "
          f"{result_df['has_item_7'].sum()} / {len(result_df)}")

    print(f"\nTop companies by word count:")
    top = (result_df.groupby("company")["word_count"]
           .mean().sort_values(ascending=False).head(8))
    print(top.round(0).to_string())

    print(f"\nSaved: {out_path}")
    print(f"\nNext: python scripts/chunk_documents.py")

    return result_df


if __name__ == "__main__":
    print("="*55)
    print("SEC FILING PARSER (iXBRL aware)")
    print("="*55)
    parse_all_filings()