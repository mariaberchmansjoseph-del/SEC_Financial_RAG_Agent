"""
Download SEC filings from EDGAR for all companies
defined in configs/companies.yaml.

SEC EDGAR is free — no API key required.
Run: python scripts/download_filings.py
"""

import os
import json
import time
import yaml
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

Path("data/raw/filings").mkdir(parents=True, exist_ok=True)
Path("data/raw/metadata").mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Research Project researcher@project.com",
    "Accept-Encoding": "gzip, deflate",
}

BASE_URL  = "https://data.sec.gov"
EDGAR_URL = "https://www.sec.gov"


def load_config(path: str = "configs/companies.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_company_submissions(cik: str) -> dict:
    cik_padded = cik.zfill(10)
    url = f"{BASE_URL}/submissions/CIK{cik_padded}.json"
    try:
        time.sleep(0.5)
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
        print(f"  HTTP {r.status_code} for CIK {cik}")
    except Exception as e:
        print(f"  Error: {e}")
    return None


def get_filings_list(
    submissions: dict,
    form_types:  list,
    start_year:  int,
    end_year:    int
) -> list:
    if not submissions:
        return []

    recent = submissions.get("filings", {}).get("recent", {})
    if not recent:
        return []

    forms        = recent.get("form", [])
    dates        = recent.get("filingDate", [])
    accessions   = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    filings = []
    for i, form in enumerate(forms):
        if form not in form_types:
            continue
        date = dates[i] if i < len(dates) else ""
        if not date:
            continue
        year = int(date[:4])
        if year < start_year or year > end_year:
            continue
        filings.append({
            "form_type":   form,
            "filing_date": date,
            "accession":   accessions[i] if i < len(accessions) else "",
            "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
        })
    return filings


def download_filing_document(
    cik:         str,
    accession:   str,
    primary_doc: str,
    save_path:   Path
) -> bool:
    if save_path.exists() and save_path.stat().st_size > 1000:
        return True  # already downloaded

    acc_clean = accession.replace("-", "")
    cik_int   = int(cik)

    # Try primary document first
    if primary_doc:
        url = (f"{EDGAR_URL}/Archives/edgar/data/"
               f"{cik_int}/{acc_clean}/{primary_doc}")
        try:
            time.sleep(0.5)
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200 and len(r.content) > 1000:
                save_path.write_bytes(r.content)
                return True
        except Exception:
            pass

    # Fallback: fetch filing index
    index_url = (f"{EDGAR_URL}/Archives/edgar/data/"
                 f"{cik_int}/{acc_clean}/{acc_clean}-index.json")
    try:
        time.sleep(0.5)
        r = requests.get(index_url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            items = r.json().get("directory", {}).get("item", [])
            for item in items:
                name = item.get("name", "")
                if (name.lower().endswith((".htm", ".html"))
                        and "exhibit" not in name.lower()
                        and "ex" not in name.lower()[:3]):
                    doc_url = (f"{EDGAR_URL}/Archives/edgar/data/"
                               f"{cik_int}/{acc_clean}/{name}")
                    time.sleep(0.5)
                    doc_r = requests.get(
                        doc_url, headers=HEADERS, timeout=30
                    )
                    if doc_r.status_code == 200 and \
                       len(doc_r.content) > 1000:
                        save_path.write_bytes(doc_r.content)
                        return True
    except Exception:
        pass

    return False


def download_all_filings(config: dict) -> pd.DataFrame:
    companies  = config["companies"]
    form_types = config["filing_types"]
    start_year = config["years"]["start"]
    end_year   = config["years"]["end"]

    all_metadata = []

    print("=" * 60)
    print("SEC EDGAR FILING DOWNLOADER")
    print("=" * 60)
    print(f"Companies:    {len(companies)}")
    print(f"Filing types: {form_types}")
    print(f"Years:        {start_year} - {end_year}")
    print()

    for company in companies:
        name   = company["name"]
        cik    = company["cik"]
        ticker = company["ticker"]
        sector = company["sector"]

        print(f"\n{'─'*55}")
        print(f"{name} ({ticker})")

        submissions = get_company_submissions(cik)
        if not submissions:
            print(f"  ❌ Could not fetch submissions")
            continue

        filings = get_filings_list(
            submissions, form_types, start_year, end_year
        )
        print(f"  Found {len(filings)} filings")

        for filing in filings:
            form_type   = filing["form_type"]
            filing_date = filing["filing_date"]
            accession   = filing["accession"]
            primary_doc = filing["primary_doc"]

            if not accession:
                continue

            safe_name = name.replace(" ", "_").replace("/", "_")
            save_dir  = (Path("data/raw/filings")
                         / sector / safe_name)
            save_dir.mkdir(parents=True, exist_ok=True)
            filename  = f"{safe_name}_{form_type}_{filing_date}.htm"
            save_path = save_dir / filename

            success = download_filing_document(
                cik, accession, primary_doc, save_path
            )

            if success:
                size_kb = save_path.stat().st_size / 1024
                print(f"  ✅ {form_type} {filing_date} "
                      f"({size_kb:.0f}KB)")
                all_metadata.append({
                    "company":      name,
                    "ticker":       ticker,
                    "sector":       sector,
                    "cik":          cik,
                    "form_type":    form_type,
                    "filing_date":  filing_date,
                    "year":         filing_date[:4],
                    "accession":    accession,
                    "file_path":    str(save_path),
                    "file_size_kb": size_kb,
                    "downloaded_at": datetime.utcnow().isoformat()
                })
            else:
                print(f"  ❌ {form_type} {filing_date} failed")

        time.sleep(0.5)

    df        = pd.DataFrame(all_metadata)
    meta_path = "data/raw/metadata/filings_metadata.csv"
    df.to_csv(meta_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"Total downloaded:  {len(df)}")
    if len(df) > 0:
        print(f"Total size:        "
              f"{df['file_size_kb'].sum()/1024:.1f} MB")
        print(f"\nBy company:")
        print(df.groupby("company")["form_type"]
              .count().to_string())
        print(f"\nBy filing type:")
        print(df.groupby("form_type")["company"]
              .count().to_string())
    print(f"\nMetadata: {meta_path}")
    print(f"\nNext step: python scripts/parse_documents.py")

    return df


if __name__ == "__main__":
    config = load_config()
    df     = download_all_filings(config)