"""
Collect all enrichment data for SEC Financial RAG Agent.
Pulls from: EdgarTools, Finnhub, Alpha Vantage, FMP.
Run: python scripts/collect_all_data.py
"""

import os
import json
import time
import yaml
import finnhub
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from edgar import Company, set_identity

load_dotenv()

# ── SETUP ─────────────────────────────────────────────────
set_identity("Portfolio Reasearch mariaberchmansjoseph@gmail.com")

FINNHUB_KEY   = os.getenv("FINNHUB_API_KEY", "")
AV_KEY        = os.getenv("ALPHA_VANTAGE_KEY", "")
FMP_KEY       = os.getenv("FMP_API_KEY", "")

Path("data/raw/structured").mkdir(parents=True, exist_ok=True)
Path("data/raw/transcripts").mkdir(parents=True, exist_ok=True)
Path("data/raw/prices").mkdir(parents=True, exist_ok=True)
Path("data/raw/analyst").mkdir(parents=True, exist_ok=True)
Path("data/raw/metadata").mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with open("configs/companies.yaml") as f:
        return yaml.safe_load(f)


# ── LAYER 2: EDGARTOOLS STRUCTURED FINANCIALS ─────────────────
def collect_structured_financials(companies: list) -> dict:
    """
    Extract structured financial data using EdgarTools.
    Returns revenue, net income, EPS for each company.
    """
    print("\n" + "="*60)
    print("LAYER 2: STRUCTURED FINANCIAL DATA (EdgarTools)")
    print("="*60)

    all_financials = {}

    for company in companies:
        name   = company["name"]
        ticker = company["ticker"]
        cik    = company["cik"]

        print(f"\n  {name} ({ticker})...")

        try:
            edgar_company = Company(ticker)
            filings = edgar_company.get_filings(form="10-K")

            if not filings:
                print(f"    No 10-K filings found")
                continue

            company_data = {
                "company": name,
                "ticker":  ticker,
                "filings": []
            }

            # Get last 4 annual reports
            for i, filing in enumerate(filings[:4]):
                try:
                    filing_data = {
                        "period":      str(filing.period_of_report
                                         if hasattr(filing,
                                         'period_of_report')
                                         else ""),
                        "filed":       str(filing.filed
                                         if hasattr(filing, 'filed')
                                         else ""),
                        "form":        filing.form,
                        "description": filing.description
                                         if hasattr(filing,
                                         'description') else ""
                    }

                    # Try to get financial data
                    try:
                        obj = filing.obj()
                        if hasattr(obj, 'financials'):
                            fins = obj.financials
                            if fins and hasattr(fins, 'income_statement'):
                                inc = fins.income_statement
                                if inc is not None:
                                    filing_data["income_statement"] = \
                                        inc.to_dict() if hasattr(
                                            inc, 'to_dict') else str(inc)
                    except Exception:
                        pass

                    company_data["filings"].append(filing_data)
                    print(f"    Filing {i+1}: {filing_data['period']}")

                except Exception as e:
                    print(f"    Filing {i+1} error: {str(e)[:50]}")

                time.sleep(0.3)

            all_financials[ticker] = company_data

            # Save per company
            path = f"data/raw/structured/{ticker}_financials.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(company_data, f,
                          ensure_ascii=False, indent=2,
                          default=str)
            print(f"    ✅ Saved: {path}")

        except Exception as e:
            print(f"    ❌ Error: {str(e)[:80]}")

        time.sleep(0.5)

    return all_financials


# ── LAYER 3: FINNHUB EARNINGS TRANSCRIPTS ─────────────────────
def collect_earnings_transcripts(companies: list) -> dict:
    """
    Download earnings call transcripts from Finnhub.
    Returns speaker-tagged Q&A sessions.
    """
    print("\n" + "="*60)
    print("LAYER 3: EARNINGS CALL TRANSCRIPTS (Finnhub)")
    print("="*60)

    if not FINNHUB_KEY:
        print("  ⚠️  FINNHUB_API_KEY not set. Skipping.")
        return {}

    client = finnhub.Client(api_key=FINNHUB_KEY)
    all_transcripts = {}

    for company in companies:
        name   = company["name"]
        ticker = company["ticker"]

        print(f"\n  {name} ({ticker})...")

        try:
            # Get list of available transcripts
            transcript_list = client.earnings_call_transcripts_list(
                ticker
            )

            if not transcript_list or \
               not transcript_list.get("transcripts"):
                print(f"    No transcripts available")
                continue

            transcripts = transcript_list["transcripts"]
            print(f"    Found {len(transcripts)} transcripts")

            company_transcripts = []

            # Get last 4 quarterly transcripts
            for t in transcripts[:4]:
                try:
                    transcript_id = t.get("id")
                    if not transcript_id:
                        continue

                    time.sleep(0.5)
                    transcript = client.earnings_call_transcripts(
                        transcript_id
                    )

                    if transcript:
                        transcript_data = {
                            "id":     transcript_id,
                            "year":   t.get("year"),
                            "quarter": t.get("quarter"),
                            "title":  t.get("title", ""),
                            "content": transcript
                        }
                        company_transcripts.append(transcript_data)
                        print(f"    ✅ Q{t.get('quarter')} "
                              f"{t.get('year')}")

                except Exception as e:
                    print(f"    Transcript error: {str(e)[:50]}")

            if company_transcripts:
                all_transcripts[ticker] = company_transcripts
                path = (f"data/raw/transcripts/"
                        f"{ticker}_transcripts.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(company_transcripts, f,
                              ensure_ascii=False, indent=2,
                              default=str)
                print(f"    ✅ Saved {len(company_transcripts)} "
                      f"transcripts")

        except Exception as e:
            print(f"    ❌ Error: {str(e)[:80]}")

        time.sleep(1)

    return all_transcripts


# ── LAYER 4: ALPHA VANTAGE COMPANY OVERVIEW + PRICES ──────────
def collect_market_data(companies: list) -> dict:
    """
    Download company overview and price data from Alpha Vantage.
    """
    print("\n" + "="*60)
    print("LAYER 4: MARKET DATA (Alpha Vantage)")
    print("="*60)

    if not AV_KEY:
        print("  ⚠️  ALPHA_VANTAGE_KEY not set. Skipping.")
        return {}

    base_url  = "https://www.alphavantage.co/query"
    all_data  = {}

    for i, company in enumerate(companies):
        name   = company["name"]
        ticker = company["ticker"]

        print(f"\n  {name} ({ticker})...")

        try:
            # Company Overview
            time.sleep(1.2)  # 5 requests/min free tier
            r = requests.get(base_url, params={
                "function": "OVERVIEW",
                "symbol":   ticker,
                "apikey":   AV_KEY
            }, timeout=15)

            overview = {}
            if r.status_code == 200:
                data = r.json()
                if "Symbol" in data:
                    overview = {
                        "name":           data.get("Name"),
                        "sector":         data.get("Sector"),
                        "industry":       data.get("Industry"),
                        "description":    data.get("Description", "")[:500],
                        "market_cap":     data.get("MarketCapitalization"),
                        "pe_ratio":       data.get("PERatio"),
                        "eps":            data.get("EPS"),
                        "revenue_ttm":    data.get("RevenueTTM"),
                        "profit_margin":  data.get("ProfitMargin"),
                        "52_week_high":   data.get("52WeekHigh"),
                        "52_week_low":    data.get("52WeekLow"),
                        "analyst_target": data.get("AnalystTargetPrice"),
                        "beta":           data.get("Beta"),
                        "dividend_yield": data.get("DividendYield"),
                        "exchange":       data.get("Exchange"),
                    }
                    print(f"    ✅ Overview: {overview.get('sector')}")

            # Monthly prices (last 5 years)
            time.sleep(1.2)
            r2 = requests.get(base_url, params={
                "function":   "TIME_SERIES_MONTHLY_ADJUSTED",
                "symbol":     ticker,
                "apikey":     AV_KEY,
                "outputsize": "full"
            }, timeout=15)

            prices = {}
            if r2.status_code == 200:
                data2 = r2.json()
                ts    = data2.get(
                    "Monthly Adjusted Time Series", {}
                )
                # Keep last 60 months (5 years)
                prices = dict(list(ts.items())[:60])
                print(f"    ✅ Prices: {len(prices)} months")

            company_data = {
                "ticker":   ticker,
                "name":     name,
                "overview": overview,
                "prices":   prices
            }
            all_data[ticker] = company_data

            path = f"data/raw/prices/{ticker}_market.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(company_data, f,
                          ensure_ascii=False, indent=2)
            print(f"    ✅ Saved: {path}")

        except Exception as e:
            print(f"    ❌ Error: {str(e)[:80]}")

    return all_data


# ── LAYER 5: FMP ANALYST RATINGS ──────────────────────────────
def collect_analyst_data(companies: list) -> dict:
    """
    Download analyst ratings and price targets from FMP.
    """
    print("\n" + "="*60)
    print("LAYER 5: ANALYST DATA (FMP)")
    print("="*60)

    if not FMP_KEY:
        print("  ⚠️  FMP_API_KEY not set. Skipping.")
        return {}

    base_url = "https://financialmodelingprep.com/api/v3"
    all_data = {}

    for company in companies:
        name   = company["name"]
        ticker = company["ticker"]

        print(f"\n  {name} ({ticker})...")

        try:
            time.sleep(0.3)
            r = requests.get(
                f"{base_url}/analyst-stock-recommendations/{ticker}",
                params={"apikey": FMP_KEY, "limit": 10},
                timeout=15
            )

            ratings = []
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    ratings = data[:10]
                    print(f"    ✅ {len(ratings)} analyst ratings")

            r2 = requests.get(
                f"{base_url}/price-target/{ticker}",
                params={"apikey": FMP_KEY, "limit": 5},
                timeout=15
            )

            targets = []
            if r2.status_code == 200:
                data2 = r2.json()
                if isinstance(data2, list):
                    targets = data2[:5]
                    print(f"    ✅ {len(targets)} price targets")

            company_data = {
                "ticker":  ticker,
                "name":    name,
                "ratings": ratings,
                "targets": targets
            }
            all_data[ticker] = company_data

            path = f"data/raw/analyst/{ticker}_analyst.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(company_data, f,
                          ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"    ❌ Error: {str(e)[:80]}")

        time.sleep(0.5)

    return all_data


# ── MASTER METADATA ───────────────────────────────────────────
def save_master_metadata(
    companies:    list,
    financials:   dict,
    transcripts:  dict,
    market_data:  dict,
    analyst_data: dict
):
    """Save unified metadata about all collected data."""
    metadata = {
        "collected_at": datetime.utcnow().isoformat(),
        "total_companies": len(companies),
        "companies": []
    }

    for company in companies:
        ticker = company["ticker"]
        entry  = {
            "name":               company["name"],
            "ticker":             ticker,
            "sector":             company["sector"],
            "cik":                company["cik"],
            "has_filings":        True,
            "has_structured_fin": ticker in financials,
            "has_transcripts":    ticker in transcripts,
            "has_market_data":    ticker in market_data,
            "has_analyst_data":   ticker in analyst_data,
            "transcript_count":   len(transcripts.get(ticker, [])),
        }
        metadata["companies"].append(entry)

    path = "data/raw/metadata/master_metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Master metadata saved: {path}")

    # Print summary table
    print("\n" + "="*60)
    print("DATA COLLECTION SUMMARY")
    print("="*60)
    print(f"{'Company':<20} {'Fin':>4} {'Trans':>5} "
          f"{'Mkt':>4} {'Anlst':>5}")
    print("-"*40)
    for c in metadata["companies"]:
        print(f"{c['name'][:20]:<20} "
              f"{'✅' if c['has_structured_fin'] else '❌':>4} "
              f"{'✅' if c['has_transcripts'] else '❌':>5} "
              f"{'✅' if c['has_market_data'] else '❌':>4} "
              f"{'✅' if c['has_analyst_data'] else '❌':>5}")


# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("SEC FINANCIAL RAG — DATA COLLECTION")
    print("="*60)
    print(f"Started: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")

    config    = load_config()
    companies = config["companies"]
    print(f"Companies: {len(companies)}")

    # Collect all layers
    financials   = collect_structured_financials(companies)
    transcripts  = collect_earnings_transcripts(companies)
    market_data  = collect_market_data(companies)
    analyst_data = collect_analyst_data(companies)

    # Save master metadata
    save_master_metadata(
        companies, financials, transcripts,
        market_data, analyst_data
    )

    print(f"\nFinished: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")
    print("\nNext step: python scripts/parse_documents.py")