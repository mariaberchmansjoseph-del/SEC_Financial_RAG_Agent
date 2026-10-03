"""
Fix missing Alpha Vantage market data.
Re-downloads only the failed companies.
Waits 15 seconds between requests (free tier safe).
Run: python scripts/fix_market_data.py
"""

import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

AV_KEY   = os.getenv("ALPHA_VANTAGE_KEY", "")
BASE_URL = "https://www.alphavantage.co/query"

# Companies that failed (file size < 1000 bytes)
FAILED_TICKERS = {
    "BK":  "BNY Mellon",
    "COP": "ConocoPhillips",
    "CVX": "Chevron",
    "MA":  "Mastercard",
    "V":   "Visa",
    "XOM": "ExxonMobil",
    "WFC": "Wells Fargo",  # replacement for HSBC
}


def fetch_with_retry(params: dict, retries: int = 3) -> dict:
    """Fetch from Alpha Vantage with retry on rate limit."""
    for attempt in range(retries):
        try:
            r = requests.get(BASE_URL, params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                # Check for rate limit message
                if "Note" in data or "Information" in data:
                    msg = data.get("Note", data.get("Information", ""))
                    print(f"    Rate limit hit. Waiting 60s...")
                    time.sleep(60)
                    continue
                return data
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return {}


def fix_company(ticker: str, name: str):
    """Re-download market data for one company."""
    print(f"\n{name} ({ticker})...")

    # Overview
    print(f"  Fetching overview...")
    overview_data = fetch_with_retry({
        "function": "OVERVIEW",
        "symbol":   ticker,
        "apikey":   AV_KEY
    })

    overview = {}
    if "Symbol" in overview_data:
        overview = {
            "name":           overview_data.get("Name"),
            "sector":         overview_data.get("Sector"),
            "industry":       overview_data.get("Industry"),
            "description":    overview_data.get("Description", "")[:500],
            "market_cap":     overview_data.get("MarketCapitalization"),
            "pe_ratio":       overview_data.get("PERatio"),
            "eps":            overview_data.get("EPS"),
            "revenue_ttm":    overview_data.get("RevenueTTM"),
            "profit_margin":  overview_data.get("ProfitMargin"),
            "52_week_high":   overview_data.get("52WeekHigh"),
            "52_week_low":    overview_data.get("52WeekLow"),
            "analyst_target": overview_data.get("AnalystTargetPrice"),
            "beta":           overview_data.get("Beta"),
            "exchange":       overview_data.get("Exchange"),
        }
        print(f"  ✅ Overview: {overview.get('sector', 'unknown')}")
    else:
        print(f"  ❌ Overview empty")

    # Wait between requests
    print(f"  Waiting 15s for rate limit...")
    time.sleep(15)

    # Prices
    print(f"  Fetching prices...")
    price_data = fetch_with_retry({
        "function":   "TIME_SERIES_MONTHLY_ADJUSTED",
        "symbol":     ticker,
        "apikey":     AV_KEY,
        "outputsize": "full"
    })

    prices = {}
    ts = price_data.get("Monthly Adjusted Time Series", {})
    if ts:
        prices = dict(list(ts.items())[:60])
        print(f"  ✅ Prices: {len(prices)} months")
    else:
        print(f"  ❌ Prices empty")

    # Save
    company_data = {
        "ticker":   ticker,
        "name":     name,
        "overview": overview,
        "prices":   prices
    }

    path = f"data/raw/prices/{ticker}_market.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(company_data, f, ensure_ascii=False, indent=2)

    size = Path(path).stat().st_size
    print(f"  ✅ Saved: {path} ({size:,} bytes)")

    # Wait before next company
    print(f"  Waiting 15s before next company...")
    time.sleep(15)


if __name__ == "__main__":
    if not AV_KEY:
        print("❌ ALPHA_VANTAGE_KEY not in .env")
        exit()

    print("="*55)
    print("FIXING MISSING ALPHA VANTAGE DATA")
    print("="*55)
    print(f"Companies to fix: {len(FAILED_TICKERS)}")
    print("Using 15s delay between requests (free tier safe)")
    print()

    for ticker, name in FAILED_TICKERS.items():
        fix_company(ticker, name)

    print("\n" + "="*55)
    print("FIX COMPLETE")
    print("="*55)
    print("Check data/raw/prices/ — all files should now")
    print("be larger than 10,000 bytes.")
    print("\nNext: python scripts/parse_documents.py")