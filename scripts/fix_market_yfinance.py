"""
Re-collect market data using yfinance.
No API key needed. No rate limits.
Overwrites all existing market JSON files.
Run: python scripts/fix_market_yfinance.py
"""

import json
import time
import yfinance as yf
from pathlib import Path

Path("data/raw/prices").mkdir(parents=True, exist_ok=True)

COMPANIES = {
    "JPM":  "JPMorgan Chase",
    "GS":   "Goldman Sachs",
    "C":    "Citigroup",
    "BAC":  "Bank of America",
    "BK":   "BNY Mellon",
    "MS":   "Morgan Stanley",
    "WFC":  "Wells Fargo",
    "MSFT": "Microsoft",
    "GOOGL":"Alphabet",
    "NVDA": "NVIDIA",
    "META": "Meta",
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "XOM":  "ExxonMobil",
    "CVX":  "Chevron",
    "COP":  "ConocoPhillips",
    "HSBC": "HSBC Holdings",
    "V":    "Visa",
    "MA":   "Mastercard",
}

print("="*55)
print("MARKET DATA COLLECTION (yfinance)")
print("="*55)
print(f"Companies: {len(COMPANIES)}")
print()

success = 0
failed  = []

for ticker, name in COMPANIES.items():
    print(f"{name} ({ticker})...")

    try:
        stock = yf.Ticker(ticker)

        # Company info
        info = stock.info
        overview = {
            "name":           info.get("longName", name),
            "sector":         info.get("sector", ""),
            "industry":       info.get("industry", ""),
            "description":    info.get("longBusinessSummary", "")[:500],
            "market_cap":     info.get("marketCap"),
            "pe_ratio":       info.get("trailingPE"),
            "eps":            info.get("trailingEps"),
            "revenue_ttm":    info.get("totalRevenue"),
            "profit_margin":  info.get("profitMargins"),
            "52_week_high":   info.get("fiftyTwoWeekHigh"),
            "52_week_low":    info.get("fiftyTwoWeekLow"),
            "analyst_target": info.get("targetMeanPrice"),
            "beta":           info.get("beta"),
            "exchange":       info.get("exchange", ""),
            "currency":       info.get("currency", "USD"),
            "country":        info.get("country", ""),
            "employees":      info.get("fullTimeEmployees"),
            "website":        info.get("website", ""),
        }
        print(f"  ✅ Overview: {overview.get('sector', 'N/A')}")

        # 5 years of monthly price history
        hist = stock.history(period="5y", interval="1mo")
        prices = {}
        for date, row in hist.iterrows():
            date_str = str(date)[:10]
            prices[date_str] = {
                "open":   round(float(row.get("Open", 0)), 2),
                "high":   round(float(row.get("High", 0)), 2),
                "low":    round(float(row.get("Low", 0)), 2),
                "close":  round(float(row.get("Close", 0)), 2),
                "volume": int(row.get("Volume", 0)),
            }
        print(f"  ✅ Prices: {len(prices)} months")

        # Annual financials
        try:
            inc = stock.financials
            fin_annual = {}
            if inc is not None and not inc.empty:
                for col in inc.columns[:4]:
                    year = str(col)[:10]
                    fin_annual[year] = {}
                    for row_name in inc.index:
                        val = inc.loc[row_name, col]
                        try:
                            fin_annual[year][str(row_name)] = \
                                float(val) if val == val else None
                        except Exception:
                            pass
            print(f"  ✅ Financials: {len(fin_annual)} years")
        except Exception:
            fin_annual = {}

        # Save
        company_data = {
            "ticker":     ticker,
            "name":       name,
            "overview":   overview,
            "prices":     prices,
            "financials": fin_annual,
        }

        path = f"data/raw/prices/{ticker}_market.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(company_data, f,
                      ensure_ascii=False, indent=2,
                      default=str)

        size = Path(path).stat().st_size
        print(f"  ✅ Saved: {size:,} bytes")
        success += 1

    except Exception as e:
        print(f"  ❌ Failed: {str(e)[:80]}")
        failed.append(ticker)

    time.sleep(1)

print(f"\n{'='*55}")
print(f"COMPLETE: {success}/{len(COMPANIES)} companies")
if failed:
    print(f"Failed: {failed}")
print(f"\nNext: python scripts/parse_documents.py")