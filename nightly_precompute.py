import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor
from utils.analysis import get_stock_analysis
from firebase.db import save_fundamentals_batch

def safe_analysis(ticker):
    try:
        return get_stock_analysis(ticker)
    except Exception as e:
        print(f"Error for {ticker}: {e}")
        return None

def main():
    tickers = pd.read_csv("data/sp500_companies.csv")['Symbol'].tolist()

    print(f"Starting analysis for {len(tickers)} tickers...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(safe_analysis, tickers))

    fundamentals = [r for r in results if r]

    print(f"Saving {len(fundamentals)} fundamentals to Firebase...")
    save_fundamentals_batch(fundamentals)

    # Also save locally as fallback/cache
    with open("data/cached_fundamentals.json", "w") as f:
        json.dump(fundamentals, f)

    print("Precompute done!")

if __name__ == "__main__":
    main()