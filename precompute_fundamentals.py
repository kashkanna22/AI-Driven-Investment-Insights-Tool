import yfinance as yf
import firebase_admin
from firebase_admin import credentials, firestore
import time
import pandas as pd

# Initialize Firebase Admin SDK (adjust path to your key file)
if not firebase_admin._apps:
    cred = credentials.Certificate("personal-project-firebasekey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

def fetch_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "ticker": ticker,
        "name": info.get("shortName", ticker),
        "sector": info.get("sector", "N/A"),
        "market_cap": info.get("marketCap", 0),
        "pe_ratio": info.get("trailingPE", None),
        "dividend_yield": info.get("dividendYield", 0),
        # Add any other fields you want here
    }

def save_fundamentals_batch(fundamentals):
    batch = db.batch()
    for stock in fundamentals:
        doc_ref = db.collection("stock_fundamentals").document(stock['ticker'])
        batch.set(doc_ref, stock)
    batch.commit()

def main():
    print("Reading tickers from CSV...")
    tickers = pd.read_csv("sp500_companies.csv")['Symbol'].tolist()

    all_fundamentals = []
    for idx, ticker in enumerate(tickers):
        try:
            data = fetch_fundamentals(ticker)
            all_fundamentals.append(data)
            print(f"[{idx+1}/{len(tickers)}] Fetched {ticker}")
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
        time.sleep(0.1)  # Short delay to avoid API rate limits

    print("Saving batch to Firebase Firestore...")
    save_fundamentals_batch(all_fundamentals)
    print("Upload complete!")

if __name__ == "__main__":
    main()
