import yfinance as yf
import pandas as pd
import requests
import streamlit as st
from config import SECTOR_ETFS, API_KEYS

def fetch_live_market_data():
    """Fetch and format key market indices with readable names and values."""
    indices = {
        "^DJI": "Dow Jones",
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ",
        "^RUT": "Russell 2000",
        "^VIX": "Volatility Index"
    }

    try:
        data = yf.download(list(indices.keys()), period="1d")['Close']
        if data.empty:
            return {name: float('nan') for name in indices.values()}
        latest = data.ffill().iloc[-1].to_dict()
        return {
            indices[ticker]: round(latest.get(ticker, float('nan')), 2)
            for ticker in indices
        }
    except Exception as e:
        print("Market data fetch error:", e)
        return {name: float('nan') for name in indices.values()}

def get_sector_performance():
    """Fetch sector performance data with full names"""
    try:
        data = yf.download(
            list(SECTOR_ETFS.keys()),
            period="1mo",
            progress=False,
            timeout=10
        )
        returns = data['Close'].pct_change().iloc[-1]
        return pd.Series(
            data=returns.values,
            index=[SECTOR_ETFS[ticker] for ticker in returns.index],
            name="Sector Returns"
        ).sort_values(ascending=False)
    except Exception as e:
        st.error(f"Error fetching sector data: {str(e)}")
        return pd.Series(dtype=float)

def fetch_sector_fallback(ticker):
    # Try Alpha Vantage
    try:
        av_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={API_KEYS['ALPHA_VANTAGE_KEY']}"
        response = requests.get(av_url)
        if response.status_code == 200:
            data = response.json()
            if "Sector" in data and data["Sector"]:
                return data["Sector"]
    except Exception as e:
        print(f"[Alpha Vantage Error] {ticker}: {e}")

    # Try Finnhub
    try:
        finnhub_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={API_KEYS['FINNHUB_API_KEY']}"
        response = requests.get(finnhub_url)
        if response.status_code == 200:
            data = response.json()
            if "finnhubIndustry" in data and data["finnhubIndustry"]:
                return data["finnhubIndustry"]
    except Exception as e:
        print(f"[Finnhub Error] {ticker}: {e}")
    return "Unknown"

def safe_get_sector(ticker):
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector")
        if sector:
            return sector
    except Exception as e:
        print(f"[Yahoo .info Error] {ticker}: {e}")

    fallback = fetch_sector_fallback(ticker)
    print(f"[Fallback] {ticker} sector = {fallback}")
    return fallback

def get_sp500_tickers():
    """Get S&P 500 tickers with company names"""
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = table[0]
        return df["Symbol"].tolist(), df["Security"].tolist()
    except:
        # Fallback if Wikipedia fails
        return ["AAPL", "MSFT", "AMZN"], ["Apple", "Microsoft", "Amazon"]