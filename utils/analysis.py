from collections import defaultdict
import numpy as np
import yfinance as yf
import pandas as pd
import streamlit as st
from utils.api import safe_get_sector
from utils.ai import guess_sector_with_ai

def get_price_data(ticker):
    if "preferences" in st.session_state:
        period = st.session_state.preferences.get("period", "1y")
        interval = st.session_state.preferences.get("interval", "1d")
    else:
        period = "1y"
        interval = "1d"

    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        df = df.dropna(subset=["Close"])
        return df
    except Exception as e:
        print(f"Error fetching price data for {ticker}: {e}")
        return None


def get_stock_analysis(ticker):
    """Comprehensive stock analysis"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Minimal validity check — skip empty/incomplete results
        if not info or 'shortName' not in info or 'sector' not in info:
            return None

        analysis = {
            "ticker": ticker,
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "forward_pe": info.get("forwardPE", None),
            "peg_ratio": info.get("pegRatio", None),
            "dividend_yield": info.get("dividendYield", 0) or 0,
            "payout_ratio": info.get("payoutRatio", None),
            "beta": info.get("beta", None),
            "52_week_high": info.get("fiftyTwoWeekHigh", None),
            "52_week_low": info.get("fiftyTwoWeekLow", None),
            "price_to_book": info.get("priceToBook", None),
            "price_to_sales": info.get("priceToSalesTrailing12Months", None),
            "profit_margins": info.get("profitMargins", None),
            "operating_margins": info.get("operatingMargins", None),
            "return_on_equity": info.get("returnOnEquity", None),
            "revenue_growth": info.get("revenueGrowth", None),
            "earnings_growth": info.get("earningsGrowth", None),
            "current_price": info.get("currentPrice", None),
            "target_mean_price": info.get("targetMeanPrice", None),
            "recommendation": info.get("recommendationKey", None),
            "esg_score": info.get("esgScore", None)
        }

        # Historical data
        hist = stock.history(period="5y")
        if not hist.empty:
            analysis["volatility"] = hist['Close'].pct_change().std() * np.sqrt(252)
            analysis["1y_return"] = hist['Close'].iloc[-1] / hist['Close'].iloc[-252] - 1 if len(hist) > 252 else None
            analysis["3y_cagr"] = (hist['Close'].iloc[-1] / hist['Close'].iloc[-756])**(1/3) - 1 if len(hist) > 756 else None

        # Financials
        try:
            balance_sheet = stock.balance_sheet
            if not balance_sheet.empty:
                debt = balance_sheet.loc['Total Debt'].iloc[0]
                equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                if equity and equity != 0:
                    analysis["debt_to_equity"] = debt / equity
        except:
            pass

        return analysis

    except Exception as e:
        print(f"[ERROR] Failed to analyze {ticker}: {e}")
        return None

def analyze_portfolio_detailed(portfolio: dict):
    """Analyze user's portfolio with performance and risk metrics."""
    if not portfolio:
        return None

    analysis = {
        "total_value": 0.0,
        "sector_allocation": defaultdict(float),
        "performance": {},
        "risk_metrics": {}
    }

    tickers = list(portfolio.keys())
    initial_value = 0.0

    try:
        hist_data = yf.download(
            tickers=tickers,
            period="2y",            # Extended period for more data points
            interval="1d",
            group_by="ticker",
            threads=False,
            progress=False
        )
    except Exception as e:
        st.error(f"⚠️ Failed to fetch price data: {e}")
        return None

    position_values = {}
    for tkr, pos in portfolio.items():
        shares = pos.get("shares", 0)
        cost_basis = pos.get("cost_basis", 0)

        try:
            data = hist_data["Close"][tkr] if isinstance(hist_data.columns, pd.MultiIndex) else hist_data["Close"]
            data = data.dropna()
            current_price = data.iloc[-1]
        except:
            current_price = cost_basis

        position_value = current_price * shares
        position_values[tkr] = position_value
        analysis["total_value"] += position_value
        initial_value += shares * cost_basis

    weighted_returns_list = []

    for tkr, pos in portfolio.items():
        try:
            data = hist_data["Close"][tkr] if isinstance(hist_data.columns, pd.MultiIndex) else hist_data["Close"]
            data = data.dropna()
            daily_ret = data.pct_change().dropna()
        except Exception as e:
            print(f"Error fetching daily returns for {tkr}: {e}")
            daily_ret = pd.Series(dtype=float)

        weight = position_values.get(tkr, 0) / analysis["total_value"] if analysis["total_value"] > 0 else 0

        weighted_ret = daily_ret * weight
        weighted_ret.name = tkr
        weighted_returns_list.append(weighted_ret)

        # Sector allocation
        sector = safe_get_sector(tkr)
        if sector == "Unknown":
            sector = guess_sector_with_ai(tkr)
        clean_sector = sector.strip().title() if sector and sector.strip() else "Other"
        analysis["sector_allocation"][clean_sector] += position_values.get(tkr, 0)

        # 1-year return per stock
        try:
            one_year_return = (data.iloc[-1] - data.iloc[0]) / data.iloc[0]
        except:
            one_year_return = 0.0
        analysis["performance"][f"{tkr}_1y_return"] = one_year_return

    if analysis["total_value"] > 0:
        for sector in analysis["sector_allocation"]:
            analysis["sector_allocation"][sector] /= analysis["total_value"]

    if weighted_returns_list:
        weighted_returns_df = pd.concat(weighted_returns_list, axis=1)
        weighted_returns_df = weighted_returns_df.dropna(how='all')  # drop rows where all tickers are NaN
        portfolio_daily_returns = weighted_returns_df.sum(axis=1).dropna()
    else:
        portfolio_daily_returns = pd.Series(dtype=float)

    if portfolio_daily_returns.empty or len(portfolio_daily_returns) < 20:
        volatility = None
        sharpe_ratio = None
    else:
        volatility = portfolio_daily_returns.std() * np.sqrt(252)  # Annualized volatility
        mean_daily_return = portfolio_daily_returns.mean()
        std_daily_return = portfolio_daily_returns.std()
        sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0

    ytd_return = (analysis["total_value"] - initial_value) / initial_value if initial_value > 0 else 0.0

    analysis["performance"]["ytd_return"] = ytd_return
    analysis["performance"]["daily_change"] = portfolio_daily_returns.mean() if not portfolio_daily_returns.empty else 0.0

    analysis["risk_metrics"] = {
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "beta": "Coming soon"
    }

    return analysis

