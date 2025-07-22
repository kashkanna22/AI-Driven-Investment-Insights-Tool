import streamlit as st  # Core library for building interactive web apps
import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations and arrays
import random  # Random number generation
import yfinance as yf  # Financial market data (Yahoo Finance API)
import matplotlib.pyplot as plt  # Static data visualization
import plotly.express as px  # High-level interactive plotting
import plotly.graph_objects as go  # Low-level custom Plotly charts
import pyrebase  # Firebase integration (auth, database, storage)
import os  # Operating system interaction (e.g., environment variables)
import json  # JSON parsing and manipulation
from collections import defaultdict  # Dictionary subclass for default values
from datetime import datetime, timedelta  # Date and time operations
from concurrent.futures import ThreadPoolExecutor, as_completed  # Parallel task execution
from sklearn.preprocessing import MinMaxScaler  # Feature scaling (0 to 1 range)
from pandas_datareader import data as pdr  # Financial data reader (e.g., FRED, IEX)
import finnhub  # Finnhub API client (stock data, news, fundamentals)
import requests  # HTTP requests (for APIs, etc.)
from PIL import Image  # Image processing (open, resize, convert, etc.)
import pytz  # Timezone support
import seaborn as sns  # Statistical data visualization (heatmaps, etc.)
from st_aggrid import AgGrid, GridOptionsBuilder  # Interactive data tables in Streamlit
import tempfile  # Temporary file creation and management
import openai  # OpenAI API access (for GPT models)
from fpdf import FPDF  # Generate PDFs from code
import base64  # Encoding/decoding binary data for web use (e.g., file downloads)
import warnings  # Manage warnings (suppress unnecessary logs)
warnings.filterwarnings('ignore')  # Suppress all warnings for cleaner output

# --- Constants ---
RISK_PROFILES = {
    "Conservative": {
        "description": "Preserve capital with minimal risk",
        "preferences": {
            "max_volatility": 0.2,
            "min_dividend": 0.03,
            "sectors": ["Utilities", "Consumer Defensive", "Healthcare"],
            "max_pe": 20,
            "min_market_cap": 10e9  # $10B+
        }
    },
    "Moderate": {
        "description": "Balance between growth and income",
        "preferences": {
            "max_volatility": 0.3,
            "min_dividend": 0.015,
            "sectors": ["Technology", "Financial Services", "Healthcare"],
            "max_pe": 25,
            "min_market_cap": 2e9  # $2B+
        }
    },
    "Aggressive": {
        "description": "Maximize growth potential",
        "preferences": {
            "max_volatility": 0.5,
            "min_dividend": 0.0,
            "sectors": ["Technology", "Communication Services", "Consumer Cyclical"],
            "max_pe": None,
            "min_market_cap": 500e6  # $500M+
        }
    },
    "Income": {
        "description": "Focus on dividend income",
        "preferences": {
            "max_volatility": 0.25,
            "min_dividend": 0.04,
            "sectors": ["Utilities", "Real Estate", "Energy"],
            "max_pe": 18,
            "min_market_cap": 5e9  # $5B+
        }
    },
    "ESG": {
        "description": "Socially responsible investing",
        "preferences": {
            "max_volatility": 0.35,
            "min_dividend": 0.01,
            "sectors": ["Healthcare", "Technology", "Consumer Defensive"],
            "excluded_sectors": ["Energy", "Tobacco", "Weapons"],
            "max_pe": 30,
            "min_market_cap": 1e9  # $1B+
        }
    }
}

# --- Firebase Config ---
firebase_config = {
    "apiKey": "AIzaSyBhLUNQ9nzm0nadAv-dQ9nEtgJs1Kb8bDY",
    "authDomain": "personal-project-80f32.firebaseapp.com",
    "databaseURL": "https://personal-project-80f32-default-rtdb.firebaseio.com/",
    "projectId": "personal-project-80f32",
    "storageBucket": "personal-project-80f32.appspot.com",
    "messagingSenderId": "969572991303",
    "appId": "1:969572991303:web:2a0c1cab8b06a6645b4662",
    "measurementId": "G-XXXXXXXXXX"
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

# --- API Clients ---
GROQ_API_KEY = "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"
finnhub_client = finnhub.Client(api_key="d0fte1pr01qr6dbu77mgd0fte1pr01qr6dbu77n0")
alpha_vantage_key = "EYK7GNAZP045LRQT"

def fetch_sector_fallback(ticker):
    # Try Alpha Vantage first
    try:
        av_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={alpha_vantage_key}"
        response = requests.get(av_url)
        if response.status_code == 200:
            data = response.json()
            if "Sector" in data and data["Sector"]:
                return data["Sector"]
    except Exception as e:
        print(f"[Alpha Vantage Error] {ticker}: {e}")

    # Try Finnhub next
    try:
        finnhub_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={finnhub_client}"
        response = requests.get(finnhub_url)
        if response.status_code == 200:
            data = response.json()
            if "finnhubIndustry" in data and data["finnhubIndustry"]:
                return data["finnhubIndustry"]
    except Exception as e:
        print(f"[Finnhub Error] {ticker}: {e}")

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

# --- Utility Functions ---
def get_user_portfolio(user_id):
    try:
        return db.child("users").child(user_id).child("portfolio").get().val() or {}
    except:
        return {}

def save_user_portfolio(user_id, portfolio):
    db.child("users").child(user_id).child("portfolio").set(portfolio)

def get_user_watchlist(user_id):
    try:
        return db.child("users").child(user_id).child("watchlist").get().val() or []
    except:
        return []

def save_user_watchlist(user_id, watchlist):
    db.child("users").child(user_id).child("watchlist").set(watchlist)

def get_user_preferences(user_id):
    try:
        return db.child("users").child(user_id).child("preferences").get().val() or {}
    except:
        return {}

def save_user_preferences(user_id, preferences):
    db.child("users").child(user_id).child("preferences").set(preferences)

def format_compact_number(value):
    if value >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value/1_000:.2f}K"
    else:
        return f"${value:.2f}"
    
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
        latest = data.ffill().iloc[-1].to_dict()  # use ffill in case of missing values
        return {
            indices[ticker]: round(latest.get(ticker, float('nan')), 2)
            for ticker in indices
        }
    except Exception as e:
        print("Market data fetch error:", e)
        return {name: float('nan') for name in indices.values()}
    
def get_sector_performance():
    """Fetch sector performance data with full names"""
    sector_etfs = {
        "XLC": "Communication Services",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLE": "Energy",
        "XLF": "Financials",
        "XLV": "Health Care",
        "XLI": "Industrials",
        "XLB": "Materials",
        "XLRE": "Real Estate",
        "XLK": "Technology",
        "XLU": "Utilities"
    }
    
    try:
        # Download data with timeout
        data = yf.download(
            list(sector_etfs.keys()),
            period="1mo",
            progress=False,
            timeout=10
        )
        # Calculate returns and map to full names
        returns = data['Close'].pct_change().iloc[-1]
        return pd.Series(
            data=returns.values,
            index=[sector_etfs[ticker] for ticker in returns.index],
            name="Sector Returns"
        ).sort_values(ascending=False)
        
    except Exception as e:
        st.error(f"Error fetching sector data: {str(e)}")
        return pd.Series(dtype=float)

def plot_sector_performance(sector_perf):
    """Plot sector performance with full names"""
    if sector_perf.empty:
        st.warning("No sector performance data available")
        return
    
    fig = go.Figure(go.Bar(
        x=sector_perf.values,
        y=sector_perf.index,
        orientation='h',
        marker_color=['#4CAF50' if v > 0 else '#F44336' for v in sector_perf.values],
        hovertemplate="<b>%{y}</b><br>Return: %{x:.2%}<extra></extra>"
    ))
    
    fig.update_layout(
        title="1-Month Sector Performance",
        xaxis_title="Return",
        yaxis_title="",
        height=400,
        margin=dict(l=50, r=30, t=50, b=30),
        yaxis={'categoryorder':'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def get_earnings_calendar(days=7):
    """Get upcoming earnings calendar"""
    now = datetime.now()
    to_date = (now + timedelta(days=days)).strftime('%Y-%m-%d')
    return finnhub_client.earnings_calendar(_from=now.strftime('%Y-%m-%d'), to=to_date, symbol="", international=False)

# --- Enhanced Stock Analysis ---
def get_stock_analysis(ticker):
    """Comprehensive stock analysis"""
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Basic info
    analysis = {
        "ticker": ticker,
        "name": info.get("shortName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap", 0),
        "pe_ratio": info.get("trailingPE", None),
        "forward_pe": info.get("forwardPE", None),
        "peg_ratio": info.get("pegRatio", None),
        "dividend_yield": info.get("dividendYield", 0),
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
        analysis["volatility"] = hist['Close'].pct_change().std() * np.sqrt(252)  # Annualized
        analysis["1y_return"] = hist['Close'].iloc[-1] / hist['Close'].iloc[-252] - 1 if len(hist) > 252 else None
        analysis["3y_cagr"] = (hist['Close'].iloc[-1] / hist['Close'].iloc[-756])**(1/3) - 1 if len(hist) > 756 else None
    
    # Financials
    try:
        balance_sheet = stock.balance_sheet
        if not balance_sheet.empty:
            analysis["debt_to_equity"] = balance_sheet.loc['Total Debt'].iloc[0] / balance_sheet.loc['Total Stockholder Equity'].iloc[0]
    except:
        pass
    
    return analysis

# --- Portfolio Analysis ---
def portfolio_management_page(portfolio: dict):
    """Analyze portfolio with fallback-safe sector logic and AI labels."""
    if not portfolio:
        return None

    analysis = {
        "total_value": 0.0,
        "sector_allocation": defaultdict(float),
        "performance": {},
        "risk_metrics": {}
    }

    tickers = list(portfolio.keys())

    try:
        price_df = yf.download(
            tickers=tickers,
            period="1d",
            progress=False,
            group_by="ticker",
            threads=False
        )
    except Exception as e:
        st.error(f"⚠️ Yahoo price fetch failed: {e}")
        price_df = None

    initial_value = 0.0

    for tkr, pos in portfolio.items():
        shares = pos.get("shares", 0)
        cost_basis = pos.get("cost_basis", 0)

        # --- Current Price ---
        price = None
        if price_df is not None:
            try:
                price = (
                    price_df["Close"][tkr].iloc[-1]
                    if isinstance(price_df.columns, pd.MultiIndex)
                    else price_df["Close"].iloc[-1]
                )
            except Exception:
                price = None

        if price is None:
            try:
                price = yf.Ticker(tkr).fast_info["lastPrice"]
            except Exception:
                price = 0.0
                st.warning(f"Price unavailable for {tkr}")

        position_value = price * shares
        analysis["total_value"] += position_value

        # --- Initial Investment (used for YTD) ---
        initial_value += cost_basis * shares

        # --- Sector ---
        sector = safe_get_sector(tkr)
        if sector == "Unknown":
            sector = guess_sector_with_ai(tkr)

        clean_sector = sector.strip().title() if sector and sector.strip() else "Other"
        analysis["sector_allocation"][clean_sector] += position_value

    # Normalize sector allocation
    total = analysis["total_value"]
    if total:
        for sector in analysis["sector_allocation"]:
            analysis["sector_allocation"][sector] /= total

    # --- Performance Metrics ---
    analysis["performance"]["daily_change"] = 0.0  # Not implemented
    if initial_value > 0:
        ytd_return = (analysis["total_value"] - initial_value) / initial_value
    else:
        ytd_return = 0.0
    analysis["performance"]["ytd_return"] = ytd_return

    return analysis

# --- AI Integration ---
def generate_ai_analysis(prompt, context=""):
    """Generate AI analysis using Groq"""
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    full_prompt = f"""
    You are Arthenic, an advanced investment research assistant. 
    Provide detailed, professional analysis for the following request.
    Context: {context}
    
    Request: {prompt}
    
    Respond with:
    1. Key insights (bullet points)
    2. Risks to consider
    3. Recommended action
    4. Additional factors to monitor
    """
    
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.3,
        "max_tokens": 1500
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"

@st.cache_data(ttl=86400)  # cache for 1 day
def guess_sector_with_ai(ticker):
    prompt = f"What is the best short descriptive label or sector category for the ticker {ticker}? Reply with 2-4 words only."
    
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 50
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[AI Sector Guess Fail] {ticker}: {e}")
        return "Other"
# --- Visualization Functions ---
@st.cache_data(ttl=3600)
def get_stock_data(ticker, period):
    data = yf.download(ticker, period=period, progress=False)
    # Flatten MultiIndex columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data

def plot_stock_performance(ticker, period="1y"):
    try:
        data = get_stock_data(ticker, period)
        
        # ✅ SAFE DATA CHECKS
        if data.empty:
            st.warning(f"No data for {ticker}")
            return
            
        if 'Close' not in data.columns:
            st.warning(f"No Close price data for {ticker}")
            return
            
        if data['Close'].isna().all():
            st.warning(f"All Close prices are NaN for {ticker}")
            return
            
        # Clean data
        clean_data = data.dropna(subset=['Close']).copy()
        
        # Market status
        latest_date = clean_data.index[-1].date()
        today = datetime.now().date()
        if latest_date < today:
            st.info(f"Market closed. Last data: {latest_date.strftime('%Y-%m-%d')}")
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=clean_data.index,
            y=clean_data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4')
        ))
        
        # Moving averages
        for window in [50, 200]:
            if len(clean_data) > window:
                ma = clean_data['Close'].rolling(window).mean()
                fig.add_trace(go.Scatter(
                    x=clean_data.index,
                    y=ma,
                    mode='lines',
                    name=f'{window}-Day Moving Aver',
                    line=dict(dash='dot')
                ))
        
        fig.update_layout(
            title=f"{ticker} Price ({period})",
            xaxis_title="Date",
            yaxis_title="USD",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading {ticker}: {str(e)}")
        st.write("Debug data:", data.head())

def plot_sector_allocation(allocation):
    """Pie chart of sector allocation"""
    df = pd.DataFrame.from_dict(allocation, orient='index', columns=['allocation'])
    df = df[df['allocation'] > 0]
    if df.empty:
        st.warning("No valid sector data to display.")
        return

    df = df.sort_values('allocation', ascending=False)
    fig = px.pie(
        df,
        values='allocation',
        names=df.index,
        title="Sector Allocation",
        hole=0.3,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)


def plot_risk_return(stocks):
    """Risk-return scatter plot"""
    df = pd.DataFrame(stocks)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['volatility'],
        y=df['1y_return'],
        mode='markers',
        marker=dict(
            size=df['market_cap'] / df['market_cap'].max() * 30 + 5,
            color=df['pe_ratio'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='P/E Ratio')
        ),
        text=df['ticker'],
        hoverinfo='text+x+y'
    ))
    
    fig.update_layout(
        title="Risk-Return Profile",
        xaxis_title="Volatility (Annualized)",
        yaxis_title="1-Year Return",
        hovermode="closest",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- PDF Report Generation ---
def create_pdf_report(data):
    """Generate PDF report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Arthenic Investment Report", ln=1, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1, align='C')
    pdf.ln(10)
    
    # Portfolio Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Portfolio Summary", ln=1)
    pdf.set_font("Arial", size=12)
    
    if 'portfolio_analysis' in data:
        pa = data['portfolio_analysis']
        pdf.cell(200, 10, txt=f"Total Portfolio Value: ${pa['total_value']:,.2f}", ln=1)
        pdf.ln(5)
        
        # Sector Allocation
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Sector Allocation:", ln=1)
        pdf.set_font("Arial", size=12)
        
        for sector, alloc in pa['sector_allocation'].items():
            pdf.cell(200, 10, txt=f"{sector}: {alloc*100:.1f}%", ln=1)
    
    # Stock Recommendations
    if 'recommendations' in data:
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Stock Recommendations", ln=1)
        pdf.set_font("Arial", size=12)
        
        for rec in data['recommendations']:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt=f"{rec['ticker']} - {rec['name']}", ln=1)
            pdf.set_font("Arial", size=12)
            
            pdf.multi_cell(0, 10, txt=f"Sector: {rec['sector']} | P/E: {rec.get('pe_ratio', 'N/A')} | Dividend: {rec.get('dividend_yield', 0)*100:.2f}%")
            pdf.multi_cell(0, 10, txt=f"AI Analysis: {rec.get('ai_analysis', 'N/A')}")
            pdf.ln(5)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

# --- Streamlit UI Components ---
def login_page():
    """User authentication"""
    st.title("🔐 Arthenic - Login")
    
    if 'user' not in st.session_state:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Login"):
                try:
                    user = auth.sign_in_with_email_and_password(email, password)
                    st.session_state.user = user
                    st.session_state.user_id = user['localId']
                    st.success("Login successful!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Login failed: {str(e)}")
        
        with tab2:
            new_email = st.text_input("New Email", key="signup_email")
            new_pass = st.text_input("New Password", type="password", key="signup_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")
            
            if st.button("Create Account"):
                if new_pass != confirm_pass:
                    st.error("Passwords don't match!")
                else:
                    try:
                        user = auth.create_user_with_email_and_password(new_email, new_pass)
                        st.session_state.user = user
                        st.session_state.user_id = user['localId']
                        
                        # Initialize user data
                        save_user_portfolio(user['localId'], {})
                        save_user_watchlist(user['localId'], [])
                        
                        st.success("Account created successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Account creation failed: {str(e)}")
        
        st.stop()

def save_user_profile(user_id, profile_data):
    db.child("users").child(user_id).child("preferences").set(profile_data)

def onboarding_page():
    """User onboarding to determine investment profile"""
    st.title("🧾 Investor Profile Setup")
    
    with st.form("profile_form"):
        st.subheader("Basic Information")
        age = st.slider("Your Age", 18, 100, 30)
        investment_horizon = st.selectbox(
            "Investment Horizon",
            ["Short-term (1-3 years)", "Medium-term (3-7 years)", "Long-term (7+ years)"]
        )
        
        st.subheader("Investment Goals")
        primary_goal = st.selectbox(
            "Primary Investment Goal",
            ["Capital Preservation", "Income Generation", "Wealth Growth", "ESG/SRI", "Speculative Growth"]
        )
        
        st.subheader("Risk Tolerance")
        risk_tolerance = st.select_slider(
            "Your Risk Tolerance",
            options=["Very Conservative", "Conservative", "Moderate", "Aggressive", "Very Aggressive"]
        )
        
        st.subheader("Financial Situation")
        investable_assets = st.selectbox(
            "Investable Assets",
            ["Under $10,000", "$10,000 - $50,000", "$50,000 - $200,000", "Over $200,000"]
        )
        
        submitted = st.form_submit_button("Save Profile")
        
        if submitted:
            # Determine profile type
            if primary_goal == "Income Generation":
                profile_type = "Income"
            elif primary_goal == "ESG/SRI":
                profile_type = "ESG"
            elif risk_tolerance in ["Aggressive", "Very Aggressive"]:
                profile_type = "Aggressive"
            elif risk_tolerance in ["Conservative", "Very Conservative"]:
                profile_type = "Conservative"
            else:
                profile_type = "Moderate"
            
            # Save preferences to session
            st.session_state.preferences = {
                "age": age,
                "investment_horizon": investment_horizon,
                "primary_goal": primary_goal,
                "risk_tolerance": risk_tolerance,
                "investable_assets": investable_assets,
                "profile_type": profile_type,
                "onboarding_complete": True
            }

            # Optional: Save to Firebase here
            if 'user_id' in st.session_state:
                save_user_profile(st.session_state.user_id, st.session_state.preferences)

            st.success("Profile saved successfully!")
            st.session_state.page = "dashboard"
            st.rerun()

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from st_aggrid import AgGrid

# --- Dashboard Page ---
def dashboard_page():
    st.title("\U0001F4CA Arthenic Dashboard")

    st.subheader("\U0001F30E Market Overview")
    market_data = fetch_live_market_data()
    cols = st.columns(5)
    for i, (ticker, value) in enumerate(market_data.items()):
        cols[i].metric(ticker, f"${value:,.2f}")

    st.subheader("\U0001F4C8 Sector Performance")
    sector_perf = get_sector_performance()
    fig = go.Figure(go.Bar(
        x=sector_perf.values,
        y=sector_perf.index,
        orientation='h',
        marker_color=['green' if x > 0 else 'red' for x in sector_perf.values]
    ))
    fig.update_layout(
        title="1-Month Sector Performance",
        xaxis_title="Return",
        yaxis_title="Sector",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("\U0001F4BC Your Portfolio")
    portfolio = get_user_portfolio(st.session_state.user_id)
    portfolio_analysis = analyze_portfolio_detailed(portfolio)

    if portfolio_analysis:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Value", f"${portfolio_analysis['total_value']:,.2f}")
        col2.metric("YTD Return", f"{portfolio_analysis['performance']['ytd_return']*100:.2f}%")
        col3.metric("Daily Change", f"{portfolio_analysis['performance']['daily_change']*100:.2f}%")

        st.subheader("\U0001F4CA Sector Allocation")
        plot_sector_allocation(portfolio_analysis["sector_allocation"])
    else:
        st.info("Your portfolio is empty. Add stocks to track your investments.")

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
    daily_returns = []

    # Pull historical data (1 year for trailing return)
    try:
        hist_data = yf.download(
            tickers=tickers,
            period="1y",
            interval="1d",
            group_by="ticker",
            threads=False,
            progress=False
        )
    except Exception as e:
        st.error(f"⚠️ Failed to fetch price data: {e}")
        return None

    for tkr, pos in portfolio.items():
        shares = pos.get("shares", 0)
        cost_basis = pos.get("cost_basis", 0)

        try:
            data = hist_data["Close"][tkr] if isinstance(hist_data.columns, pd.MultiIndex) else hist_data["Close"]
            data = data.dropna()
            current_price = data.iloc[-1]
            day_ago_price = data.iloc[-2]
            year_ago_price = data.iloc[0]
        except:
            ticker_obj = yf.Ticker(tkr)
            data = ticker_obj.history(period="1y")
            current_price = data["Close"].iloc[-1] if not data.empty else cost_basis
            day_ago_price = data["Close"].iloc[-2] if len(data) > 1 else cost_basis
            year_ago_price = data["Close"].iloc[0] if len(data) > 0 else cost_basis

        position_value = current_price * shares
        analysis["total_value"] += position_value
        initial_value += shares * cost_basis

        try:
            daily_ret = (current_price - day_ago_price) / day_ago_price
            daily_returns.append(daily_ret)
        except:
            daily_returns.append(0.0)

        sector = safe_get_sector(tkr)
        if sector == "Unknown":
            sector = guess_sector_with_ai(tkr)
        clean_sector = sector.strip().title() if sector and sector.strip() else "Other"
        analysis["sector_allocation"][clean_sector] += position_value

        try:
            one_year_return = (current_price - year_ago_price) / year_ago_price
        except:
            one_year_return = 0.0
        analysis["performance"][f"{tkr}_1y_return"] = one_year_return

    if analysis["total_value"] > 0:
        for sector in analysis["sector_allocation"]:
            analysis["sector_allocation"][sector] /= analysis["total_value"]

    ytd_return = (analysis["total_value"] - initial_value) / initial_value if initial_value > 0 else 0.0
    daily_change = np.mean(daily_returns)

    analysis["performance"]["ytd_return"] = ytd_return
    analysis["performance"]["daily_change"] = daily_change

    volatility = np.std(daily_returns)
    sharpe_ratio = (np.mean(daily_returns) / volatility) * np.sqrt(252) if volatility else 0

    analysis["risk_metrics"] = {
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "beta": "Coming soon"
    }

    return analysis
@st.cache_data(ttl=86400)  # cache for 1 day
def explain_stock_with_ai(ticker):
    """Explain the stock's significance and investment profile using AI."""
    context = f"This is for a user tracking {ticker} in their portfolio."
    prompt = f"Explain what the stock {ticker} represents and its investment outlook. Include valuation, market position, and macro factors if relevant."
    return generate_ai_analysis(prompt, context)

def portfolio_management_page():
    """Full-featured Portfolio Management with performance, risk, insights, trends, and breakdown."""
    st.title("📊 Portfolio Management")

    portfolio = get_user_portfolio(st.session_state.user_id)
    if not portfolio:
        st.info("Your portfolio is empty. Add stocks to begin analysis.")
        return

    analysis = analyze_portfolio_detailed(portfolio)
    if not analysis:
        st.error("Failed to analyze portfolio.")
        return

    # --- Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Value", f"${analysis['total_value']:,.2f}")
    col2.metric("YTD Return", f"{analysis['performance']['ytd_return']*100:.2f}%")
    col3.metric("Daily Change", f"{analysis['performance']['daily_change']*100:.2f}%")

    # --- Sector Allocation Chart ---
    st.markdown("### 🧩 Sector Allocation")
    plot_sector_allocation(analysis['sector_allocation'])

    # --- Risk Metrics ---
    st.markdown("### 🧠 Risk Metrics")
    st.json(analysis["risk_metrics"])

    # --- Individual Stock Breakdown ---
    st.markdown("### 📌 Stock Breakdown")
    stock_rows = []
    for tkr, pos in portfolio.items():
        shares = pos.get("shares", 0)
        cost = pos.get("cost_basis", 0)
        try:
            current_price = yf.Ticker(tkr).history(period="1d")["Close"].iloc[-1]
        except:
            current_price = cost
        value = shares * current_price
        ret = analysis["performance"].get(f"{tkr}_1y_return", 0.0)

        if ret > 0.1:
            trend = "📈"
        elif ret < -0.1:
            trend = "📉"
        else:
            trend = "➡️"

        stock_rows.append({
            "Ticker": tkr,
            "Shares": shares,
            "Cost Basis": f"${cost:.2f}",
            "Current Price": f"${current_price:.2f}",
            "Current Value": f"${value:.2f}",
            "1Y Return": f"{ret*100:.2f}% {trend}"
        })

    st.dataframe(pd.DataFrame(stock_rows))

    # --- AI Insights Per Stock ---
    st.markdown("### 💡 AI Summary")
    for tkr in portfolio.keys():
        with st.expander(f"🧠 {tkr} – What does this signify?"):
            try:
                ai_summary = explain_stock_with_ai(tkr)
                st.write(ai_summary)
            except:
                st.warning("AI summary not available.")

    # --- Price Chart Per Stock ---
    st.markdown("### 📉 Stock Charts")
    for tkr in portfolio.keys():
        with st.expander(f"📊 {tkr} Chart"):
            try:
                df = yf.Ticker(tkr).history(period="6mo")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
                fig.update_layout(title=f"{tkr} Price History", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.warning(f"Chart not available for {tkr}.")

    # --- Current Price Button ---
    st.markdown("### 📍 On-Demand Prices")
    tkr = st.selectbox("Choose a stock to check current price:", list(portfolio.keys()))
    if st.button("Check Current Price"):
        try:
            current = yf.Ticker(tkr).info.get("regularMarketPrice")
            st.success(f"💲 {tkr} is currently ${current:.2f}")
        except:
            st.error("Unable to fetch price.")

def research_page():
    """Stock research interface"""
    st.title("🔍 Stock Research")
    
    # Stock lookup
    ticker = st.text_input("Enter Stock Ticker", key="research_ticker").upper()
    
    if ticker:
        try:
            analysis = get_stock_analysis(ticker)
            
            if not analysis:
                st.error("Could not retrieve data for this ticker")
                return
            
            st.subheader(f"{analysis['name']} ({analysis['ticker']})")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Price", f"${analysis['current_price']:,.2f}" if analysis['current_price'] else "N/A")
                st.metric("Market Cap", f"${analysis['market_cap']/1e9:,.2f}B" if analysis['market_cap'] else "N/A")
            
            with col2:
                st.metric("P/E Ratio", f"{analysis['pe_ratio']:,.2f}" if analysis['pe_ratio'] else "N/A")
                st.metric("Dividend Yield", f"{analysis['dividend_yield']*100:.2f}%" if analysis['dividend_yield'] else "N/A")
            
            with col3:
                st.metric("52 Week Range", 
                         f"${analysis['52_week_low']:,.2f} - ${analysis['52_week_high']:,.2f}" 
                         if analysis['52_week_low'] and analysis['52_week_high'] else "N/A")
                st.metric("Beta", f"{analysis['beta']:,.2f}" if analysis['beta'] else "N/A")
            
            # Performance chart
            plot_stock_performance(ticker)
            
            # AI Analysis
            st.subheader("🧠 AI Analysis")
            ai_prompt = f"""
            Provide a comprehensive analysis of {ticker} considering:
            - Current valuation metrics
            - Growth prospects
            - Competitive position
            - Risks
            - How it fits a {st.session_state.preferences['profile_type']} investor profile
            """
            
            ai_response = generate_ai_analysis(ai_prompt)
            st.markdown(ai_response)
            
            # Add to watchlist
            watchlist = get_user_watchlist(st.session_state.user_id)
            
            if ticker not in watchlist:
                if st.button(f"➕ Add {ticker} to Watchlist"):
                    watchlist.append(ticker)
                    save_user_watchlist(st.session_state.user_id, watchlist)
                    st.success(f"Added {ticker} to watchlist")
            else:
                st.info(f"{ticker} is already in your watchlist")
            
        except Exception as e:
            st.error(f"Error retrieving data: {str(e)}")

def recommendations_page():
    """Personalized stock recommendations"""
    st.title("💎 Personalized Recommendations")
    
    if 'preferences' not in st.session_state:
        st.warning("Please complete your investor profile first")
        return
    
    profile_type = st.session_state.preferences['profile_type']
    risk_profile = RISK_PROFILES.get(profile_type, RISK_PROFILES["Moderate"])
    
    st.subheader(f"Recommendations for {profile_type} Investor")
    st.markdown(f"*{risk_profile['description']}*")
    
    # Get universe of stocks
    if st.button("Generate Recommendations"):
        with st.spinner("Analyzing stocks..."):
            # In production, this would be a larger universe
            sample_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", 
                            "V", "PG", "JNJ", "WMT", "NVDA", "HD", "PYPL", "DIS"]
            
            # Get analysis for each stock
            stocks = []
            for ticker in sample_tickers:
                try:
                    analysis = get_stock_analysis(ticker)
                    if analysis:
                        stocks.append(analysis)
                except:
                    continue
            
            # Score stocks based on profile
            scored_stocks = []
            for stock in stocks:
                score = 0
                reasons = []
                
                # P/E ratio check
                if risk_profile['preferences']['max_pe'] and stock['pe_ratio']:
                    if stock['pe_ratio'] <= risk_profile['preferences']['max_pe']:
                        score += 1
                        reasons.append(f"P/E ratio ({stock['pe_ratio']:.1f}) within acceptable range")
                
                # Dividend yield check
                if stock['dividend_yield'] >= risk_profile['preferences']['min_dividend']:
                    score += 1
                    reasons.append(f"Dividend yield ({stock['dividend_yield']*100:.2f}%) meets minimum")
                
                # Sector check
                if 'excluded_sectors' in risk_profile['preferences']:
                    if stock['sector'] not in risk_profile['preferences']['excluded_sectors']:
                        score += 1
                        reasons.append(f"Sector ({stock['sector']}) aligns with profile")
                elif stock['sector'] in risk_profile['preferences']['sectors']:
                    score += 1
                    reasons.append(f"Sector ({stock['sector']}) aligns with profile")
                
                # Volatility check
                if 'volatility' in stock and stock['volatility'] <= risk_profile['preferences']['max_volatility']:
                    score += 1
                    reasons.append(f"Volatility ({stock['volatility']:.2f}) within acceptable range")
                
                # Market cap check
                if stock['market_cap'] >= risk_profile['preferences']['min_market_cap']:
                    score += 1
                    reasons.append(f"Market cap (${stock['market_cap']/1e9:.1f}B) meets minimum")
                
                if score > 0:
                    scored_stocks.append({
                        **stock,
                        "match_score": score,
                        "match_reasons": reasons
                    })
            
            # Sort by score
            scored_stocks.sort(key=lambda x: x['match_score'], reverse=True)
            
            # Display top recommendations
            for stock in scored_stocks[:5]:
                with st.expander(f"⭐ {stock['ticker']} - {stock['name']} (Match: {stock['match_score']}/5)"):
                    st.markdown(f"**Sector:** {stock['sector']}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("P/E", f"{stock['pe_ratio']:.1f}" if stock['pe_ratio'] else "N/A")
                    
                    with col2:
                        st.metric("Div Yield", f"{stock['dividend_yield']*100:.2f}%" if stock['dividend_yield'] else "N/A")
                    
                    with col3:
                        st.metric("Volatility", f"{stock.get('volatility', 'N/A')}")
                    
                    st.markdown("**Why it matches your profile:**")
                    for reason in stock['match_reasons']:
                        st.markdown(f"- {reason}")
                    
                    # AI analysis
                    ai_prompt = f"""
                    Explain why {stock['ticker']} ({stock['name']}) is a good match for a {profile_type} investor.
                    Consider these factors: {', '.join(stock['match_reasons'])}.
                    Keep the analysis concise but insightful (3-5 sentences).
                    """
                    
                    ai_response = generate_ai_analysis(ai_prompt)
                    st.markdown(f"**🧠 AI Insight:** {ai_response}")
                    
                    plot_stock_performance(stock['ticker'])
            
            # Risk-return visualization
            st.subheader("Risk-Return Profile")
            plot_risk_return(scored_stocks)

def watchlist_page():
    """User watchlist management"""
    st.title("👀 Your Watchlist")
    
    watchlist = get_user_watchlist(st.session_state.user_id)
    
    if watchlist:
        for ticker in watchlist:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                current_price = stock.history(period="1d")['Close'].iloc[-1]
                
                with st.expander(f"{ticker} - {info.get('shortName', ticker)}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Price", f"${current_price:,.2f}")
                        st.metric("P/E", f"{info.get('trailingPE', 'N/A')}")
                    
                    with col2:
                        st.metric("52W High", f"${info.get('fiftyTwoWeekHigh', 'N/A'):,.2f}")
                        st.metric("52W Low", f"${info.get('fiftyTwoWeekLow', 'N/A'):,.2f}")
                    
                    # Quick actions
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"📈 Research {ticker}", key=f"research_{ticker}"):
                            st.session_state.research_ticker = ticker
                            st.switch_page("Research")
                    
                    with col2:
                        if st.button(f"💼 Add to Portfolio", key=f"add_{ticker}"):
                            st.session_state.add_ticker = ticker
                            st.switch_page("Portfolio")
                    
                    with col3:
                        if st.button(f"❌ Remove", key=f"remove_{ticker}"):
                            watchlist.remove(ticker)
                            save_user_watchlist(st.session_state.user_id, watchlist)
                            st.rerun()
                    
                    plot_stock_performance(ticker, period="1y")
            except:
                st.warning(f"Could not load data for {ticker}")
                if st.button(f"❌ Remove {ticker}", key=f"remove_bad_{ticker}"):
                    watchlist.remove(ticker)
                    save_user_watchlist(st.session_state.user)
# -------------------------
# 🎯 Goal Tracker
# -------------------------
def goal_tracker_page():
    st.title("🎯 Investment Goals Tracker")

    if 'goals' not in st.session_state:
        st.session_state.goals = []

    with st.form("add_goal"):
        name = st.text_input("Goal Name")
        target_amount = st.number_input("Target Amount ($)", min_value=100.0, step=100.0)
        deadline = st.date_input("Deadline", value=datetime.date.today() + datetime.timedelta(days=365))
        submitted = st.form_submit_button("Add Goal")

        if submitted and name:
            st.session_state.goals.append({
                "name": name,
                "target": target_amount,
                "deadline": deadline,
                "progress": 0.0
            })
            st.success("Goal added!")

    if st.session_state.goals:
        st.subheader("📈 Your Goals")
        for goal in st.session_state.goals:
            current_progress = st.number_input(f"{goal['name']} – Current Saved ($)", min_value=0.0, value=goal["progress"], key=goal['name'])
            goal["progress"] = current_progress
            pct = min(current_progress / goal["target"], 1.0)
            st.progress(pct, text=f"{pct*100:.1f}% toward ${goal['target']:,.0f} by {goal['deadline']}")

    else:
        st.info("Add your first goal to begin tracking progress.")

# -------------------------
# 🏗️ Build-a-Company Tycoon
# -------------------------
def build_a_company_game():
    st.title("🏗️ Build-a-Company Tycoon")

    st.subheader("Step 1: Choose Your Company Traits")
    sector = st.selectbox("Pick a Sector", ["AI", "Pharma", "Clean Energy", "Fintech", "Defense"])
    model = st.selectbox("Choose a Business Model", ["Subscription", "Ad-based", "Freemium", "Licensing"])
    risk = st.radio("Pick Risk Level", ["Conservative", "Balanced", "Aggressive", "YOLO"])  

    if st.button("🚀 Simulate My Company!"):
        base_growth = {
            "Conservative": random.uniform(0.05, 0.12),
            "Balanced": random.uniform(0.1, 0.25),
            "Aggressive": random.uniform(0.2, 0.5),
            "YOLO": random.uniform(-0.5, 1.0)
        }[risk]

        macro_events = [
            ("🦠 Pandemic hits", random.uniform(-0.3, -0.05)),
            ("📉 Market crash", random.uniform(-0.4, -0.1)),
            ("🚀 AI boom", random.uniform(0.3, 0.7)),
            ("🏦 Interest rate spike", random.uniform(-0.2, -0.05)),
            ("🌱 ESG explosion", random.uniform(0.1, 0.4)),
            ("⚡ Energy crisis", random.uniform(0.15, 0.3))
        ]
        random.shuffle(macro_events)
        event = random.choice(macro_events)

        final_growth = base_growth + event[1]
        result = """### 📈 Results
**Sector:** {}  
**Model:** {}  
**Risk:** {}  

{}

**Total Simulated Return (5 yrs):** `{:+.2%}`
        """.format(sector, model, risk, event[0], final_growth)

        st.markdown(result)
        if final_growth > 2:
            st.success("You're the next unicorn. Acquired by Google for $4.2B")
        elif final_growth > 0:
            st.info("Solid company. IPO success!")
        else:
            st.error("Your company failed. SEC is now investigating...")


# -------------------------
# 🕵️ Insider Trader
# -------------------------
# Set Groq API key
@st.cache_data(ttl=86400)
def get_sp500_tickers():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return table[0]["Symbol"].tolist(), table[0]["Security"].tolist()

@st.cache_data(ttl=86400)
def fetch_company_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName", ticker)
    except:
        return ticker
    
# --- AI Integration with Groq ---
def generate_trade_ai_commentary(company, ticker, user_choice, correct_move, amount, result_profit):
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

    full_prompt = f"""
You are Arthenic, a ruthless and witty AI trading mentor in a high-stakes Wall Street RPG.

Your client just made a trade:
- Company: {company} ({ticker})
- Insider Tip Said: {correct_move}
- Player Chose: {user_choice}
- Amount: ${amount:,}
- Profit/Loss: {'+$' if result_profit > 0 else '-$'}{abs(result_profit):,}

Respond with:
🎯 **Trade Summary**: Give a one-line zinger (funny, brutal, or sarcastic)
🧠 **Verdict**: Smart or Dumb Play (keep it short with 1–2 brutal reasons)
🚨 **SEC Radar**: One-liner reaction from the regulators or the Street
🔮 **Next Move Tease**: End with a cryptic/motivational line teasing their next move

Be ruthless, fast, and fun. Game-style commentary. No essays or boring explanations.
"""

    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.5,
        "max_tokens": 600
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"🧠 AI jammed by SEC: {str(e)}"

# --- S&P 500 Tickers ---
@st.cache_data
def get_sp500_tickers():
    df = pd.read_csv("sp500_companies.csv")
    tickers = df["Symbol"].tolist()
    companies = df["Shortname"].tolist()
    return tickers, companies

# --- Insider Tip Generator ---
def generate_story_tip(company, ticker):
    tips = [
        f"Rumors are swirling that {company} ({ticker}) might beat earnings expectations next week.",
        f"Whispers on the Street suggest a major shake-up at {company} ({ticker}) is coming.",
        f"Insiders have been unusually active at {company} ({ticker}) — something big's about to drop.",
        f"Word is {company} ({ticker}) just signed a huge undisclosed deal behind closed doors.",
        f"There’s buzz that {company} ({ticker}) is about to announce layoffs to cut costs.",
        f"A leaked memo hints that {company} ({ticker}) will miss earnings targets.",
        f"Executives at {company} ({ticker}) are buying stock hand over fist. Confidence, or cover-up?",
        f"{company} ({ticker}) is being quietly investigated by the FTC — mum’s the word.",
        f"A major hedge fund just took a huge short position on {company} ({ticker}).",
        f"Rumors point to {company} ({ticker}) being acquired — insiders are gearing up.",
        f"{company} ({ticker}) is allegedly launching a game-changing product next week.",
        f"Wall Street is betting big against {company} ({ticker})... but the CEO is buying in.",
        f"{company} ({ticker}) just delayed a major product — insiders are scrambling.",
        f"Insiders believe {company} ({ticker}) will crush Q2 forecasts. Load up?",
        f"A controversial exec just returned to {company} ({ticker}) — expect chaos or brilliance.",
        f"Leaked earnings call audio suggests {company} ({ticker}) is hiding losses.",
        f"Whales are moving on {company} ({ticker}) — someone knows something.",
        f"{company} ({ticker}) is rumored to be the target of a cyberattack cover-up.",
        f"Chatter at the country club says {company} ({ticker}) is in bed with regulators.",
        f"{company} ({ticker}) just got a huge government contract — the ink's not even dry.",
        f"A big-name short seller just flipped bullish on {company} ({ticker}). Weird.",
        f"Someone bet $50M in options against {company} ({ticker}) — just a hunch?",
        f"Private equity is sniffing around {company} ({ticker}) like sharks in blood water.",
        f"Tech insiders are claiming {company} ({ticker}) has cracked the next-gen AI code.",
        f"{company} ({ticker}) quietly cut R&D by 40%. Innovation or desperation?",
        f"An insider just dumped their entire stake in {company} ({ticker}) overnight.",
        f"A whistleblower from {company} ({ticker}) just filed with the SEC.",
        f"Goldman Sachs just upgraded {company} ({ticker})... but someone on the floor is shorting.",
        f"{company} ({ticker}) execs are throwing lavish parties — bonus season or last hurrah?",
        f"Reddit is pumping {company} ({ticker}) — is it a trap or a moonshot?",
        f"{company} ({ticker})'s CFO has gone silent. Bad sign? You decide.",
        f"Insiders are betting the farm on {company} ({ticker}) recovering fast.",
        f"An ex-employee says {company} ({ticker}) is cooking the books.",
        f"{company} ({ticker}) is about to drop their long-awaited earnings bombshell.",
        f"A major investor pulled out of {company} ({ticker}) 48 hours ago. Panic or precision?",
        f"{company} ({ticker}) just missed payroll. Execs are calling it a 'glitch.'",
        f"Backroom chatter says {company} ({ticker}) is set to split its stock next week.",
        f"Dark pool volume on {company} ({ticker}) is off the charts today.",
        f"Insider call logs show 37 conversations with SEC lawyers this week. {company} ({ticker}) is sweating.",
        f"{company} ({ticker}) is making hush payments to former employees — what’s the secret?",
        f"Sudden uptick in unusual options volume at {company} ({ticker}). Someone’s in the know.",
        f"Private jet logs show execs from {company} ({ticker}) meeting with rivals.",
        f"{company} ({ticker}) stock was just added to a mysterious billionaire’s fund.",
        f"An activist investor is trying to replace the CEO of {company} ({ticker}).",
        f"Leaked internal memo: '{company} ({ticker}) is in survival mode.'",
        f"{company} ({ticker}) just bought a crypto startup. Boom or bust?",
        f"The CEO of {company} ({ticker}) just rage-quit a shareholder meeting.",
        f"Rumor mill says {company} ({ticker}) will announce a massive buyback next week.",
        f"Insiders are shorting {company} ({ticker}) while promoting it publicly.",
        f"{company} ({ticker})'s earnings report was mysteriously pushed back. Never good."
    ]
    return random.choice(tips)

# --- Game Logic ---
def insider_trader_game():
    st.title("🕵️ Insider Trader")

    keys = ['started', 'money', 'score', 'round', 'persona', 'current_tip', 'current_ticker',
            'current_company', 'game_log', 'locked']
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = False if k == 'started' else (0 if k in ['money', 'score', 'round'] else None)

    if st.session_state.locked:
        st.error("🚨 You've been permanently banned from trading. The SEC has you on speed dial.")
        return

    # Persona Setup
    if not st.session_state.started:
        st.markdown("## Choose Your Insider Persona")
        persona = st.radio("How did you get your money?", [
            "Street Hustler ($10M)", "Startup Mogul ($50M)", "Hedge Fund Baby ($100M)", "Oil Heir ($500M)"
        ])
        if st.button("💼 Enter Game"):
            persona_map = {
                "Street Hustler ($10M)": 10_000_000,
                "Startup Mogul ($50M)": 50_000_000,
                "Hedge Fund Baby ($100M)": 100_000_000,
                "Oil Heir ($500M)": 500_000_000
            }
            st.session_state.money = persona_map[persona]
            st.session_state.persona = persona.split(" ($")[0]
            st.session_state.started = True
            st.session_state.score = 0
            st.session_state.round = 0
            st.session_state.game_log = []
            st.rerun()
        return

    # Generate New Tip
    tickers, names = get_sp500_tickers()
    if st.session_state.current_tip is None:
        idx = random.randint(0, len(tickers) - 1)
        ticker = tickers[idx]
        name = names[idx]
        st.session_state.current_ticker = ticker
        st.session_state.current_company = name
        st.session_state.current_tip = generate_story_tip(name, ticker)

    # Show Tip
    st.subheader(f"💬 Insider Tip #{st.session_state.round + 1}")
    st.markdown(f"*{st.session_state.current_tip}*")

    choice = st.radio("Your move:", ["Buy", "Sell", "Hold"], key=f"choice_{st.session_state.round}")
    amount = st.radio("Trade amount:", ["$1M", "$5M", "$10M", "$25M"], key=f"amt_{st.session_state.round}")
    amt = int(amount.replace("$", "").replace("M", "000000"))

    if st.button("🎯 Execute Trade"):
        if choice != "Hold" and amt > st.session_state.money:
            st.warning("You can’t trade more than your net worth.")
            return

        result = random.choice(["Buy", "Sell"])
        multiplier = random.uniform(0.6, 1.4)
        change = int(amt * multiplier)
        gain_loss = change if choice == result else (-change if choice != "Hold" else 0)

        st.session_state.money += gain_loss
        if choice == result:
            st.session_state.score += 1

        ai_story = generate_trade_ai_commentary(
            st.session_state.current_company,
            st.session_state.current_ticker,
            choice,
            result,
            amt,
            gain_loss
        )

        st.session_state.game_log.append({
            "round": st.session_state.round + 1,
            "tip": st.session_state.current_tip,
            "choice": choice,
            "amount": amt,
            "result": result,
            "change": gain_loss,
            "ai_story": ai_story
        })

        # Next Round
        st.session_state.round += 1
        st.session_state.current_tip = None

        # Win/Lose
        if st.session_state.money >= 1_000_000_000:
            st.success("🏝️ You hit $1B. Time to fake your death or flee to Dubai.")
            return
        elif st.session_state.money <= 0:
            st.session_state.locked = True
            st.error("🧯 Your empire collapsed. You’re banned.")
            return

        st.rerun()

    # Stats
    st.markdown(f"### 💰 Net Worth: ${st.session_state.money:,}")
    st.markdown(f"### ✅ Score: {st.session_state.score} correct trades")

    # Show Game Log Immediately After Trade
    if st.session_state.game_log:
        last_log = st.session_state.game_log[-1]
        st.markdown("---")
        st.markdown(f"## 🎬 Round {last_log['round']} Recap")
        st.markdown(f"**Tip:** {last_log['tip']}")
        st.markdown(f"**Your Move:** {last_log['choice']} ${last_log['amount']:,}")
        st.markdown(f"**Correct Move:** {last_log['result']} | {'Gain' if last_log['change'] > 0 else 'Loss'}: ${abs(last_log['change']):,}")
        st.markdown(last_log['ai_story'])
        st.markdown("---")

    # Full History
    if st.checkbox("📜 Show Full Trade History"):
        for log in st.session_state.game_log[::-1]:
            st.markdown(f"#### Round {log['round']}")
            st.markdown(f"**Tip:** {log['tip']}")
            st.markdown(f"**Move:** {log['choice']} ${log['amount']:,}")
            st.markdown(f"**Result:** {log['result']} | {'Gain' if log['change'] > 0 else 'Loss'}: ${abs(log['change']):,}")
            st.markdown(log['ai_story'])
            st.markdown("---")

    # Restart Game
    if st.button("🔁 Play Again"):
        for key in list(st.session_state.keys()):
            if key not in ['persona']:
                del st.session_state[key]
        st.session_state.started = True
        st.rerun()
# --- Main App Navigation ---
def main():
    if "page" not in st.session_state:
        st.session_state.page = "login"

    if "user" not in st.session_state:
        login_page()
        return

    if st.session_state.page == "onboarding":
        onboarding_page()
        return

    with st.sidebar:
        nav_choice = st.radio("Navigation", [
            "Dashboard", 
            "Portfolio", 
            "Research", 
            "Recommendations", 
            "Watchlist", 
            "Goal Tracker",
            "Tycoon Game",       # ✅ NEW
            "Insider Trader"        # ✅ NEW
        ])
        if st.button("🎯 Edit Investor Profile"):
            st.session_state.page = "onboarding"
            st.rerun()

        st.markdown("---")
        st.write("🔵 **Arthenic**")
        st.caption("Your Wealth, With Purpose")

    if nav_choice == "Dashboard":
        dashboard_page()
    elif nav_choice == "Portfolio":
        portfolio_management_page()
    elif nav_choice == "Research":
        research_page()
    elif nav_choice == "Recommendations":
        recommendations_page()
    elif nav_choice == "Watchlist":
        watchlist_page()
    elif nav_choice == "Goal Tracker":
        goal_tracker_page()
    elif nav_choice == "Tycoon Game":
        build_a_company_game()
    elif nav_choice == "Insider Trader":
        insider_trader_game()

if __name__ == "__main__":
    main()

