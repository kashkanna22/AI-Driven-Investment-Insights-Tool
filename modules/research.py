import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, time as dt_time
import pytz
import time

# Define your timeframe mapping
TIMEFRAMES = {
    "1 Day": ("1d", "5m"),
    "1 Week": ("5d", "15m"),
    "1 Month": ("1mo", "30m"),
    "3 Months": ("3mo", "1h"),
    "6 Months": ("6mo", "1d"),
    "1 Year": ("1y", "1d"),
    "5 Years": ("5y", "1mo"),
    "All Time": ("max", "1mo"),
}

def is_market_open():
    ny_time = datetime.now(pytz.timezone("America/New_York"))
    return (
        ny_time.weekday() < 5 and 
        dt_time(9, 30) <= ny_time.time() <= dt_time(16, 0)
    )

def get_price_data(ticker, period, interval):
    for attempt in range(3):
        try:
            raw = yf.download(ticker, period=period, interval=interval, progress=False)
            if raw.empty:
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                if ('Close', ticker) in raw.columns:
                    data = raw[('Close', ticker)].to_frame(name='Close')
                else:
                    continue
            else:
                if ticker in raw.columns:
                    data = raw[[ticker]].rename(columns={ticker: 'Close'})
                elif 'Close' in raw.columns:
                    data = raw[['Close']]
                elif 'Adj Close' in raw.columns:
                    data = raw[['Adj Close']].rename(columns={'Adj Close': 'Close'})
                else:
                    continue

            return data.dropna()

        except Exception as e:
            if attempt == 2:
                st.error(f"Failed to fetch data after 3 attempts: {str(e)}")
            time.sleep(1)
    return None

def plot_stock_performance(ticker, period, interval):
    if not is_market_open() and interval in ["5m", "15m", "30m", "1h"]:
        st.info("Market closed - showing daily data instead")
        interval = "1d"

    data = get_price_data(ticker, period, interval)
    if data is None or data.empty or 'Close' not in data.columns:
        st.warning(f"No valid price data available for {ticker}")
        return

    data.index = pd.to_datetime(data.index)
    data = data[data.index <= datetime.now()]
    data = data.dropna(subset=["Close"])

    if data.empty:
        st.warning("No valid historical data to show.")
        return

    latest_date = data.index[-1]
    st.caption(f"Last available data: {latest_date.strftime('%Y-%m-%d %H:%M')} | Interval: {interval}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data["Close"],
        mode='lines+markers',
        name="Price",
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=6)
    ))

    for window in [50, 200]:
        if len(data) >= window:
            ma = data["Close"].rolling(window).mean()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=ma,
                mode='lines',
                name=f"{window}-Day MA",
                line=dict(dash='dot', width=2)
            ))

    y_min = data["Close"].min() * 0.98
    y_max = data["Close"].max() * 1.02

    fig.update_layout(
        title=f"{ticker} Price Chart ({period})",
        xaxis_title="Date",
        yaxis_title="USD",
        hovermode="x unified",
        xaxis=dict(
            range=[data.index.min(), data.index.max()],
            tickformat="%b %d\n%Y"
        ),
        yaxis=dict(range=[y_min, y_max]),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

def research_page():
    st.title("🔬 Research")

    ticker = st.text_input("Enter Stock Ticker", "LULU").upper()

    if ticker:
        stock = yf.Ticker(ticker)
        try:
            info = stock.info
            st.subheader(f"{info.get('longName', ticker)} ({info.get('symbol', ticker)})")

            col1, col2, col3 = st.columns(3)
            col1.metric("Price", f"${info.get('currentPrice', 'N/A')}")
            col2.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
            col3.metric("52 Week Range", f"${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}")

            col4, col5, col6 = st.columns(3)
            col4.metric("Market Cap", f"${round(info.get('marketCap', 0) / 1e9, 2)}B")
            col5.metric("Dividend Yield", f"{round(info.get('dividendYield', 0)*100, 2)}%" if info.get("dividendYield") else "N/A")
            col6.metric("Beta", f"{info.get('beta', 'N/A')}")

            st.subheader("Select Timeframe")

            # Use radio buttons for timeframe selection
            timeframe = st.radio("", list(TIMEFRAMES.keys()), index=5)

            period, interval = TIMEFRAMES[timeframe]

            plot_stock_performance(ticker, period, interval)

        except Exception as e:
            st.error(f"Error retrieving stock data: {str(e)}")

        # AI analysis section as before, unchanged
        st.subheader("\U0001F9E0 AI Analysis")
        try:
            from utils.ai import generate_ai_analysis
            ai_output = generate_ai_analysis(ticker, TIMEFRAMES[timeframe][0])
            st.success(ai_output)
        except Exception as e:
            st.error(f"Error retrieving AI analysis: {str(e)}")

if __name__ == "__main__":
    research_page()
