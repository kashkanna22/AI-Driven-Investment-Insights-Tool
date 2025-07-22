import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime  # For date handling
from config import SECTOR_ETFS
from utils.analysis import get_price_data

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

def plot_stock_performance(ticker):
    if "preferences" not in st.session_state:
        st.session_state.preferences = {"period": "1y", "interval": "1d"}

    period = st.session_state.preferences["period"]
    interval = st.session_state.preferences["interval"]

    data = get_price_data(ticker, period=period, interval=interval)

    if data.empty:
        st.warning(f"No valid data available for {ticker}")
        return

    if 'Close' not in data.columns:
        st.warning(f"No Close price data for {ticker}")
        return

    latest_date = data.index[-1].date()
    today = datetime.now().date()
    if latest_date < today:
        st.info(f"Market closed. Last data: {latest_date.strftime('%Y-%m-%d')}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Price',
        line=dict(color='#1f77b4')
    ))

    for window in [50, 200]:
        if len(data) > window:
            ma = data['Close'].rolling(window).mean()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=ma,
                mode='lines',
                name=f'{window}-Day MA',
                line=dict(dash='dot')
            ))

    fig.update_layout(
        title=f"{ticker} Price ({period})",
        xaxis_title="Date",
        yaxis_title="USD",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_sector_allocation(allocation):
    """Pie chart of sector allocation"""
    df = pd.DataFrame.from_dict(allocation, orient='index', columns=['allocation'])
    df = df[df['allocation'] > 0]
    if df.empty:
        st.warning("No valid sector data to display.")
        return

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