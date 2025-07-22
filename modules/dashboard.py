import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.api import fetch_live_market_data, get_sector_performance
from utils.visualization import plot_sector_performance, plot_sector_allocation
from firebase.db import get_user_portfolio
from utils.analysis import analyze_portfolio_detailed

def dashboard_page():
    st.title("📊 Arthenic Dashboard")

    # Market overview
    st.subheader("🌍 Market Overview")
    market_data = fetch_live_market_data()
    cols = st.columns(5)
    for i, (ticker, value) in enumerate(market_data.items()):
        cols[i].metric(ticker, f"${value:,.2f}")

    # Sector performance
    st.subheader("📈 Sector Performance")
    sector_perf = get_sector_performance()
    plot_sector_performance(sector_perf)

    # Portfolio overview
    st.subheader("💼 Your Portfolio")
    portfolio = get_user_portfolio(st.session_state.user_id)
    portfolio_analysis = analyze_portfolio_detailed(portfolio)
    
    if portfolio_analysis:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Value", f"${portfolio_analysis['total_value']:,.2f}")
        col2.metric("YTD Return", f"{portfolio_analysis['performance']['ytd_return']*100:.2f}%")
        col3.metric("Daily Change", f"{portfolio_analysis['performance']['daily_change']*100:.2f}%")

        st.subheader("🧩 Sector Allocation")
        plot_sector_allocation(portfolio_analysis["sector_allocation"])
    else:
        st.info("Your portfolio is empty. Add stocks to track your investments.")