import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
from firebase.db import get_user_portfolio, save_user_portfolio
from utils.analysis import analyze_portfolio_detailed
from utils.visualization import plot_sector_allocation
from utils.ai import explain_stock_with_ai, generate_strategic_stock_guidance

def portfolio_management_page():
    st.title("📊 Portfolio Management")

    # --- Add New Stock ---
    st.markdown("### ➕ Add Stock")
    with st.form("add_stock_form"):
        new_ticker = st.text_input("Ticker Symbol").upper().strip()
        shares = st.number_input("Number of Shares", min_value=0.0)
        cost = st.number_input("Cost Basis per Share ($)", min_value=0.0)
        purchase_date = st.date_input("Purchase Date", value=datetime.today())

        submitted = st.form_submit_button("Add Stock")
        if submitted:
            if shares > 0 and cost > 0 and new_ticker:
                portfolio = get_user_portfolio(st.session_state.user_id) or {}
                portfolio[new_ticker] = {
                    "shares": shares,
                    "cost_basis": cost,
                    "purchase_date": str(purchase_date)
                }
                if save_user_portfolio(st.session_state.user_id, portfolio):
                    st.success("Stock added!")
                else:
                    st.error("Failed to update portfolio.")
            else:
                st.warning("Please enter all valid fields.")

    # --- Load Portfolio ---
    portfolio = get_user_portfolio(st.session_state.user_id)
    if not portfolio:
        st.info("Your portfolio is empty. Add stocks to begin analysis.")
        return

    analysis = analyze_portfolio_detailed(portfolio)
    if not analysis:
        st.error("Failed to analyze portfolio.")
        return

    # --- Calculate Invested / Current / Profit ---
    invested = 0.0
    current = 0.0
    for tkr, pos in portfolio.items():
        shares = pos.get("shares", 0)
        cost = pos.get("cost_basis", 0)
        invested += shares * cost
        try:
            current_price = yf.Ticker(tkr).history(period="1d")["Close"].iloc[-1]
        except:
            current_price = cost
        current += shares * current_price

    profit = current - invested
    arrow = "🔺" if profit > 0 else ("🔻" if profit < 0 else "➡️")

    # --- Summary Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Current Value", f"${current:,.2f}")
    col2.metric("📥 Invested", f"${invested:,.2f}")
    if profit > 0:
        arrow = "⬆️"
    elif profit < 0:
        arrow = "⬇️"
    else:
        arrow = "➡️"
    col3.metric("📈 Net Gain", f"{arrow} ${abs(profit):,.2f}")
    col4.metric("📅 YTD Return", f"{analysis['performance']['ytd_return']*100:.2f}%")

    # --- Sector Allocation ---
    st.markdown("### 🧩 Sector Allocation")
    plot_sector_allocation(analysis['sector_allocation'])

    # --- Edit Stock ---
    st.markdown("### ✏️ Edit Stock")
    ticker_to_edit = st.selectbox("Select Ticker to Edit", list(portfolio.keys()))
    pos = portfolio[ticker_to_edit]
    new_shares = st.number_input("Shares", value=pos["shares"], min_value=0.0, key="edit_shares")
    new_cost = st.number_input("Cost Basis", value=pos["cost_basis"], min_value=0.0, key="edit_cost")
    try:
        default_date = pd.to_datetime(pos["purchase_date"])
    except:
        default_date = datetime.today()
    new_date = st.date_input("Purchase Date", value=default_date, key="edit_date")

    if st.button("Save Changes"):
        portfolio[ticker_to_edit] = {
            "shares": new_shares,
            "cost_basis": new_cost,
            "purchase_date": str(new_date)
        }
        if save_user_portfolio(st.session_state.user_id, portfolio):
            st.success("Portfolio updated.")
        else:
            st.error("Failed to update portfolio.")
    # --- Delete Stock ---
    st.markdown("### 🗑️ Delete Stock")
    ticker_to_delete = st.selectbox("Select Ticker to Delete", list(portfolio.keys()), key="delete_ticker")

    if st.button("Delete Selected Stock"):
        confirm = st.checkbox(f"Yes, I want to delete {ticker_to_delete}")
        if confirm:
            try:
                del portfolio[ticker_to_delete]
                if save_user_portfolio(st.session_state.user_id, portfolio):
                    st.success(f"{ticker_to_delete} has been removed from your portfolio.")
                else:
                    st.error("Failed to update portfolio.")
            except Exception as e:
                st.error(f"Error deleting: {e}")
        else:
            st.warning("Please confirm deletion.")

    # --- Stock Breakdown Table ---
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
        trend = "📈" if ret > 0.1 else ("📉" if ret < -0.1 else "➡️")

        stock_rows.append({
            "Ticker": tkr,
            "Shares": shares,
            "Cost Basis": f"${cost:.2f}",
            "Current Price": f"${current_price:.2f}",
            "Current Value": f"${value:.2f}",
            "1Y Return": f"{ret*100:.2f}% {trend}",
            "Purchase Date": pos.get("purchase_date", "N/A")
        })

    st.dataframe(pd.DataFrame(stock_rows))

    # --- AI Summary ---
    st.markdown("### 💡 AI Summary")
    for tkr in portfolio.keys():
        with st.expander(f"📘 {tkr} – What does this stock represent?"):
            try:
                ai_summary = explain_stock_with_ai(tkr)
                st.markdown(ai_summary)
            except:
                st.warning("AI summary not available.")

    # --- AI Strategy Guidance + Sell Target ---
    st.markdown("### 🎯 AI Strategic Guidance + Sell Strategy")
    for tkr, pos in portfolio.items():
        with st.expander(f"🧭 {tkr} – Personalized Plan"):
            try:
                shares = pos.get("shares", 0)
                cost = pos.get("cost_basis", 0)
                try:
                    raw_date = pos.get("purchase_date", "")
                    purchase_date = datetime.strptime(raw_date, "%Y-%m-%d") if raw_date else datetime.today() - pd.DateOffset(months=6)
                except:
                    purchase_date = datetime.today() - pd.DateOffset(months=6)

                strategy = generate_strategic_stock_guidance(tkr, shares, cost, purchase_date)
                st.markdown(strategy)
            except Exception as e:
                st.warning(f"Could not generate strategy for {tkr}: {e}")

    # --- Stock Charts ---
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

    # --- On-Demand Price Check ---
    st.markdown("### 📍 On-Demand Prices")
    tkr = st.selectbox("Choose a stock to check current price:", list(portfolio.keys()), key="price_check")
    if st.button("Check Current Price"):
        try:
            current = yf.Ticker(tkr).info.get("regularMarketPrice")
            st.success(f"💲 {tkr} is currently ${current:.2f}")
        except:
            st.error("Unable to fetch price.")