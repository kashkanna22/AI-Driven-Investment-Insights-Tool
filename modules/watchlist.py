import streamlit as st
import yfinance as yf
from firebase.db import get_user_watchlist, save_user_watchlist
from modules.research import get_price_data, plot_stock_performance

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = info.get('regularMarketPrice', 
                                 info.get('currentPrice', 
                                          stock.history(period="1d")['Close'].iloc[-1] if not stock.history(period="1d").empty else None))
        
        return {
            'ticker': ticker,
            'name': info.get('shortName', ticker),
            'price': current_price,
            'pe': info.get('trailingPE'),
            'high': info.get('fiftyTwoWeekHigh'),
            'low': info.get('fiftyTwoWeekLow'),
            'valid': True
        }
    except:
        return {
            'ticker': ticker,
            'valid': False
        }

def watchlist_page():
    st.title("👀 Your Watchlist")
    
    if 'user_id' not in st.session_state:
        st.warning("Please login first")
        return
        
    watchlist = get_user_watchlist(st.session_state.user_id)
    
    if not watchlist:
        st.info("Your watchlist is empty. Add stocks to track them here.")
        return

    if st.button("🔄 Refresh All"):
        st.experimental_rerun()

    for ticker in watchlist[:50]:
        stock_data = get_stock_data(ticker)
        
        # Use consistent period & interval for price data & plots
        period = "1y"
        interval = "1d"

        price_data = get_price_data(ticker, period=period, interval=interval)
        
        if not stock_data['valid']:
            st.warning(f"Could not load data for {ticker}")
            if st.button(f"❌ Remove {ticker}", key=f"remove_{ticker}"):
                watchlist.remove(ticker)
                save_user_watchlist(st.session_state.user_id, watchlist)
                st.experimental_rerun()
            continue

        with st.expander(f"{stock_data['ticker']} - {stock_data['name']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Price", f"${stock_data['price']:,.2f}" if stock_data['price'] else "N/A")
                st.metric("P/E", f"{stock_data['pe']:.1f}" if stock_data['pe'] else "N/A")
            
            with col2:
                st.metric("52W High", f"${stock_data['high']:,.2f}" if stock_data['high'] else "N/A")
                st.metric("52W Low", f"${stock_data['low']:,.2f}" if stock_data['low'] else "N/A")
            
            if price_data is not None:
                plot_stock_performance(ticker, period=period, interval=interval)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📈 Research", key=f"research_{ticker}"):
                    st.session_state.research_ticker = ticker
            with col2:
                if st.button("💼 Add to Portfolio", key=f"add_{ticker}"):
                    st.session_state.add_ticker = ticker
            with col3:
                if st.button("❌ Remove", key=f"remove_{ticker}"):
                    watchlist.remove(ticker)
                    save_user_watchlist(st.session_state.user_id, watchlist)
                    st.experimental_rerun()
