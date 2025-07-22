import streamlit as st
from utils.ai import generate_ai_analysis
from config import RISK_PROFILES
from firebase.db import save_user_recommendations, get_user_recommendations
from utils.pdf import create_recommendations_pdf
from utils.fundamentals_cache import load_cached_fundamentals_firebase, get_fundamental_by_ticker

def recommendations_page():
    st.title("💎 Personalized Recommendations")

    if 'preferences' not in st.session_state or 'user_id' not in st.session_state:
        st.warning("Please complete your investor profile and sign in.")
        return

    user_id = st.session_state.user_id
    profile_type = st.session_state.preferences['profile_type']
    risk_profile = RISK_PROFILES.get(profile_type, RISK_PROFILES["Moderate"])

    st.subheader(f"Recommendations for {profile_type} Investor")
    st.markdown(f"*{risk_profile['description']}*")

    tickers = [t['Symbol'] for t in st.session_state.get('sp500_companies', [])]
    if not tickers:
        import pandas as pd
        tickers = pd.read_csv("sp500_companies.csv")['Symbol'].tolist()
        st.session_state['sp500_companies'] = [{'Symbol': t} for t in tickers]

    if st.button("Generate Recommendations"):
        with st.spinner("Loading cached fundamentals..."):
            fundamentals = load_cached_fundamentals_firebase()
        
        if not fundamentals:
            st.error("No cached data available. Please run nightly precompute script.")
            return
        
        scored_stocks = []
        for ticker in tickers:
            stock = get_fundamental_by_ticker(ticker, fundamentals)
            if not stock:
                continue

            score = 0
            reasons = []

            # P/E ratio check
            max_pe = risk_profile['preferences'].get('max_pe', None)
            if max_pe and stock.get('pe_ratio') is not None:
                if stock['pe_ratio'] <= max_pe:
                    score += 1
                    reasons.append(f"P/E ratio ({stock['pe_ratio']:.1f}) within acceptable range")

            # Dividend yield check
            min_div = risk_profile['preferences'].get('min_dividend', 0)
            if stock.get('dividend_yield', 0) >= min_div:
                score += 1
                reasons.append(f"Dividend yield ({stock['dividend_yield']*100:.2f}%) meets minimum")

            # Sector check
            prefs = risk_profile['preferences']
            excluded_sectors = prefs.get('excluded_sectors', [])
            allowed_sectors = prefs.get('sectors', [])
            sector = stock.get('sector', None)
            if sector:
                if excluded_sectors:
                    if sector not in excluded_sectors:
                        score += 1
                        reasons.append(f"Sector ({sector}) aligns with profile")
                elif allowed_sectors:
                    if sector in allowed_sectors:
                        score += 1
                        reasons.append(f"Sector ({sector}) aligns with profile")

            # Volatility check
            max_vol = prefs.get('max_volatility', None)
            vol = stock.get('volatility', None)
            if max_vol is not None and vol is not None:
                if vol <= max_vol:
                    score += 1
                    reasons.append(f"Volatility ({vol:.2f}) within acceptable range")

            # Market cap check
            min_mc = prefs.get('min_market_cap', 0)
            mc = stock.get('market_cap', 0)
            if mc >= min_mc:
                score += 1
                reasons.append(f"Market cap (${mc/1e9:.1f}B) meets minimum")

            if score > 0:
                scored_stocks.append({
                    **stock,
                    "match_score": score,
                    "match_reasons": reasons
                })

        scored_stocks.sort(key=lambda x: x['match_score'], reverse=True)
        top_recommendations = scored_stocks[:5]
        save_user_recommendations(user_id, top_recommendations)
        st.success("Recommendations generated and saved!")

    # Load saved recommendations
    recommendations = get_user_recommendations(user_id) or []

    if not recommendations:
        st.info("Click 'Generate Recommendations' to see suggestions tailored to you.")
        return

    for stock in recommendations:
        with st.expander(f"⭐ {stock['ticker']} - {stock.get('name', stock['ticker'])} (Match: {stock['match_score']}/5)"):
            st.markdown(f"**Sector:** {stock.get('sector', 'N/A')}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("P/E", f"{stock['pe_ratio']:.1f}" if stock.get('pe_ratio') else "N/A")
            with col2:
                st.metric("Div Yield", f"{stock['dividend_yield']*100:.2f}%" if stock.get('dividend_yield') else "N/A")
            with col3:
                st.metric("Volatility", f"{stock.get('volatility', 'N/A')}")

            st.markdown("**Why it matches your profile:**")
            for reason in stock['match_reasons']:
                st.markdown(f"- {reason}")

            ai_prompt = f"""
            Explain why {stock['ticker']} ({stock.get('name', stock['ticker'])}) is a good match for a {profile_type} investor.
            Consider these factors: {', '.join(stock['match_reasons'])}.
            Keep the analysis concise but insightful (3-5 sentences).
            """
            ai_response = generate_ai_analysis(ai_prompt)
            st.markdown(f"**🧠 AI Insight:** {ai_response}")

            st.markdown("For detailed price trends and charts, please visit the 📊 Research tab.")

    st.subheader("Risk-Return Profile")
    # Assuming plot_risk_return can take the recommendations list
    from utils.visualization import plot_risk_return
    plot_risk_return(recommendations)

    # PDF export button
    pdf_bytes = create_recommendations_pdf(recommendations, profile_type)
    st.download_button(
        label="📄 Download Recommendations PDF",
        data=pdf_bytes,
        file_name=f"recommendations_{profile_type.lower()}.pdf",
        mime="application/pdf"
    )
