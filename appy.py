import streamlit as st
from firebase.auth import login_page
from modules.auth import onboarding_page

st.set_page_config(layout="wide", page_title="Arthenic", page_icon="📈")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles.css")

def preload_data():
    import pandas as pd
    try:
        if 'sp500_companies' not in st.session_state:
            df = pd.read_csv("sp500_companies.csv")
            st.session_state['sp500_companies'] = df.to_dict(orient="records")
            print("✅ Preloaded S&P 500 companies.")
    except Exception as e:
        print(f"❌ Failed to preload: {e}")

def main():
    preload_data()  
    if "page" not in st.session_state:
        st.session_state.page = "login"

    if "user" not in st.session_state:
        login_page()
        return

    if st.session_state.page == "onboarding":
        onboarding_page()
        return

    # Import modules
    from modules.dashboard import dashboard_page
    from modules.portfolio import portfolio_management_page
    from modules.research import research_page
    from modules.recommendations import recommendations_page
    from modules.watchlist import watchlist_page
    from modules.goals import goal_tracker_page
    from modules.games.tycoon import build_a_company_game
    from modules.games.insider import insider_trader_game

    with st.sidebar:
        nav_choice = st.radio("Navigation", [
            "Dashboard", "Portfolio", "Research", 
            "Recommendations", "Watchlist", "Goal Tracker",
            "Tycoon Game", "Insider Trader"
        ])
        
        if st.button("🎯 Edit Investor Profile"):
            st.session_state.page = "onboarding"
            st.rerun()

        st.markdown("---")
        st.write("🔵 **Arthenic**")
        st.caption("Your Wealth, With Purpose")

    # Route to selected page
    page_map = {
        "Dashboard": dashboard_page,
        "Portfolio": portfolio_management_page,
        "Research": research_page,
        "Recommendations": recommendations_page,
        "Watchlist": watchlist_page,
        "Goal Tracker": goal_tracker_page,
        "Tycoon Game": build_a_company_game,
        "Insider Trader": insider_trader_game
    }
    
    if nav_choice in page_map:
        page_map[nav_choice]()

if __name__ == "__main__":
    main()