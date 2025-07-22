import streamlit as st
from firebase.db import save_user_preferences
from config import RISK_PROFILES

def onboarding_page():
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
            
            # Save preferences
            st.session_state.preferences = {
                "age": age,
                "investment_horizon": investment_horizon,
                "primary_goal": primary_goal,
                "risk_tolerance": risk_tolerance,
                "investable_assets": investable_assets,
                "profile_type": profile_type,
                "onboarding_complete": True
            }

            if 'user_id' in st.session_state:
                save_user_preferences(st.session_state.user_id, st.session_state.preferences)

            st.success("Profile saved successfully!")
            st.session_state.page = "dashboard"
            st.rerun()