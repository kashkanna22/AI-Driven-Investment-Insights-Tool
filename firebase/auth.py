import streamlit as st
from firebase import auth

def login_page():
    st.title("🔐 Arthenic - Login")
    
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
                    st.success("Account created successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Account creation failed: {str(e)}")