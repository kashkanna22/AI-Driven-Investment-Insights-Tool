import streamlit as st
import datetime

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
            current_progress = st.number_input(
                f"{goal['name']} – Current Saved ($)", 
                min_value=0.0, 
                value=goal["progress"], 
                key=goal['name']
            )
            goal["progress"] = current_progress
            pct = min(current_progress / goal["target"], 1.0)
            st.progress(pct, text=f"{pct*100:.1f}% toward ${goal['target']:,.0f} by {goal['deadline']}")

    else:
        st.info("Add your first goal to begin tracking progress.")