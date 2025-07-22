import streamlit as st
import random

def build_a_company_game():
    st.title("🏗️ Build-a-Company Tycoon")

    st.subheader("Step 1: Choose Your Company Traits")
    sector = st.selectbox("Pick a Sector", ["AI", "Pharma", "Clean Energy", "Fintech", "Defense"])
    model = st.selectbox("Choose a Business Model", ["Subscription", "Ad-based", "Freemium", "Licensing"])
    risk = st.radio("Pick Risk Level", ["Conservative", "Balanced", "Aggressive", "YOLO"])  

    if st.button("🚀 Simulate My Company!"):
        base_growth = {
            "Conservative": random.uniform(0.05, 0.12),
            "Balanced": random.uniform(0.1, 0.25),
            "Aggressive": random.uniform(0.2, 0.5),
            "YOLO": random.uniform(-0.5, 1.0)
        }[risk]

        macro_events = [
            ("🦠 Pandemic hits", random.uniform(-0.3, -0.05)),
            ("📉 Market crash", random.uniform(-0.4, -0.1)),
            ("🚀 AI boom", random.uniform(0.3, 0.7)),
            ("🏦 Interest rate spike", random.uniform(-0.2, -0.05)),
            ("🌱 ESG explosion", random.uniform(0.1, 0.4)),
            ("⚡ Energy crisis", random.uniform(0.15, 0.3))
        ]
        random.shuffle(macro_events)
        event = random.choice(macro_events)

        final_growth = base_growth + event[1]
        result = f"""### 📈 Results
**Sector:** {sector}  
**Model:** {model}  
**Risk:** {risk}  

{event[0]}

**Total Simulated Return (5 yrs):** `{final_growth:+.2%}`
        """
        st.markdown(result)
        
        if final_growth > 2:
            st.success("You're the next unicorn. Acquired by Google for $4.2B")
        elif final_growth > 0:
            st.info("Solid company. IPO success!")
        else:
            st.error("Your company failed. SEC is now investigating...")