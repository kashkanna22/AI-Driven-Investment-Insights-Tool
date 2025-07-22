import random
import streamlit as st
from utils.api import get_sp500_tickers
from utils.ai import generate_trade_ai_commentary

def insider_trader_game():
    st.title("🕵️ Insider Trader")
    
    # Game state initialization
    keys = ['started', 'money', 'score', 'round', 'persona', 'current_tip', 
            'current_ticker', 'current_company', 'game_log', 'locked']
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = False if k == 'started' else (0 if k in ['money', 'score', 'round'] else None)

    if st.session_state.locked:
        st.error("🚨 You've been permanently banned from trading. The SEC has you on speed dial.")
        return

    # Persona Setup
    if not st.session_state.started:
        st.markdown("## Choose Your Insider Persona")
        persona = st.radio("How did you get your money?", [
            "Street Hustler ($10M)", "Startup Mogul ($50M)", "Hedge Fund Baby ($100M)", "Oil Heir ($500M)"
        ])
        if st.button("💼 Enter Game"):
            persona_map = {
                "Street Hustler ($10M)": 10_000_000,
                "Startup Mogul ($50M)": 50_000_000,
                "Hedge Fund Baby ($100M)": 100_000_000,
                "Oil Heir ($500M)": 500_000_000
            }
            st.session_state.money = persona_map[persona]
            st.session_state.persona = persona.split(" ($")[0]
            st.session_state.started = True
            st.session_state.score = 0
            st.session_state.round = 0
            st.session_state.game_log = []
            st.rerun()
        return

    # Generate New Tip
    tickers, names = get_sp500_tickers()
    if st.session_state.current_tip is None:
        idx = random.randint(0, len(tickers) - 1)
        ticker = tickers[idx]
        name = names[idx]
        st.session_state.current_ticker = ticker
        st.session_state.current_company = name
        st.session_state.current_tip = generate_story_tip(name, ticker)

    # Show Tip
    st.subheader(f"💬 Insider Tip #{st.session_state.round + 1}")
    st.markdown(f"*{st.session_state.current_tip}*")

    choice = st.radio("Your move:", ["Buy", "Sell", "Hold"], key=f"choice_{st.session_state.round}")
    amount = st.radio("Trade amount:", ["$1M", "$5M", "$10M", "$25M"], key=f"amt_{st.session_state.round}")
    amt = int(amount.replace("$", "").replace("M", "000000"))

    if st.button("🎯 Execute Trade"):
        if choice != "Hold" and amt > st.session_state.money:
            st.warning("You can't trade more than your net worth.")
            return

        result = random.choice(["Buy", "Sell"])
        multiplier = random.uniform(0.6, 1.4)
        change = int(amt * multiplier)
        gain_loss = change if choice == result else (-change if choice != "Hold" else 0)

        st.session_state.money += gain_loss
        if choice == result:
            st.session_state.score += 1

        ai_story = generate_trade_ai_commentary(
            st.session_state.current_company,
            st.session_state.current_ticker,
            choice,
            result,
            amt,
            gain_loss
        )

        st.session_state.game_log.append({
            "round": st.session_state.round + 1,
            "tip": st.session_state.current_tip,
            "choice": choice,
            "amount": amt,
            "result": result,
            "change": gain_loss,
            "ai_story": ai_story
        })

        # Next Round
        st.session_state.round += 1
        st.session_state.current_tip = None

        # Win/Lose
        if st.session_state.money >= 1_000_000_000:
            st.success("🏝️ You hit $1B. Time to fake your death or flee to Dubai.")
            return
        elif st.session_state.money <= 0:
            st.session_state.locked = True
            st.error("🧯 Your empire collapsed. You're banned.")
            return

        st.rerun()

    # Stats
    st.markdown(f"### 💰 Net Worth: ${st.session_state.money:,}")
    st.markdown(f"### ✅ Score: {st.session_state.score} correct trades")

    # Show Game Log
    if st.session_state.game_log:
        last_log = st.session_state.game_log[-1]
        st.markdown("---")
        st.markdown(f"## 🎬 Round {last_log['round']} Recap")
        st.markdown(f"**Tip:** {last_log['tip']}")
        st.markdown(f"**Your Move:** {last_log['choice']} ${last_log['amount']:,}")
        st.markdown(f"**Correct Move:** {last_log['result']} | {'Gain' if last_log['change'] > 0 else 'Loss'}: ${abs(last_log['change']):,}")
        st.markdown(last_log['ai_story'])
        st.markdown("---")

    # Full History
    if st.checkbox("📜 Show Full Trade History"):
        for log in st.session_state.game_log[::-1]:
            st.markdown(f"#### Round {log['round']}")
            st.markdown(f"**Tip:** {log['tip']}")
            st.markdown(f"**Move:** {log['choice']} ${log['amount']:,}")
            st.markdown(f"**Result:** {log['result']} | {'Gain' if log['change'] > 0 else 'Loss'}: ${abs(log['change']):,}")
            st.markdown(log['ai_story'])
            st.markdown("---")

    # Restart Game
    if st.button("🔁 Play Again"):
        for key in list(st.session_state.keys()):
            if key not in ['persona']:
                del st.session_state[key]
        st.session_state.started = True
        st.rerun()

def generate_story_tip(company, ticker):
    tips = [
        f"Rumors are swirling that {company} ({ticker}) might beat earnings expectations next week.",
        f"Whispers on the Street suggest a major shake-up at {company} ({ticker}) is coming.",
        f"Insiders have been unusually active at {company} ({ticker}) — something big's about to drop.",
        f"Word is {company} ({ticker}) just signed a huge undisclosed deal behind closed doors.",
        f"There’s buzz that {company} ({ticker}) is about to announce layoffs to cut costs.",
        f"A leaked memo hints that {company} ({ticker}) will miss earnings targets.",
        f"Executives at {company} ({ticker}) are buying stock hand over fist. Confidence, or cover-up?",
        f"{company} ({ticker}) is being quietly investigated by the FTC — mum’s the word.",
        f"A major hedge fund just took a huge short position on {company} ({ticker}).",
        f"Rumors point to {company} ({ticker}) being acquired — insiders are gearing up.",
        f"{company} ({ticker}) is allegedly launching a game-changing product next week.",
        f"Wall Street is betting big against {company} ({ticker})... but the CEO is buying in.",
        f"{company} ({ticker}) just delayed a major product — insiders are scrambling.",
        f"Insiders believe {company} ({ticker}) will crush Q2 forecasts. Load up?",
        f"A controversial exec just returned to {company} ({ticker}) — expect chaos or brilliance.",
        f"Leaked earnings call audio suggests {company} ({ticker}) is hiding losses.",
        f"Whales are moving on {company} ({ticker}) — someone knows something.",
        f"{company} ({ticker}) is rumored to be the target of a cyberattack cover-up.",
        f"Chatter at the country club says {company} ({ticker}) is in bed with regulators.",
        f"{company} ({ticker}) just got a huge government contract — the ink's not even dry.",
        f"A big-name short seller just flipped bullish on {company} ({ticker}). Weird.",
        f"Someone bet $50M in options against {company} ({ticker}) — just a hunch?",
        f"Private equity is sniffing around {company} ({ticker}) like sharks in blood water.",
        f"Tech insiders are claiming {company} ({ticker}) has cracked the next-gen AI code.",
        f"{company} ({ticker}) quietly cut R&D by 40%. Innovation or desperation?",
        f"An insider just dumped their entire stake in {company} ({ticker}) overnight.",
        f"A whistleblower from {company} ({ticker}) just filed with the SEC.",
        f"Goldman Sachs just upgraded {company} ({ticker})... but someone on the floor is shorting.",
        f"{company} ({ticker}) execs are throwing lavish parties — bonus season or last hurrah?",
        f"Reddit is pumping {company} ({ticker}) — is it a trap or a moonshot?",
        f"{company} ({ticker})'s CFO has gone silent. Bad sign? You decide.",
        f"Insiders are betting the farm on {company} ({ticker}) recovering fast.",
        f"An ex-employee says {company} ({ticker}) is cooking the books.",
        f"{company} ({ticker}) is about to drop their long-awaited earnings bombshell.",
        f"A major investor pulled out of {company} ({ticker}) 48 hours ago. Panic or precision?",
        f"{company} ({ticker}) just missed payroll. Execs are calling it a 'glitch.'",
        f"Backroom chatter says {company} ({ticker}) is set to split its stock next week.",
        f"Dark pool volume on {company} ({ticker}) is off the charts today.",
        f"Insider call logs show 37 conversations with SEC lawyers this week. {company} ({ticker}) is sweating.",
        f"{company} ({ticker}) is making hush payments to former employees — what’s the secret?",
        f"Sudden uptick in unusual options volume at {company} ({ticker}). Someone’s in the know.",
        f"Private jet logs show execs from {company} ({ticker}) meeting with rivals.",
        f"{company} ({ticker}) stock was just added to a mysterious billionaire’s fund.",
        f"An activist investor is trying to replace the CEO of {company} ({ticker}).",
        f"Leaked internal memo: '{company} ({ticker}) is in survival mode.'",
        f"{company} ({ticker}) just bought a crypto startup. Boom or bust?",
        f"The CEO of {company} ({ticker}) just rage-quit a shareholder meeting.",
        f"Rumor mill says {company} ({ticker}) will announce a massive buyback next week.",
        f"Insiders are shorting {company} ({ticker}) while promoting it publicly.",
        f"{company} ({ticker})'s earnings report was mysteriously pushed back. Never good."
    ]
    return random.choice(tips)