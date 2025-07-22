import requests
from datetime import datetime
from config import API_KEYS
import yfinance as yf

GROQ_HEADERS = {
    "Authorization": f"Bearer {API_KEYS['GROQ_API_KEY']}",
    "Content-Type": "application/json"
}


def generate_ai_analysis(prompt, context=""):
    """Generate AI analysis using Groq"""
    full_prompt = f"""
You are Arthenic, an advanced investment research assistant. 
Provide detailed, professional analysis for the following request.
Context: {context}

Request: {prompt}

Respond with:
1. Key insights (bullet points)
2. Risks to consider
3. Recommended action
4. Additional factors to monitor
"""

    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.3,
        "max_tokens": 1500
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                 headers=GROQ_HEADERS,
                                 json=payload)
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"


def guess_sector_with_ai(ticker):
    """Guess sector or label for a given stock"""
    prompt = f"What is the best short descriptive label or sector category for the ticker {ticker}? Reply with 2-4 words only."

    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 50
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                 headers=GROQ_HEADERS,
                                 json=payload)
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[AI Sector Guess Fail] {ticker}: {e}")
        return "Other"


def explain_stock_with_ai(ticker):
    """Explain what the stock represents"""
    context = f"This is for a user tracking {ticker} in their portfolio."
    prompt = f"Explain what the stock {ticker} represents and its investment outlook. Include valuation, market position, and macro factors if relevant."
    return generate_ai_analysis(prompt, context)


def generate_trade_ai_commentary(company, ticker, user_choice, correct_move, amount, result_profit):
    """Fun trading commentary for game-like responses"""
    full_prompt = f"""
You are Arthenic, a ruthless and witty AI trading mentor in a high-stakes Wall Street RPG.

Your client just made a trade:
- Company: {company} ({ticker})
- Insider Tip Said: {correct_move}
- Player Chose: {user_choice}
- Amount: ${amount:,}
- Profit/Loss: {'+$' if result_profit > 0 else '-$'}{abs(result_profit):,}

Respond with:
🎯 **Trade Summary**: Give a one-line zinger (funny, brutal, or sarcastic)
🧠 **Verdict**: Smart or Dumb Play (keep it short with 1–2 brutal reasons)
🚨 **SEC Radar**: One-liner reaction from the regulators or the Street
🔮 **Next Move Tease**: End with a cryptic/motivational line teasing their next move

Be ruthless, fast, and fun. Game-style commentary. No essays or boring explanations.
"""

    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.5,
        "max_tokens": 600
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                 headers=GROQ_HEADERS,
                                 json=payload)
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"🧠 AI jammed by SEC: {str(e)}"


def generate_strategic_stock_guidance(ticker, shares, cost_basis, purchase_date):
    """Generate personalized investment strategy for a stock."""
    headers = {
        "Authorization": f"Bearer {API_KEYS['GROQ_API_KEY']}",
        "Content-Type": "application/json"
    }

    # Calculate holding period
    try:
        days_held = (datetime.today() - purchase_date).days
    except Exception as e:
        days_held = 180  # fallback to 6 months

    context = f"""
    - Ticker: {ticker}
    - Shares Held: {shares}
    - Average Cost Basis: ${cost_basis:.2f}
    - Days Held: {days_held}

    Analyze this holding. Assume it's part of a retail investor's real-money portfolio. Determine:
    1. Should this be a long-term core position or a short-term swing trade based on past trends of stock and future news?
    2. Estimate a realistic target price to sell if the thesis plays out
    3. List factors that would justify an early exit
    4. Provide a forward-looking strategy
    """

    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": context}],
        "temperature": 0.3,
        "max_tokens": 1200
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        data = response.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        else:
            return f"⚠️ Strategic guidance unavailable: No valid AI response. Response: {data}"
    except Exception as e:
        return f"⚠️ Strategic guidance error: {str(e)}"

        response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                 headers=GROQ_HEADERS,
                                 json=payload)
        return response.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return f"⚠️ Strategic guidance unavailable: {str(e)}"