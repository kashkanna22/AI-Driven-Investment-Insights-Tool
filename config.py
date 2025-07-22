RISK_PROFILES = {
    "Conservative": {
        "description": "Preserve capital with minimal risk",
        "preferences": {
            "max_volatility": 0.2,
            "min_dividend": 0.03,
            "sectors": ["Utilities", "Consumer Defensive", "Healthcare"],
            "max_pe": 20,
            "min_market_cap": 10e9  # $10B+
        }
    },
    "Moderate": {
        "description": "Balance between growth and income",
        "preferences": {
            "max_volatility": 0.3,
            "min_dividend": 0.015,
            "sectors": ["Technology", "Financial Services", "Healthcare"],
            "max_pe": 25,
            "min_market_cap": 2e9  # $2B+
        }
    },
    "Aggressive": {
        "description": "Maximize growth potential",
        "preferences": {
            "max_volatility": 0.5,
            "min_dividend": 0.0,
            "sectors": ["Technology", "Communication Services", "Consumer Cyclical"],
            "max_pe": None,
            "min_market_cap": 500e6  # $500M+
        }
    },
    "Income": {
        "description": "Focus on dividend income",
        "preferences": {
            "max_volatility": 0.25,
            "min_dividend": 0.04,
            "sectors": ["Utilities", "Real Estate", "Energy"],
            "max_pe": 18,
            "min_market_cap": 5e9  # $5B+
        }
    },
    "ESG": {
        "description": "Socially responsible investing",
        "preferences": {
            "max_volatility": 0.35,
            "min_dividend": 0.01,
            "sectors": ["Healthcare", "Technology", "Consumer Defensive"],
            "excluded_sectors": ["Energy", "Tobacco", "Weapons"],
            "max_pe": 30,
            "min_market_cap": 1e9  # $1B+
        }
    }
}

FIREBASE_CONFIG = {
    "apiKey": "AIzaSyBhLUNQ9nzm0nadAv-dQ9nEtgJs1Kb8bDY",
    "authDomain": "personal-project-80f32.firebaseapp.com",
    "databaseURL": "https://personal-project-80f32-default-rtdb.firebaseio.com/",
    "projectId": "personal-project-80f32",
    "storageBucket": "personal-project-80f32.appspot.com",
    "messagingSenderId": "969572991303",
    "appId": "1:969572991303:web:2a0c1cab8b06a6645b4662",
    "measurementId": "G-XXXXXXXXXX"
}

API_KEYS = {
    "GROQ_API_KEY": "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",
    "FINNHUB_API_KEY": "d0fte1pr01qr6dbu77mgd0fte1pr01qr6dbu77n0",
    "ALPHA_VANTAGE_KEY": "EYK7GNAZP045LRQT"
}

SECTOR_ETFS = {
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLK": "Technology",
    "XLU": "Utilities"
}