import json
import firebase_admin
from firebase_admin import firestore, credentials

if not firebase_admin._apps:
    cred = credentials.Certificate("personal-project-firebasekey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

def load_cached_fundamentals_local(filepath="data/cached_fundamentals.json"):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading local fundamentals cache: {e}")
        return []

def load_cached_fundamentals_firebase():
    try:
        fundamentals = []
        docs = db.collection("stock_fundamentals").stream()
        for doc in docs:
            fundamentals.append(doc.to_dict())
        return fundamentals
    except Exception as e:
        print(f"Error loading fundamentals from Firebase: {e}")
        return []

def get_fundamental_by_ticker(ticker, fundamentals):
    return next((item for item in fundamentals if item["ticker"] == ticker), None)
