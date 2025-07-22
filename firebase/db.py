import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase app if not already done
if not firebase_admin._apps:
    cred = credentials.Certificate("personal-project-firebasekey.json")  # <-- local JSON file
    firebase_admin.initialize_app(cred)

db = firestore.client()


# ---------------------------
# 🔹 Portfolio CRUD
# ---------------------------
def get_user_portfolio(user_id):
    try:
        doc = db.collection("users").document(user_id).get()
        return doc.to_dict().get("portfolio", {}) if doc.exists else {}
    except Exception as e:
        print(f"Error fetching portfolio: {e}")
        return {}

def save_user_portfolio(user_id, portfolio):
    try:
        db.collection("users").document(user_id).set({"portfolio": portfolio}, merge=True)
    except Exception as e:
        print(f"Error saving portfolio: {e}")


# ---------------------------
# 🔹 Watchlist CRUD
# ---------------------------
def get_user_watchlist(user_id):
    try:
        doc = db.collection("users").document(user_id).get()
        return doc.to_dict().get("watchlist", []) if doc.exists else []
    except Exception as e:
        print(f"Error fetching watchlist: {e}")
        return []

def save_user_watchlist(user_id, watchlist):
    try:
        db.collection("users").document(user_id).set({"watchlist": watchlist}, merge=True)
    except Exception as e:
        print(f"Error saving watchlist: {e}")


# ---------------------------
# 🔹 Preferences CRUD
# ---------------------------
def get_user_preferences(user_id):
    try:
        doc = db.collection("users").document(user_id).get()
        return doc.to_dict().get("preferences", {}) if doc.exists else {}
    except Exception as e:
        print(f"Error fetching preferences: {e}")
        return {}

def save_user_preferences(user_id, preferences):
    try:
        db.collection("users").document(user_id).set({"preferences": preferences}, merge=True)
    except Exception as e:
        print(f"Error saving preferences: {e}")


# ---------------------------
# 🔹 Recommendations CRUD
# ---------------------------
def get_user_recommendations(user_id):
    try:
        doc = db.collection("users").document(user_id).get()
        return doc.to_dict().get("recommendations", []) if doc.exists else []
    except Exception as e:
        print(f"Error fetching recommendations: {e}")
        return []

def save_user_recommendations(user_id, recommendations):
    try:
        db.collection("users").document(user_id).set({"recommendations": recommendations}, merge=True)
    except Exception as e:
        print(f"Error saving recommendations: {e}")


# ---------------------------
# 🔹 Fundamentals (Batch Write)
# ---------------------------
def save_fundamentals_batch(fundamentals):
    try:
        batch = db.batch()
        for stock in fundamentals:
            doc_ref = db.collection("stock_fundamentals").document(stock['ticker'])
            batch.set(doc_ref, stock)
        batch.commit()
        print("Successfully saved fundamentals batch.")
    except Exception as e:
        print(f"Error saving fundamentals batch: {e}")
