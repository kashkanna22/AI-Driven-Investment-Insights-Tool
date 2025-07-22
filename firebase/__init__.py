import pyrebase
from config import FIREBASE_CONFIG

firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
auth = firebase.auth()
db = firebase.database()