from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import sqlite3
import requests
import os

# -------------------------
# 1. CONFIGURATION
# -------------------------
CONFIDENCE_THRESHOLD = 0.35  # Hardcoded

# Replace with your frontend URL, or use "*" to allow all origins (for testing)
FRONTEND_URL = "https://chatbot-customer-success.netlify.app"

# -------------------------
# 2. INIT APP
# -------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],  # allows frontend to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# 3. LOAD ML MODELS
# -------------------------
with open("ml/intent_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("ml/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

RESPONSES = {
    "pricing": "Our pricing starts at $10/month.",
    "account_help": "You can reset your password or contact support for account issues.",
    "refund_policy": "Refunds take up to 5 business days.",
    "contact_support": "You can reach support at support@example.com."
}

# -------------------------
# 4. DATABASE
# -------------------------
DB_PATH = "chat_logs.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

conn = get_connection()
conn.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_message TEXT,
    predicted_intent TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()
conn.close()

# -------------------------
# 5. Pydantic model
# -------------------------
class Message(BaseModel):
    message: str

# -------------------------
# 6. Optional Zendesk fallback
# -------------------------
ZENDESK_SUBDOMAIN = ""  # Leave empty if not using
ZENDESK_EMAIL = ""
ZENDESK_API_TOKEN = ""

def create_zendesk_ticket(message):
    if not all([ZENDESK_SUBDOMAIN, ZENDESK_EMAIL, ZENDESK_API_TOKEN]):
        return
    url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/tickets.json"
    data = {"ticket": {"subject": "Chatbot fallback", "comment": {"body": message}, "priority": "normal"}}
    requests.post(url, json=data, auth=(f"{ZENDESK_EMAIL}/token", ZENDESK_API_TOKEN))

# -------------------------
# 7. ROUTES
# -------------------------
@app.get("/")
def read_root():
    return {"message": "Backend is running 🚀"}

@app.post("/chat")
def chat(msg: Message):
    X = vectorizer.transform([msg.message])
    probs = model.predict_proba(X)[0]
    intent = model.classes_[probs.argmax()]
    confidence = probs.max()

    # Log conversation
    conn = get_connection()
    conn.execute(
        "INSERT INTO conversations (user_message, predicted_intent, confidence) VALUES (?, ?, ?)",
        (msg.message, intent, confidence)
    )
    conn.commit()
    conn.close()

    # Low-confidence fallback
    if confidence < CONFIDENCE_THRESHOLD:
        create_zendesk_ticket(msg.message)
        return {"reply": "Connecting to support...", "handoff": True}

    return {"reply": RESPONSES.get(intent, "I’m not sure I understand."), "intent": intent, "confidence": confidence}

