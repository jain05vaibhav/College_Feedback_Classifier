"""
CampusLens Backend API
----------------------
Run from project root:  uvicorn backend.api:app --reload --port 8000
"""

import os, sys, json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── locate models relative to this file ───────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT_MODEL_DIR  = os.path.join(ROOT, "cat_model")
SENT_MODEL_DIR = os.path.join(ROOT, "sent_model")
REVIEWS_FILE   = os.path.join(ROOT, "reviews.json")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="CampusLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── load models once at startup ────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[CampusLens] Loading models on {device}...")

cat_tokenizer  = AutoTokenizer.from_pretrained(CAT_MODEL_DIR)
cat_model      = AutoModelForSequenceClassification.from_pretrained(
                     CAT_MODEL_DIR, torch_dtype=torch.float32).to(device)
sent_tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL_DIR)
sent_model     = AutoModelForSequenceClassification.from_pretrained(
                     SENT_MODEL_DIR, torch_dtype=torch.float32).to(device)
cat_model.eval()
sent_model.eval()

print("[CampusLens] Models ready ✅")

ID2CAT  = {0:"Academics", 1:"Administration", 2:"Facilities", 3:"Faculty", 4:"Hostel", 5:"Mess", 6:"Others"}
ID2SENT = {0:"Negative", 1:"Neutral", 2:"Positive"}

# ── helpers ───────────────────────────────────────────────────
import math

def safe_float(val: float) -> float:
    """Return 0.0 for NaN/Inf so JSON serialisation never blows up."""
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return val

def sanitize_obj(obj):
    """Recursively replace NaN/Inf floats in any dict/list structure."""
    if isinstance(obj, float):
        return safe_float(obj)
    if isinstance(obj, dict):
        return {k: sanitize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_obj(v) for v in obj]
    return obj

def load_reviews():
    if os.path.exists(REVIEWS_FILE):
        try:
            with open(REVIEWS_FILE, "r", encoding="utf-8") as f:
                raw = f.read()
            # Python's json.loads accepts 'NaN' — sanitize after loading
            data = json.loads(raw)
            return sanitize_obj(data)
        except Exception:
            return []
    return []

def save_reviews(reviews):
    with open(REVIEWS_FILE, "w", encoding="utf-8") as f:
        json.dump(reviews, f, indent=2, ensure_ascii=False)

# ── schemas ───────────────────────────────────────────────────
class ReviewRequest(BaseModel):
    text: str
    author: str = "Anonymous"

# ── routes ────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "device": device}


@app.post("/api/reviews")
def submit_review(req: ReviewRequest):
    if not req.text.strip():
        raise HTTPException(400, "Review text cannot be empty.")

    # category prediction
    inp_cat = cat_tokenizer(req.text, return_tensors="pt", truncation=True, max_length=512).to(device)
    # sentiment prediction
    inp_sent = sent_tokenizer(req.text, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        logits_cat  = cat_model(**inp_cat).logits
        logits_sent = sent_model(**inp_sent).logits

    # Cast to float32 — DeBERTa saves in fp16 which causes NaN in softmax on CPU
    logits_cat  = logits_cat.float()
    logits_sent = logits_sent.float()

    cat_idx  = logits_cat.argmax().item()
    sent_idx = logits_sent.argmax().item()

    cat_probs  = torch.nn.functional.softmax(logits_cat,  dim=-1)[0]
    sent_probs = torch.nn.functional.softmax(logits_sent, dim=-1)[0]

    # all category probs for a mini-breakdown (guarded against NaN)
    all_cats = {
        ID2CAT[i]: round(safe_float(cat_probs[i].item()) * 100, 1)
        for i in range(len(ID2CAT))
    }

    review = {
        "id":             datetime.now().isoformat(),
        "author":         req.author.strip() or "Anonymous",
        "text":           req.text.strip(),
        "category":       ID2CAT[cat_idx],
        "sentiment":      ID2SENT[sent_idx],
        "cat_confidence": round(safe_float(cat_probs[cat_idx].item())  * 100, 1),
        "sent_confidence":round(safe_float(sent_probs[sent_idx].item())* 100, 1),
        "all_cats":       all_cats,
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    reviews = load_reviews()
    reviews.append(review)
    save_reviews(reviews)

    return review


@app.get("/api/reviews")
def get_reviews():
    return list(reversed(load_reviews()))   # newest first


@app.get("/api/leaderboard")
def get_leaderboard():
    reviews = load_reviews()
    stats: dict[str, dict] = {}
    for r in reviews:
        cat = r["category"]
        if cat not in stats:
            stats[cat] = {
                "category": cat,
                "total":    0,
                "positive": 0,
                "negative": 0,
                "neutral":  0,
                "score":    0,   # positive - negative
            }
        stats[cat]["total"]   += 1
        key = r["sentiment"].lower()
        stats[cat][key]       += 1
        stats[cat]["score"]   += (1 if key == "positive" else -1 if key == "negative" else 0)

    return list(stats.values())


@app.delete("/api/reviews/{review_id}")
def delete_review(review_id: str):
    reviews = load_reviews()
    filtered = [r for r in reviews if r["id"] != review_id]
    if len(filtered) == len(reviews):
        raise HTTPException(404, "Review not found.")
    save_reviews(filtered)
    return {"deleted": True}
