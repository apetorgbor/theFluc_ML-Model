from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
import uvicorn
import os

app = FastAPI(title="Hybrid Recommendation API")

# Request / Response Models

class PredictRequest(BaseModel):
    user_id: str
    news_id: str
    
    
class RecommendRequest(BaseModel):
    user_id: str
    top_k: Optional[int] = 10
    # optional history in case user is new or we want to override
    history: Optional[List[str]] = None
    
        
class RecommendResponseItem(BaseModel):
    news_id: str
    score: float
    
    
class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendResponseItem]
    
    
# Startup: load artifacts

MODEL_PATH = os.getenv("MODEL_PATH", "hybrid_recommendation_model.keras")
USER_EMB_PATH = os.getenv("USER_EMB_PATH", "user_embeddings.npy")
ITEM_EMB_PATH = os.getenv("ITEM_EMB_PATH", "item_embeddings.npy")
USER_ITEM_MATRIX_PATH = os.getenv("USER_ITEM_MATRIX_PATH", "user_item_matrix.npy")
NEWS_CSV = os.getenv("NEWS_CSV", "news.csv")
BEHAVIOR_CSV = os.getenv("BEHAVIOR_CSV", "behaviors.csv")
print("[API] Loading model + artifacts...")      

# Load Keras model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")    


# Load embeddings
user_embeddings = np.load(USER_EMB_PATH)
item_embeddings = np.load(ITEM_EMB_PATH)
user_item_matrix = np.load(USER_ITEM_MATRIX_PATH)

# Load CSVs to reconstruct mappings if needed
news_df = pd.read_csv(NEWS_CSV)
behaviors_df = pd.read_csv(BEHAVIOR_CSV)


# Try to find existing index mapping columns in CSVs
# News: expecting 'NewsID' and optionally 'news_idx'
if 'news_idx' in news_df.columns:
    news_id_to_index = dict(zip(news_df['NewsID'].astype(str), news_df['news_idx'].astype(int)))
else:
    unique_news = news_df['NewsID'].astype(str).unique().tolist()
    news_id_to_index = {nid: i for i, nid in enumerate(unique_news)}

index_to_news_id = {v: k for k, v in news_id_to_index.items()}


# Users: behaviors_df may have user_idx
if 'user_idx' in behaviors_df.columns and 'UserID' in behaviors_df.columns:
    user_id_to_index = dict(zip(behaviors_df['UserID'].astype(str), behaviors_df['user_idx'].astype(int)))
else:
    unique_users = behaviors_df['UserID'].astype(str).unique().tolist()
    user_id_to_index = {uid: i for i, uid in enumerate(unique_users)}

index_to_user_id = {v: k for k, v in user_id_to_index.items()}


# Sizes
num_users, user_emb_dim = user_embeddings.shape
num_items, item_emb_dim = item_embeddings.shape

print(f"[API] Loaded model. user_emb_dim={user_emb_dim}, item_emb_dim={item_emb_dim}")
print(f"[API] num_users={num_users}, num_items={num_items}")
print(f"[API] Known users in mapping: {len(user_id_to_index)}; Known news in mapping: {len(news_id_to_index)}")


# Utility helpers
# ---------------------------
def get_user_index(user_id: str) -> Optional[int]:
    """Return user_idx for a given external user_id or None if not found."""
    user_id = str(user_id)
    return user_id_to_index.get(user_id)


def get_news_index(news_id: str) -> Optional[int]:
    """Return news_idx for a given external news_id or None if not found."""
    news_id = str(news_id)
    return news_id_to_index.get(news_id)


def build_user_embedding_from_history(history_ids: List[str]) -> np.ndarray:
    """
    Create a user embedding by averaging item embeddings of items in history.
    If history empty or unknown ids, returns zero-vector.
    """
    if not history_ids:
        return np.zeros((user_emb_dim,), dtype=np.float32)
    valid_indices = [news_id_to_index[nid] for nid in history_ids if nid in news_id_to_index]
    if not valid_indices:
        return np.zeros((user_emb_dim,), dtype=np.float32)
    emb = np.mean(item_embeddings[valid_indices], axis=0)
    return emb


def recommend_by_dot(user_vec: np.ndarray, candidate_indices: Optional[List[int]] = None, top_k: int = 10):
    """
    Compute similarity scores (dot product) between user_vec and item_embeddings.
    Returns list of (news_idx, score) sorted descending.
    """
    if candidate_indices is None:
        candidate_embeddings = item_embeddings  # shape (num_items, dim)
        scores = candidate_embeddings.dot(user_vec)
        top_idx = np.argpartition(-scores, top_k-1)[:top_k]
        top_sorted = top_idx[np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i])) for i in top_sorted]
    else:
        cand_emb = item_embeddings[candidate_indices]
        scores = cand_emb.dot(user_vec)
        order = np.argsort(-scores)[:top_k]
        return [(int(candidate_indices[i]), float(scores[i])) for i in order]
    
    
# Endpoints
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok", "num_users": num_users, "num_items": num_items}    

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Predict click probability for a (user_id, news_id) pair.
    """
    user_idx = get_user_index(req.user_id)
    news_idx = get_news_index(req.news_id)
    
    # If user or news unknown, attempt to handle gracefully:
    if user_idx is None:
        # try to compute embedding from optional behaviors.csv history if exists
        # fallback: return 400 asking for history
        raise HTTPException(status_code=400, detail=f"Unknown user_id: {req.user_id}. Provide history or register user first.")
    if news_idx is None:
        raise HTTPException(status_code=400, detail=f"Unknown news_id: {req.news_id}.")

    # Model expects arrays of shape (batch,)
    pred = model.predict([np.array([user_idx]), np.array([news_idx])], verbose=0)
    prob = float(pred[0][0])
    return {"user_id": req.user_id, "news_id": req.news_id, "click_probability": prob}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """
    Return top-k recommendations for a user.
    Behavior:
    - If user exists in mapping, we use precomputed user_embeddings[user_idx].
    - If user is unknown but 'history' provided, average item embeddings of history.
    - Otherwise error.
    """
    top_k = int(req.top_k or 10)
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be > 0")

    user_idx = get_user_index(req.user_id)
    if user_idx is not None and 0 <= user_idx < user_embeddings.shape[0]:
        user_vec = user_embeddings[user_idx]
    else:
        # try to build from provided history if available
        if req.history:
            user_vec = build_user_embedding_from_history(req.history)
        else:
            raise HTTPException(status_code=400, detail="Unknown user and no history supplied to build embedding.")

    # Optionally you could filter candidate items (e.g., exclude items in user's history)
    # Build candidate set (all items)
    recs = recommend_by_dot(user_vec, candidate_indices=None, top_k=top_k)

    # Convert indices back to news IDs and return scores
    rec_items = []
    for news_idx, score in recs:
        news_id = index_to_news_id.get(news_idx, str(news_idx))
        rec_items.append(RecommendResponseItem(news_id=news_id, score=score))

    return RecommendResponse(user_id=req.user_id, recommendations=rec_items)

# ---------------------------
# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)









