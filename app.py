from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sqlite3
import pandas as pd
import numpy as np
import re
import os
import pickle
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import faiss

app = FastAPI()

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
base_dir         = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "models")
data_dir = os.path.join(base_dir, "data")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

db_path = os.path.join(data_dir, "papers_warehouse.db")
faiss_path = os.path.join(models_dir, "paper_index.faiss")
df_path = os.path.join(models_dir, "papers_df.pkl")


REPO_ID = "v-sriram/paperlens-model-files"

def download_if_missing(local_path, filename):
    if not os.path.exists(local_path):
        print(f"Downloading {filename}...")

        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset"
        )

        import shutil
        shutil.copy(downloaded, local_path)

        print(f"{filename} downloaded successfully!")



download_if_missing(df_path, "papers_df.pkl")
download_if_missing(faiss_path, "paper_index.faiss")
# ─────────────────────────────────────────
# STEP 1 — Load DataFrame
# First time → loads from DB → saves pkl
# Next time  → loads from pkl (5 secs!)
# ────────────────────────────────────────
print("Loading DataFrame...")
with open(df_path, "rb") as f:
    df = pickle.load(f)
print(f"Loaded {len(df)} papers")
print("Loading FAISS index...")
index = faiss.read_index(faiss_path)
print("FAISS index loaded!")
print(f"Total vectors: {index.ntotal}")




print("Loading BERT model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("BERT model loaded!")
# ─────────────────────────────────────────
# QUERY CLEANING
# Same cleaning as preprocessing.py!
# ─────────────────────────────────────────
def clean_query(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if len(w) > 2]
    return " ".join(words)
def clean_link(val):
    v = str(val).strip()
    if v in ["nan", "none", "None", "", "NaN"]:
        return ""
    return v
def clean_subjects(val):
    if not val:
        return ""

    s = str(val)
    # remove brackets and quotes
    s = re.sub(r"[\[\]']", "", s)
    # split + clean + join
    parts = [x.strip() for x in s.split(",") if x.strip()]
    
    return ", ".join(parts)
# ─────────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str

# ─────────────────────────────────────────
# ROUTE 1 — Home Page
# ─────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home():
    html_path = os.path.join(base_dir, "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# ─────────────────────────────────────────
# ROUTE 2 — Recommend API
# ─────────────────────────────────────────
@app.post("/recommend")
def recommend(req: QueryRequest):
    query = req.query.strip()
    if not query:
        return []
    # Clean query same way as preprocessing!
    cleaned_query = clean_query(query)
    print(f"Original : {query}")
    print(f"Cleaned  : {cleaned_query}")
    if not cleaned_query:
        return []
    # Encode with BERT
    query_embedding = model.encode(
        [cleaned_query],
        convert_to_numpy=True
    )
    # Cosine similarity against all embeddings
    query_embedding = query_embedding.astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, 10)

    top_indices = indices[0]
    top_scores = scores[0]

    
    results = []

    for idx, score in zip(top_indices, top_scores):

        if idx == -1:
            continue
            
        if score<0.2:
            continue
        row = df.iloc[idx]

        results.append({
            "title": str(row.get("title","N/A")),
            "authors": str(row.get("authors","N/A")),
            "category": str(row.get("category","N/A")),
            "primary_subject": str(row.get("primary_subject","N/A")),
            "subjects": str(row.get("subjects","N/A")),
            "date": str(row.get("date","N/A")),
            "description": str(row.get("description","")),
            "link": clean_link(row.get("link","")),
            "link_of_paper": clean_link(row.get("link_of_paper","")),
            "link_of_pdf": clean_link(row.get("link_of_pdf","")),
            "score": round(float(score),4)
        })
    return results

# ─────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8001,
        reload=False
    )