import os
import pickle
import numpy as np
import faiss

# ------------------------------------
# Paths
# ------------------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(base_dir, "models")
embeddings_path = os.path.join(models_dir, "bert_embeddings.pkl")
faiss_path = os.path.join(models_dir, "paper_index.faiss")

# ------------------------------------
# Load embeddings
# ------------------------------------
print("Loading embeddings...")
with open(embeddings_path, "rb") as f:
    embeddings = pickle.load(f)

embeddings = np.asarray(embeddings).astype("float32")

print("Shape:", embeddings.shape)

# ------------------------------------
# Normalize vectors
# ------------------------------------
print("Normalizing vectors...")

faiss.normalize_L2(embeddings)

# ------------------------------------
# Build index
# ------------------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("Total vectors:", index.ntotal)

# ------------------------------------
# Save index
# ------------------------------------
faiss.write_index(index, faiss_path)

print("FAISS index saved!")
print(faiss_path)