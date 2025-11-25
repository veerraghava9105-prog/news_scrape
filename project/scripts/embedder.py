# embedder.py
import os
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List
from .utils_io import read_jsonl, write_jsonl
import time

BASE = Path(__file__).resolve().parents[1]  # project/
DATA = BASE / "data"
ARTICLES = DATA / "articles.jsonl"
EMB_NPZ = DATA / "embeddings.npz"
INDEX_JSON = DATA / "embeddings_index.json"

# CONFIG
MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 64
NORM = True  # L2 normalize embeddings for cosine via dot

def load_articles(limit: int = None):
    rows = []
    for i, doc in enumerate(read_jsonl(str(ARTICLES))):
        rows.append(doc)
        if limit and len(rows) >= limit:
            break
    return rows

def embed_texts(texts: List[str], model: SentenceTransformer, batch_size=BATCH_SIZE):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        arr = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
        embs.append(arr)
    embs = np.vstack(embs).astype(np.float32)
    if NORM:
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        embs = embs / norms
    return embs

def main(limit=None, model_name=MODEL_NAME):
    print("🔁 Loading model one-time only...")
    model = SentenceTransformer(model_name)
    print("⚡ Model loaded:", model_name)

    rows = load_articles(limit)
    texts = []
    metadata = []
    for doc in rows:
        text = doc.get("text") or doc.get("content") or doc.get("title", "")
        # fallback to title if no text
        texts.append(text)
        metadata.append({"id": doc.get("id") or doc.get("url"), "title": doc.get("title"), "url": doc.get("url")})

    print(f"📚 Embedding {len(texts)} documents in batches of {BATCH_SIZE} ...")
    start = time.time()
    embs = embed_texts(texts, model)
    dur = time.time() - start
    print(f"✅ Done embeddings in {dur:.1f}s, shape {embs.shape}")

    # Save compressed
    np.savez_compressed(str(EMB_NPZ), embeddings=embs)
    with open(INDEX_JSON, "w", encoding="utf8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("💾 Saved:", EMB_NPZ, INDEX_JSON)

if __name__ == "__main__":
    # quick CLI: python embedder.py 100
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=limit)
