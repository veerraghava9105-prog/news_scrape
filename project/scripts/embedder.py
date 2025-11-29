# project/scripts/embedder.py
#docker run -d -p 6333:6333 -v qdrant-storage:/qdrant/storage qdrant/qdrant
#localhost:6333/dashboard


import json
import time
import uuid
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from typing import List

# Paths
BASE = Path(__file__).resolve().parents[1]  # project/
DATA = BASE / "data"
ARTICLES = DATA / "articles.jsonl"

# Config
COLLECTION = "news_vectors"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH = 64

# --- Helpers for JSONL ---
def read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def fetch_qdrant_ids(client: QdrantClient):
    """Scroll DB once to collect all existing IDs."""
    ids = set()
    offset = None
    while True:
        page, offset = client.scroll(
            collection_name=COLLECTION, limit=10_000, offset=offset, with_payload=False, with_vectors=False
        )
        for p in page:
            ids.add(p.id)
        if offset is None:
            break
    return ids

def embed_batch(texts: List[str], model: SentenceTransformer):
    arr = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    # normalize_embeddings=True gives L2 norm for cosine via dot at query time
    return arr.astype(np.float32)

def upsert_points(client: QdrantClient, ids: List[str], embs: np.ndarray, payloads: List[dict]):
    """Upsert a set of points to Qdrant."""
    pts = []
    for i, pid in enumerate(ids):
        pts.append(PointStruct(id=pid, vector=embs[i].tolist(), payload=payloads[i]))
    client.upsert(collection_name=COLLECTION, points=pts)

# --- Main ---
def main():
    rows = read_jsonl(ARTICLES)
    print(f"📁 total docs in jsonl: {len(rows)}, new to embed: {len(new_docs)}")
    client = QdrantClient("localhost", port=6333)
    print("🔎 Fetching existing IDs from Qdrant…")
    existing_ids = fetch_qdrant_ids(client)
    print(f"✅ Qdrant already has {len(existing_ids)} vectors")

    # Determine actually new docs
    new_docs = []
    new_ids = []
    payloads = []

    for doc in rows:
        url = doc.get("url") or doc.get("id")
        if not url:
            continue
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, url))
        if pid not in existing_ids:
            new_docs.append(doc)
            new_ids.append(pid)
            payloads.append({
                "title": doc.get("title", ""),
                "url": url,
                "source": doc.get("source", ""),
                "time_scraped": doc.get("time", "")
            })

    print(f"🆕 Found {len(new_ids)} new docs that need embeddings.")

    if not new_ids:
        print("🛑 Nothing new to embed. Go sleep.")
        return

    print("🔁 Loading model (one-time SO keep this process alive!)…")
    model = SentenceTransformer(MODEL_NAME, device="cpu")  # CPU default
    # if GPU available we’ll auto-switch using device="cuda"
    print(f"⚡ Model ready: {MODEL_NAME}")

    print(f"📚 Embedding {len(new_ids)} new documents in batches of {BATCH} …")
    t0 = time.time()

    for i in range(0, len(new_ids), BATCH):
        batch_docs = new_docs[i:i+BATCH]
        batch_texts = [d.get("text") or d.get("content") or d.get("title","") for d in batch_docs]
        batch_ids = new_ids[i:i+BATCH]
        batch_payloads = payloads[i:i+BATCH]

        arr = embed_batch(batch_texts, model)
        upsert_points(client, batch_ids, arr, batch_payloads)

        print(f"  ⚡ upserted batch {i//BATCH+1} ({len(batch_ids)} vectors)")

    dur = time.time() - t0
    print(f"✅ Done in {dur:.1f}s total!")

if __name__ == "__main__":
    main()
