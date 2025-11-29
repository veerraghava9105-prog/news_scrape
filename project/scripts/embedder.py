# project/scripts/embedder.py
# Docker: run Qdrant before running this file
# docker run -d -p 6333:6333 -v qdrant-storage:/qdrant/storage qdrant/qdrant
# Open dashboard: localhost:6333/dashboard

import json
import time
import uuid
import numpy as np
from pathlib import Path
from urllib.parse import urlparse
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# ---- Universal Config ----
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "news_articles"
VECTOR_SIZE = 768  # BGE model dim
MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBED_BATCH = 64   # docs per batch

# ---- Paths ----
BASE = Path(__file__).resolve().parents[1]  # project/
DATA = BASE / "data"
ARTICLES_JSONL = DATA / "articles.jsonl"

client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

# ---- Utils ----
def read_jsonl(path: Path):
    docs = []
    if not path.exists():
        print("⚠️ JSONL file not found:", path)
        return docs
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            if line.strip():
                try:
                    docs.append(json.loads(line))
                except Exception as e:
                    print("❗ Bad JSONL line skipped:", e)
    return docs

def fetch_existing_ids():
    ids = set()
    offset = None
    while True:
        page, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10_000,
            offset=offset,
            with_payload=False,
            with_vectors=False
        )
        for p in page:
            ids.add(str(p.id))
        if offset is None:
            break
    return ids

def embed_texts(texts: List[str], model: SentenceTransformer):
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype(np.float32)

# ---- Main ----
def main():
    print("\n🚀 Booting pipeline universally for the people and the gods...")

    # 1. Make sure collection exists
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"✅ Collection '{COLLECTION_NAME}' exists. Dim={collection_info.config.params.vectors.size}")
    except Exception:
        print("🔥 No collection found. Creating one...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print(f"🎯 Created collection '{COLLECTION_NAME}' fresh ✅")

    # 2. Load docs
    rows = read_jsonl(ARTICLES_JSONL)
    print(f"📚 Docs in JSONL: {len(rows)}")
    if not rows:
        print("🛑 JSONL empty, fix ingestion first.")
        return

    # 3. Filter new docs
    print("🔎 Scrolling DB to collect existing vector IDs...")
    existing_ids = fetch_existing_ids()
    print(f"🧠 Existing IDs in DB: {len(existing_ids)}")

    new_docs = []
    new_ids = []
    payloads = []

    print("🧩 Generating UUID5 IDs and mapping metadata...")

    for doc in rows:
        url = doc.get("url") or doc.get("id")
        if not url:
            continue
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, url))

        if pid not in existing_ids:
            new_docs.append(doc)
            new_ids.append(pid)

            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            source = doc.get("feed") or doc.get("source") or domain
            published = doc.get("published", "")
            scraped_at = doc.get("scraped_at") or doc.get("time") or ""

            payloads.append({
                "title": doc.get("title", ""),
                "url": url,
                "domain": domain,
                "source": source,
                "published": published,
                "scraped_at": scraped_at,
                "time_scraped": scraped_at
            })

    print(f"🚀 New docs to embed this run: {len(new_ids)}")
    if not new_ids:
        print("✅ Nothing new. DB already packed with stars.")
        return

    # 4. Load model on GPU if possible
    print("⚡ Loading model...")
    try:
        model = SentenceTransformer(MODEL_NAME, device="cuda")
        print("🔥 GPU mode ON ✅")
    except Exception:
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        print("⚡ GPU not found → CPU fallback ✅")

    # 5. Embed batches
    print(f"📚 Embedding {len(new_ids)} documents in batches of {EMBED_BATCH}...")
    t0 = time.time()

    for i in range(0, len(new_ids), EMBED_BATCH):
        batch_docs = new_docs[i:i+EMBED_BATCH]
        batch_ids = new_ids[i:i+EMBED_BATCH]
        batch_payloads = payloads[i:i+EMBED_BATCH]
        texts = [d.get("text") or d.get("content") or d.get("title", "") for d in batch_docs]
        vectors = embed_texts(texts, model)

        points = []
        for j, pid in enumerate(batch_ids):
            points.append(PointStruct(id=pid, vector=vectors[j].tolist(), payload=batch_payloads[j]))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  ✅ upserted batch {i//EMBED_BATCH+1} ({len(batch_ids)} vectors)")

    dur_sec = time.time() - t0
    dur_min = dur_sec / 60

    print(f"\n🎉 Embedding complete → {dur_sec:.1f} seconds ({dur_min:.2f} minutes)")
    print("📊 Collection status:")
    print(client.get_collection(COLLECTION_NAME))
    print("🌍 Pipeline universal and stable ✅")

if __name__ == "__main__":
    main()
