# qdrant_backfill.py
import os
import json
import time
import hashlib
import numpy as np
from tqdm import tqdm
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

BASE = Path(__file__).resolve().parents[1]  # project/
DATA_DIR = BASE / "data"
EMB_NPZ = DATA_DIR / "embeddings.npz"
INDEX_JSON = DATA_DIR / "embeddings_index.json"

COLLECTION_NAME = "news_vectors"
COLLECTION_SIZE = 768
UPSERT_BATCH = 512

client = QdrantClient("localhost", port=6333)

def stable_id(text_id: str) -> int:
    """ Generate consistent 64-bit hash for each article URL/id """
    return int(hashlib.sha256(text_id.encode("utf-8")).hexdigest()[:16], 16)

def main():
    print("📦 Loading stored embeddings...")
    data = np.load(str(EMB_NPZ), allow_pickle=True)

    embeddings = data["embeddings"]
    print(f"✅ Loaded vectors: {embeddings.shape}")

    with open(INDEX_JSON, "r", encoding="utf8") as f:
        metadata = json.load(f)
    print(f"🧾 Loaded metadata records: {len(metadata)}")

    print("🧠 Rebuilding Qdrant collection...")
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=COLLECTION_SIZE,
            distance=Distance.COSINE
        )
    )

    print("🚚 Upserting vectors into Qdrant in batches...")
    points = []
    t0 = time.time()

    for i, doc in enumerate(tqdm(metadata)):
        tid = doc["id"]
        pid = stable_id(tid)

        payload = {"title": doc.get("title"), "url": doc.get("url")}

        points.append(PointStruct(id=pid, vector=embeddings[i].tolist(), payload=payload))

        if len(points) >= UPSERT_BATCH:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"✅ Insert complete in {time.time()-t0:.1f}s")
    print(client.get_collection(COLLECTION_NAME))

if __name__ == "__main__":
    main()
