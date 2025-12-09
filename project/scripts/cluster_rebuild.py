# project/scripts/cluster_rebuild.py
"""
Mutual-kNN clustering for news deduplication.
- No giant clusters ever again
- Same-story = grouped
- Unrelated stories = separate
- MAX_CLUSTER_SIZE = 7 (trim by similarity)
- canonical chosen by earliest timestamp
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient

# ---------------- CONFIG ----------------
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "news_articles")

K_NEIGHBORS = 10
MIN_SIM = 0.70                      # GOOD DEFAULT (adjustable)
MAX_CLUSTER_SIZE = 7                # HARD LIMIT — keeps feed clean

BATCH_SIZE_SCROLL = 1000

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CLUSTERS_JSON = DATA_DIR / "clusters.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)
# ----------------------------------------


# ---------- HELPERS ----------
def normalize_rows(mat: np.ndarray):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def fetch_all_points(client: QdrantClient):
    pts = []
    offset = None

    print("📡 Scrolling points from Qdrant (with vectors)...")
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=BATCH_SIZE_SCROLL,
            with_payload=True,
            with_vectors=True,
            offset=offset,
        )
        if not batch:
            break
        pts.extend(batch)
        if offset is None:
            break

    print(f"➡️  Fetched {len(pts)} points")
    return pts


def parse_time(payload: dict) -> Optional[float]:
    if not payload:
        return None
    if "scraped_at" in payload and isinstance(payload["scraped_at"], (int, float)):
        return float(payload["scraped_at"])
    pub = payload.get("published")
    if isinstance(pub, (int, float)):
        return float(pub)
    return None


def choose_canonical(payloads, members):
    best = members[0]
    best_t = float("inf")

    for idx in members:
        t = parse_time(payloads[idx]) or float("inf")
        if t < best_t:
            best_t = t
            best = idx

    return best


# ---------- MUTUAL-kNN CLUSTERING ----------
class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int):
        a = self.find(a)
        b = self.find(b)
        if a == b:
            return
        if self.r[a] < self.r[b]:
            self.p[a] = b
        else:
            self.p[b] = a
            if self.r[a] == self.r[b]:
                self.r[a] += 1


def mutual_knn_clustering(vectors: np.ndarray, k: int, min_sim: float):
    n = vectors.shape[0]
    print(f"🔢 Building mutual-kNN (n={n}, k={k}, min_sim={min_sim}) ...")

    sims = vectors @ vectors.T  # FULL SIMILARITY MATRIX (n x n)
    np.fill_diagonal(sims, -1.0)

    topk_idx = np.argpartition(-sims, k, axis=1)[:, :k]

    edges = []
    for i in range(n):
        for j in topk_idx[i]:
            if sims[i, j] >= min_sim and i in topk_idx[j]:
                edges.append((i, j))

    print(f"→ mutual edges (undirected): {len(edges)}")

    dsu = DSU(n)
    for i, j in edges:
        dsu.union(i, j)

    comps = {}
    for i in range(n):
        r = dsu.find(i)
        comps.setdefault(r, []).append(i)

    print(f"🧩 Discovered {len(comps)} clusters (mutual-kNN).")
    return list(comps.values())


# ---------- MAIN ----------
def run():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    pts = fetch_all_points(client)
    if not pts:
        print("⚠️ No points found.")
        return

    q_ids = [p.id for p in pts]
    payloads = [p.payload or {} for p in pts]

    vecs = normalize_rows(
        np.array([np.array(p.vector, dtype=np.float32) for p in pts])
    )

    raw_clusters = mutual_knn_clustering(vecs, K_NEIGHBORS, MIN_SIM)

    clusters_out = {}
    label_for_idx = {}

    print("✂️ Applying MAX_CLUSTER_SIZE trimming (=", MAX_CLUSTER_SIZE, ")")

    for label, members in enumerate(raw_clusters):

        # ----- HARD TRIM (TOP-7 BY SIMILARITY TO CENTROID) -----
        if len(members) > MAX_CLUSTER_SIZE:
            mvecs = vecs[members]
            centroid = mvecs.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-8

            sims = mvecs @ centroid
            top = np.argsort(-sims)[:MAX_CLUSTER_SIZE]
            members = [members[i] for i in top]

        # centroid (for saving)
        mvecs = vecs[members]
        centroid = mvecs.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8

        canon_local = choose_canonical(payloads, members)
        canon_qid = q_ids[canon_local]
        canon_url = payloads[canon_local].get("url", str(canon_qid))

        clusters_out[str(label)] = {
            "label": label,
            "size": len(members),
            "canonical_point_id": str(canon_qid),
            "canonical_url": canon_url,
            "members": [str(q_ids[m]) for m in members],
        }

        for m in members:
            label_for_idx[m] = label

    # ---- SAVE CLUSTERS.JSON ----
    save_obj = {
        "k_neighbors": K_NEIGHBORS,
        "threshold": MIN_SIM,
        "max_cluster_size": MAX_CLUSTER_SIZE,
        "num_points": len(pts),
        "clusters": clusters_out,
    }

    with open(CLUSTERS_JSON, "w", encoding="utf8") as f:
        json.dump(save_obj, f, indent=2)

    print(f"✅ Saved {CLUSTERS_JSON} ({len(clusters_out)} clusters)")

    # ---- UPDATE PAYLOADS ----
    print("✏️ Updating Qdrant payloads: event_id, canonical, event_size ...")

    for idx, qid in enumerate(q_ids):
        label = label_for_idx.get(idx, -1)
        size = clusters_out.get(str(label), {}).get("size", 1)
        is_canon = clusters_out.get(str(label), {}).get("canonical_point_id", "") == str(qid)

        newp = payloads[idx].copy()
        newp.update({
            "event_id": label,
            "event_size": size,
            "canonical": bool(is_canon),
        })

        client.set_payload(
            collection_name=COLLECTION_NAME,
            points=[qid],
            payload=newp,
        )

    print("🎯 Done!")
    print("Example clusters:")
    for k, v in list(clusters_out.items())[:5]:
        print(f" - {k}: size={v['size']} canonical={v['canonical_url']}")


if __name__ == "__main__":
    run()
