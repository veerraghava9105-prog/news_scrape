# cluster_service.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import json
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from .utils_io import read_jsonl
import os

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
EMB_NPZ = DATA / "embeddings.npz"
INDEX_JSON = DATA / "embeddings_index.json"
CLUSTERS_JSON = DATA / "clusters.json"

app = FastAPI(title="Clustering & Embedding Service")

# Globals
MODEL = None
EMBEDDINGS = None   # numpy array (N, D)
INDEX = []          # metadata list
USE_HDBSCAN = False

# Config defaults
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_MIN_SAMPLES = 1
DEFAULT_CLUSTER_METHOD = "hdbscan"  # or "agglomerative" or "dbscan"


class ClusterOptions(BaseModel):
    method: Optional[str] = DEFAULT_CLUSTER_METHOD
    min_cluster_size: Optional[int] = DEFAULT_MIN_CLUSTER_SIZE
    min_samples: Optional[int] = DEFAULT_MIN_SAMPLES
    distance_threshold: Optional[float] = None  # for agglo
    # Add more params as needed

@app.on_event("startup")
def startup():
    global MODEL, EMBEDDINGS, INDEX, USE_HDBSCAN
    print("🔥 Loading embedder model (deferred if needed).")
    MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # only needed if we embed on the fly later

    # load embeddings if already saved
    if EMB_NPZ.exists():
        print("🔁 Loading embeddings from disk:", EMB_NPZ)
        arr = np.load(str(EMB_NPZ))["embeddings"]
        EMBEDDINGS = arr.astype(np.float32)
        print("⚡ embeddings shape:", EMBEDDINGS.shape)
    else:
        print("⚠️ embeddings file not found. Run embedder first.")

    if INDEX_JSON.exists():
        INDEX = json.load(open(INDEX_JSON, "r", encoding="utf8"))
    else:
        INDEX = []

    # check if hdbscan available
    try:
        import hdbscan  # noqa
        USE_HDBSCAN = True
        print("✅ hdbscan available, will use hdbscan by default.")
    except Exception:
        USE_HDBSCAN = False
        print("⚠️ hdbscan not available; falling back on sklearn clustering.")

def pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (m,d), b: (n,d) -> returns (m,n) cosine via dot because we L2-normalized.
    return (a @ b.T).astype(np.float32)

@app.post("/knn")
def knn(query_idx: int, top_k: int = 10):
    if EMBEDDINGS is None:
        raise HTTPException(500, "embeddings not loaded")
    if not (0 <= query_idx < EMBEDDINGS.shape[0]):
        raise HTTPException(400, "query_idx out of bounds")
    vec = EMBEDDINGS[[query_idx]]
    sims = (vec @ EMBEDDINGS.T).squeeze()
    idxs = np.argsort(-sims)[: top_k + 1]
    # skip itself
    res = [(int(i), float(sims[int(i)])) for i in idxs if int(i) != query_idx][:top_k]
    return {"query_idx": query_idx, "neighbors": res}

@app.post("/cluster")
def cluster(opts: ClusterOptions):
    """
    Run clustering with options. Saves clusters.json
    """
    global EMBEDDINGS
    if EMBEDDINGS is None:
        raise HTTPException(500, "embeddings not found, run embedder first")

    X = EMBEDDINGS  # normalized

    method = opts.method or DEFAULT_CLUSTER_METHOD
    clusters = None

    if method == "hdbscan" and USE_HDBSCAN:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=opts.min_cluster_size or DEFAULT_MIN_CLUSTER_SIZE,
                                    min_samples=opts.min_samples or DEFAULT_MIN_SAMPLES,
                                    metric="euclidean",
                                    cluster_selection_method="eom")
        labels = clusterer.fit_predict(X)
    elif method == "dbscan":
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=opts.min_samples or 0.5, min_samples=opts.min_cluster_size or 2,
                        metric="cosine").fit_predict(X)
    elif method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        # Agglomerative expects distance; we transform via 1-cosine
        dists = 1.0 - (X @ X.T)
        # To avoid memory for huge N, use a different method or approximate
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=opts.distance_threshold or 0.7,
                                        linkage="average", compute_full_tree=True, affinity='precomputed')
        labels = model.fit_predict(dists)
    else:
        raise HTTPException(400, f"Unknown clustering method {method}")

    # collect clusters
    clusters = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(idx)

    # build cluster objects, skip noise (HDBSCAN labels -1 are noise; keep if needed)
    cluster_list = []
    for cid, member_idxs in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        rep_idx = member_idxs[0]
        rep = INDEX[rep_idx]["title"] if INDEX and len(INDEX) > rep_idx else f"article_{rep_idx}"
        cluster_list.append({
            "cluster_id": int(cid),
            "member_count": len(member_idxs),
            "members": member_idxs,
            "representative": rep
        })

    with open(str(CLUSTERS_JSON), "w", encoding="utf8") as f:
        json.dump({"clusters": cluster_list}, f, ensure_ascii=False, indent=2)

    return {"clusters_count": len(cluster_list), "saved": str(CLUSTERS_JSON)}

@app.get("/clusters")
def get_clusters():
    if not CLUSTERS_JSON.exists():
        raise HTTPException(404, "clusters.json not found; run /cluster")
    return json.load(open(CLUSTERS_JSON, "r", encoding="utf8"))

if __name__ == "__main__":
    uvicorn.run("project.scripts.cluster_service:app", host="127.0.0.1", port=5000, reload=True)
