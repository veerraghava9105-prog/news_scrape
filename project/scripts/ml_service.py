# project/scripts/ml_service.py
# Run with:
#   uvicorn scripts.ml_service:app --reload --port 8000

from typing import List, Optional, Dict

import json
import time
from pathlib import Path

import hdbscan
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer

# ------------------ CONFIG ------------------

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "news_articles"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
VECTOR_SIZE = 768

# threshold for “same story” grouping (cosine-ish score)
DEFAULT_SAME_STORY_THRESHOLD = 0.82

# threshold for assigning to an existing topic cluster
DEFAULT_CLUSTER_ASSIGN_THRESHOLD = 0.72

BASE_DIR = Path(__file__).resolve().parents[1]  # project/
DATA_DIR = BASE_DIR / "data"
CLUSTERS_JSON = DATA_DIR / "clusters.json"

app = FastAPI(title="News ML Service", version="2.1.0")


# ------------------ Pydantic Schemas ------------------

class HealthResponse(BaseModel):
    status: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 5


class SearchResult(BaseModel):
    # 👉 This is the PUBLIC ID we expose to the outside world = URL
    id: str            # URL
    score: float
    title: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    domain: Optional[str] = None
    published: Optional[str] = None
    cluster: Optional[int] = None


class SameStoryByIdRequest(BaseModel):
    # 👉 Here id = URL (not Qdrant UUID)
    id: str            # URL of the anchor article
    limit: int = 25
    min_score: float = DEFAULT_SAME_STORY_THRESHOLD


class SameStoryByTextRequest(BaseModel):
    text: str
    limit: int = 25
    min_score: float = DEFAULT_SAME_STORY_THRESHOLD


class SameStoryResponse(BaseModel):
    anchor: SearchResult
    items: List[SearchResult]
    took_seconds: float
    took_minutes: float


class ClusterAssignRequest(BaseModel):
    text: str
    min_score: float = DEFAULT_CLUSTER_ASSIGN_THRESHOLD


class ClusterAssignResponse(BaseModel):
    has_clusters: bool           # do we have centroids loaded at all?
    assigned_cluster: int        # -1 if nothing passes threshold
    best_score: float
    num_clusters_considered: int


class ClusterRequest(BaseModel):
    ids: List[str]               # still Qdrant UUIDs – debug only
    min_cluster_size: int = 5


class ClusterResultItem(BaseModel):
    id: str
    label: int


# ------------------ Startup ------------------

@app.on_event("startup")
def startup_event():
    """
    - Connect to Qdrant
    - Ensure collection exists
    - Load sentence transformer
    - Load cluster centroids (if clusters.json present)
    """
    print("🌍 booting ML service for all souls…")

    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

    # Ensure collection exists (safe if already there)
    collections = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in collections:
        print(f"⚠️  Collection '{COLLECTION_NAME}' not found. Creating empty one...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"✅ Created collection '{COLLECTION_NAME}'")

    # Model
    try:
        model = SentenceTransformer(MODEL_NAME, device="cuda")
        print("🔥 Model loaded on GPU")
    except Exception:
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        print("⚡ GPU not available → model on CPU")

    # Load clusters metadata (centroids etc.)
    cluster_centroids: Optional[np.ndarray] = None
    cluster_labels: Optional[np.ndarray] = None

    if CLUSTERS_JSON.exists():
        try:
            with open(CLUSTERS_JSON, "r", encoding="utf8") as f:
                data = json.load(f)
            clusters = data.get("clusters", {})
            labels = []
            centroids = []
            for label_str, meta in clusters.items():
                labels.append(int(label_str))
                centroids.append(np.array(meta["centroid"], dtype=np.float32))
            if centroids:
                cluster_centroids = np.stack(centroids, axis=0)
                cluster_labels = np.array(labels, dtype=np.int32)
            print(f"✅ Loaded {len(labels)} cluster centroids from clusters.json")
        except Exception as e:
            print(f"⚠️ Failed to load clusters.json: {e}")

    app.state.client = client
    app.state.model = model
    app.state.cluster_centroids = cluster_centroids
    app.state.cluster_labels = cluster_labels

    print("✅ ML service awake and alive ✅")


# ------------------ Helpers ------------------

def embed_text(text: str) -> np.ndarray:
    """Embed a single string into a normalized vector."""
    model: SentenceTransformer = app.state.model
    vec = model.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)[0]
    return vec


def scoredpoint_to_result(p: ScoredPoint) -> SearchResult:
    """
    Convert a Qdrant point into API result.
    PUBLIC id = url (fallback to Qdrant ID if missing).
    """
    payload = p.payload or {}
    url = payload.get("url")
    public_id = url if url is not None else str(p.id)

    return SearchResult(
        id=public_id,
        score=float(p.score),
        title=payload.get("title"),
        url=url,
        source=payload.get("source"),
        domain=payload.get("domain"),
        published=payload.get("published"),
        cluster=payload.get("cluster"),
    )


def fetch_anchor_by_url(client: QdrantClient, url: str) -> ScoredPoint:
    """
    Given a URL, find the anchor point (with vectors & payload).
    We use scroll + filter on payload.url == url.
    """
    flt = Filter(
        must=[
            FieldCondition(
                key="url",
                match=MatchValue(value=url),
            )
        ]
    )

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=flt,
        limit=1,
        with_payload=True,
        with_vectors=True,
    )
    if not points:
        raise HTTPException(status_code=404, detail="Article with this URL not found")

    # scroll returns 'Record', not ScoredPoint, but has same fields we need
    p = points[0]
    # wrap into fake ScoredPoint-like object for type compatibility
    sp = ScoredPoint(
        id=p.id,
        payload=p.payload,
        vector=p.vector,
        score=1.0,
        version=0,
    )
    return sp


# ------------------ Routes ------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Simple heartbeat so your backend can check if service is alive."""
    return HealthResponse(status="ok")


# -------- 1) Semantic search over all news --------

@app.post("/search_text", response_model=List[SearchResult])
def search_text(body: SearchRequest):
    """
    Semantic search on `news_articles` using the same BGE model
    that was used during ingestion.

    Body:
    {
      "query": "COP30 climate finance Brazil draft",
      "limit": 5
    }

    Response items:
    {
      "id": "<url here>",      # <= URL as id
      "score": 0.93,
      "title": "...",
      "url": "...",
      "source": "...",
      "domain": "...",
      "published": "...",
      "cluster": 8
    }
    """
    client: QdrantClient = app.state.client

    query_vec = embed_text(body.query).tolist()

    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=body.limit,
        with_payload=True,
        with_vectors=False,
    )
    hits: List[ScoredPoint] = res.points

    return [scoredpoint_to_result(p) for p in hits]


# -------- 2) “Same story” group by URL (full coverage) --------

@app.post("/same_story_by_id", response_model=SameStoryResponse)
def same_story_by_id(body: SameStoryByIdRequest):
    """
    FULL COVERAGE-style endpoint.

    Here `id` == URL of the anchor article.

    Body:
    {
      "id": "https://telanganatoday.com/top-maoist-commander-madivi-hidma-killed-in-ap-encounter",
      "limit": 25,
      "min_score": 0.82
    }
    """
    client: QdrantClient = app.state.client

    # 1) find the anchor article by URL
    anchor_sp = fetch_anchor_by_url(client, body.id)
    anchor_vec = np.array(anchor_sp.vector, dtype=np.float32)
    anchor_payload = anchor_sp.payload or {}

    t0 = time.time()

    # 2) vector search around this anchor
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=anchor_vec.tolist(),
        limit=body.limit,
        with_payload=True,
        with_vectors=False,
    )
    hits: List[ScoredPoint] = res.points

    # 3) filter by min_score and sort
    filtered = [p for p in hits if p.score is not None and p.score >= body.min_score]
    filtered.sort(key=lambda p: p.score, reverse=True)

    items = [scoredpoint_to_result(p) for p in filtered]

    anchor_result = SearchResult(
        id=anchor_payload.get("url", body.id),
        score=1.0,
        title=anchor_payload.get("title"),
        url=anchor_payload.get("url"),
        source=anchor_payload.get("source"),
        domain=anchor_payload.get("domain"),
        published=anchor_payload.get("published"),
        cluster=anchor_payload.get("cluster"),
    )

    took = time.time() - t0
    return SameStoryResponse(
        anchor=anchor_result,
        items=items,
        took_seconds=round(took, 3),
        took_minutes=round(took / 60.0, 3),
    )


# -------- 3) “Same story” group by free text (no URL yet) --------

@app.post("/same_story_by_text", response_model=SameStoryResponse)
def same_story_by_text(body: SameStoryByTextRequest):
    """
    For raw text (title or summary), find a same-story group.

    Body:
    {
      "text": "Smriti Mandhana’s wedding delayed after father falls ill",
      "limit": 25,
      "min_score": 0.82
    }
    """
    client: QdrantClient = app.state.client

    query_vec = embed_text(body.text)
    t0 = time.time()

    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec.tolist(),
        limit=body.limit,
        with_payload=True,
        with_vectors=False,
    )
    hits: List[ScoredPoint] = res.points

    filtered = [p for p in hits if p.score is not None and p.score >= body.min_score]
    filtered.sort(key=lambda p: p.score, reverse=True)

    items = [scoredpoint_to_result(p) for p in filtered]

    anchor = SearchResult(
        id="query-text",
        score=1.0,
        title=body.text,
        url=None,
        source=None,
        domain=None,
        published=None,
        cluster=None,
    )

    took = time.time() - t0
    return SameStoryResponse(
        anchor=anchor,
        items=items,
        took_seconds=round(took, 3),
        took_minutes=round(took / 60.0, 3),
    )


# -------- 4) Cluster assignment using centroids.json --------

@app.post("/route_cluster", response_model=ClusterAssignResponse)
def route_cluster(body: ClusterAssignRequest):
    """
    Given article text (title or full text), assign it to an existing
    topic cluster based on centroids from clusters.json.

    Body:
    {
      "text": "US Fed hikes interest rates again",
      "min_score": 0.72
    }
    """
    centroids: Optional[np.ndarray] = app.state.cluster_centroids
    labels: Optional[np.ndarray] = app.state.cluster_labels

    if centroids is None or labels is None or len(centroids) == 0:
        return ClusterAssignResponse(
            has_clusters=False,
            assigned_cluster=-1,
            best_score=0.0,
            num_clusters_considered=0,
        )

    vec = embed_text(body.text)
    sims = centroids @ vec
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_label = int(labels[best_idx])

    if best_score < body.min_score:
        assigned = -1
    else:
        assigned = best_label

    return ClusterAssignResponse(
        has_clusters=True,
        assigned_cluster=assigned,
        best_score=best_score,
        num_clusters_considered=int(len(labels)),
    )


# -------- 5) Small “ad-hoc” clustering of given IDs --------

@app.post("/cluster_ids", response_model=List[ClusterResultItem])
def cluster_ids(body: ClusterRequest):
    """
    Cluster a set of specific article IDs with HDBSCAN.
    NOTE: here ids are still Qdrant UUIDs, used only for debugging.
    """
    client: QdrantClient = app.state.client

    if not body.ids:
        return []

    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=body.ids,
        with_payload=False,
        with_vectors=True,
    )

    if not points:
        return []

    vectors = np.array([p.vector for p in points], dtype=np.float32)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=body.min_cluster_size,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(vectors)

    out: List[ClusterResultItem] = []
    for p, lbl in zip(points, labels):
        out.append(ClusterResultItem(id=str(p.id), label=int(lbl)))
    return out
