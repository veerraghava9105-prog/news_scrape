# project/scripts/ml_service.py
# Run with:
# uvicorn project.scripts.ml_service:app --reload --port 8000

from typing import List, Optional, Dict
import json
import time
from pathlib import Path
import re

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
from pymongo import MongoClient  # <--- ADDED


# ------------------ CONFIG ------------------

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "news_articles"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
VECTOR_SIZE = 768

DEFAULT_SAME_STORY_THRESHOLD = 0.82
DEFAULT_CLUSTER_ASSIGN_THRESHOLD = 0.72

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CLUSTERS_JSON = DATA_DIR / "clusters.json"

app = FastAPI(title="News ML Service", version="2.4.0")


# ------------------ Pydantic Schemas ------------------

class HealthResponse(BaseModel):
    status: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 5


class SearchResult(BaseModel):
    id: str
    score: float
    title: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    domain: Optional[str] = None
    published: Optional[str] = None
    cluster: Optional[int] = None


class SameStoryByIdRequest(BaseModel):
    id: str
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
    has_clusters: bool
    assigned_cluster: int
    best_score: float
    num_clusters_considered: int


# ======= NEW SUMMARIZATION SCHEMAS =======

class SummarizeClusterRequest(BaseModel):
    id: str
    max_sentences: int = 4
    per_article_limit: int = 5


class SummarizeClusterResponse(BaseModel):
    anchor: SearchResult
    items: List[SearchResult]
    summary: str
    highlights: Dict[str, List[str]]
    took_seconds: float


class EventIdRequest(BaseModel):
    url: str


# ------------------ Startup ------------------

@app.on_event("startup")
def startup_event():

    print("🌍 booting ML service for all souls…")

    # QDRANT
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    collections = {c.name for c in client.get_collections().collections}

    if COLLECTION_NAME not in collections:
        print(f"⚠ Missing collection '{COLLECTION_NAME}'. Creating…")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print("✅ Collection created")

    # MODEL
    try:
        model = SentenceTransformer(MODEL_NAME, device="cuda")
        print("🔥 Model on GPU")
    except Exception:
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        print("⚡ Model on CPU")

    # CLUSTERS (optional)
    cluster_centroids = None
    cluster_labels = None
    if CLUSTERS_JSON.exists():
        try:
            with open(CLUSTERS_JSON, "r", encoding="utf8") as f:
                data = json.load(f)

            clusters = data.get("clusters", {})
            centroids = []
            labels = []
            for label_str, meta in clusters.items():
                labels.append(int(label_str))
                centroids.append(np.array(meta["centroid"], dtype=np.float32))

            if centroids:
                cluster_centroids = np.stack(centroids)
                cluster_labels = np.array(labels)

            print(f"✅ Loaded {len(labels)} cluster centroids")
        except Exception as e:
            print("⚠ Failed loading clusters:", e)

    # MONGO FIX 🔥🔥🔥
    try:
        mongo_client = MongoClient("mongodb://localhost:27017")
        mongo_db = mongo_client["elevenSenseDB"]   # <-- EXACT DB NAME
        mongo_col = mongo_db["posts"]              # <-- EXACT COLLECTION

        app.state.mongo = mongo_col
        print("🍃 Connected to MongoDB elevenSenseDB.posts")
    except Exception as e:
        print("❌ Mongo connect failed:", e)
        app.state.mongo = None

    # SAVE TO STATE
    app.state.client = client
    app.state.model = model
    app.state.cluster_centroids = cluster_centroids
    app.state.cluster_labels = cluster_labels

    print("✅ ML service awake and alive")


# ------------------ Helpers ------------------

def embed_text(text: str) -> np.ndarray:
    model = app.state.model
    vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
    return vec.astype(np.float32)


def scoredpoint_to_result(p: ScoredPoint) -> SearchResult:
    payload = p.payload or {}
    url = payload.get("url")

    return SearchResult(
        id=url if url else str(p.id),
        score=float(p.score),
        title=payload.get("title"),
        url=url,
        source=payload.get("source"),
        domain=payload.get("domain"),
        published=payload.get("published"),
        cluster=payload.get("cluster"),
    )


def fetch_anchor_by_url(client: QdrantClient, url: str) -> ScoredPoint:
    flt = Filter(must=[FieldCondition(key="url", match=MatchValue(value=url))])

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=flt,
        limit=1,
        with_payload=True,
        with_vectors=True,
    )

    if not points:
        raise HTTPException(404, "URL not found")

    p = points[0]

    return ScoredPoint(
        id=p.id,
        payload=p.payload,
        vector=p.vector,
        score=1.0,
        version=0
    )


# ===== SUMMARIZATION HELPERS =====

def split_sentences(text: str):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def embed_sentences(sentences: list) -> np.ndarray:
    vecs = app.state.model.encode(sentences, normalize_embeddings=True, convert_to_numpy=True)
    return np.array(vecs, dtype=np.float32)


# ------------------ ROUTES ------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")


@app.post("/search_text", response_model=List[SearchResult])
def search_text(body: SearchRequest):
    client = app.state.client
    vec = embed_text(body.query).tolist()

    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        limit=body.limit,
        with_payload=True,
        with_vectors=False
    )

    return [scoredpoint_to_result(p) for p in res.points]


@app.post("/same_story_by_id", response_model=SameStoryResponse)
def same_story_by_id(body: SameStoryByIdRequest):

    client = app.state.client
    anchor = fetch_anchor_by_url(client, body.id)
    anchor_vec = np.array(anchor.vector, dtype=np.float32)

    t0 = time.time()
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=anchor_vec.tolist(),
        limit=body.limit,
        with_payload=True,
        with_vectors=False
    )

    hits = [p for p in res.points if p.score >= body.min_score]
    hits.sort(key=lambda p: p.score, reverse=True)

    items = [scoredpoint_to_result(p) for p in hits]

    took = time.time() - t0

    return SameStoryResponse(
        anchor=scoredpoint_to_result(anchor),
        items=items,
        took_seconds=round(took, 3),
        took_minutes=round(took / 60, 3),
    )


@app.post("/same_story_by_text", response_model=SameStoryResponse)
def same_story_by_text(body: SameStoryByTextRequest):

    client = app.state.client
    vec = embed_text(body.text)

    t0 = time.time()

    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec.tolist(),
        limit=body.limit,
        with_payload=True,
        with_vectors=False
    )

    hits = [p for p in res.points if p.score >= body.min_score]
    hits.sort(key=lambda p: p.score, reverse=True)

    items = [scoredpoint_to_result(p) for p in hits]

    took = time.time() - t0

    return SameStoryResponse(
        anchor=SearchResult(id="query-text", score=1.0, title=body.text),
        items=items,
        took_seconds=round(took, 3),
        took_minutes=round(took / 60, 3),
    )


@app.post("/route_cluster", response_model=ClusterAssignResponse)
def route_cluster(body: ClusterAssignRequest):

    centroids = app.state.cluster_centroids
    labels = app.state.cluster_labels

    if centroids is None:
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

    assigned = best_label if best_score >= body.min_score else -1

    return ClusterAssignResponse(
        has_clusters=True,
        assigned_cluster=assigned,
        best_score=best_score,
        num_clusters_considered=len(labels),
    )


@app.post("/event-id")
def get_event_id(body: EventIdRequest):

    client = app.state.client

    flt = Filter(must=[FieldCondition(key="url", match=MatchValue(value=body.url))])

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=flt,
        limit=1,
        with_payload=True,
        with_vectors=False
    )

    if not points:
        return {"event_id": None}

    payload = points[0].payload or {}
    return {"event_id": payload.get("event_id")}


# ========== SUMMARIZE CLUSTER (FIXED + MONGO WORKING) ===========

@app.post("/summarize_cluster", response_model=SummarizeClusterResponse)
def summarize_cluster(body: SummarizeClusterRequest):

    client = app.state.client
    mongo = app.state.mongo
    if mongo is None:
        raise HTTPException(500, "MongoDB not available")

    t0 = time.time()

    # 1. Fetch anchor
    anchor = fetch_anchor_by_url(client, body.id)

    # 2. Query similar articles
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=np.array(anchor.vector, dtype=np.float32).tolist(),
        limit=body.per_article_limit + 40,
        with_payload=True,
        with_vectors=False
    )

    neighbors = [p for p in res.points if p.score >= DEFAULT_SAME_STORY_THRESHOLD]
    neighbors.sort(key=lambda p: p.score, reverse=True)

    items = [scoredpoint_to_result(p) for p in neighbors]

    # 3. Fetch REAL TEXT from MongoDB
    url_to_text = {}
    for p in neighbors:
        pl = p.payload or {}
        url = pl.get("url")
        if not url:
            continue

        doc = mongo.find_one({"url": url})
        txt = doc.get("text") if doc else ""

        url_to_text[url] = txt

    # 4. Build sentence candidates
    cand = []
    for url, txt in url_to_text.items():
        if not txt:
            continue

        sents = split_sentences(txt)[: body.per_article_limit * 4]
        for s in sents[: body.per_article_limit]:
            cand.append((url, s))

    if not cand:
        return SummarizeClusterResponse(
            anchor=scoredpoint_to_result(anchor),
            items=items,
            summary="No text available",
            highlights={},
            took_seconds=round(time.time() - t0, 3)
        )

    sentences = [s for (_, s) in cand]
    urls = [u for (u, _) in cand]

    # 5. Embed sentences
    vecs = embed_sentences(sentences)
    centroid = vecs.mean(axis=0, keepdims=True)
    sims = (vecs @ centroid.T).squeeze()

    idxs = np.argsort(-sims)

    # pick top
    chosen = []
    per_source = {}
    for i in idxs:
        if len(chosen) >= body.max_sentences:
            break
        url = urls[i]
        if per_source.get(url, 0) >= 2:  # max 2 per article
            continue
        chosen.append(i)
        per_source[url] = per_source.get(url, 0) + 1

    chosen_sorted = sorted(chosen)
    summary = " ".join(sentences[i] for i in chosen_sorted)

    # highlights
    highlights = {}
    for url in set(urls):
        idxs_url = [i for i, u in enumerate(urls) if u == url]
        idxs_url = sorted(idxs_url, key=lambda i: -sims[i])[:2]
        highlights[url] = [sentences[i] for i in idxs_url]

    return SummarizeClusterResponse(
        anchor=scoredpoint_to_result(anchor),
        items=items,
        summary=summary,
        highlights=highlights,
        took_seconds=round(time.time() - t0, 3)
    )


print("REGISTERED ROUTES:")
for r in app.routes:
    print(r.path)
