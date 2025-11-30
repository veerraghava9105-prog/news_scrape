"""
Topic-aware clustering job for the news vector universe.

Modes (auto-detected):

1) INITIAL FULL CLUSTERING
   - Triggered when data/clusters.json does NOT exist.
   - Fetches ALL points (id, vector, payload) from Qdrant.
   - Runs HDBSCAN on full set.
   - Computes centroid + medoid per cluster.
   - Infers a high-level TOPIC CATEGORY for each cluster:
       politics, economy, tech, geopolitics, national, cinema, sports, etc.
   - Writes `cluster` (int) and `category` (str) into each point's payload.
   - Saves metadata to data/clusters.json.

2) NOISE-ONLY RE-CLUSTERING
   - Triggered when clusters.json exists.
   - Fetches ONLY points with cluster == -1 (noise/unassigned) or missing.
   - Runs HDBSCAN on this subset.
   - For each new cluster, assigns NEW global cluster IDs
     (continuing from previous max label).
   - Infers topic category for each new cluster.
   - Writes labels + categories back into Qdrant.
   - Updates clusters.json with appended cluster metadata.

→ Use this for "topic buckets" like politics / tech / sports.
→ Use your ML service's /same_story* endpoints for per-event grouping
   (multi-publisher coverage of the exact same story).
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import hdbscan
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# ----------------- CONFIG ----------------- #

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "news_articles"
VECTOR_SIZE = 768
SCROLL_PAGE_SIZE = 10_000

# HDBSCAN config
MIN_CLUSTER_SIZE = 15      # tune if you want bigger/smaller topic buckets
MIN_SAMPLES = None
HDBSCAN_METRIC = "euclidean"

BASE = Path(__file__).resolve().parents[1]  # project/
DATA_DIR = BASE / "data"
CLUSTERS_JSON = DATA_DIR / "clusters.json"

# Topic categories – tweak/extend as you like
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "politics": [
        "election", "vote", "poll", "parliament", "congress", "bjp",
        "minister", "mp ", "mla", "chief minister", "cm ",
        "government", "govt", "policy", "bill ", "cabinet", "politics",
    ],
    "economy": [
        "gdp", "inflation", "economic", "economy", "sensex", "nifty",
        "stock market", "stock", "bond", "loan", "bank", "rbi",
        "federal reserve", "fed ", "interest rate", "finance", "budget",
    ],
    "tech": [
        "ai ", "artificial intelligence", "machine learning", "startup",
        "app ", "software", "hardware", "semiconductor", "chip", "nvidia",
        "microsoft", "google", "apple", "iphone", "android", "data",
        "cyber", "hacker", "hack ", "security", "social media", "tech",
    ],
    "geopolitics": [
        "summit", "cop30", "cop 30", "climate summit", "climate talks",
        "border", "dispute", "military", "defence", "defense", "nato",
        "china", "russia", "ukraine", "israel", "palestine", "gaza",
        "white house", "pentagon", "united nations", "u.n.", "un ",
        "sanction", "trade war", "geopolitics",
    ],
    "national": [
        "india ", "indian ", "delhi", "mumbai", "hyderabad", "bengaluru",
        "kolkata", "chennai", "state", "district", "assembly",
        "ls poll", "lok sabha", "rajya sabha", "national",
    ],
    "cinema": [
        "film", "movie", "box office", "actor", "actress", "director",
        "trailer", "teaser", "song", "album", "bollywood", "tollywood",
        "kollywood", "hollywood", "ott ", "netflix", "prime video",
        "cinema", "biopic",
    ],
    "sports": [
        "match", "tournament", "series", "world cup", "championship",
        "league", "fixture", "score", "goal", "wicket", "run chase",
        "cricket", "football", "soccer", "tennis", "badminton",
        "olympics", "coach", "captain", "team", "ipl ",
        "nba", "fifa",
    ],
    "business": [
        "startup", "valuation", "funding", "ipo", "merger", "acquisition",
        "corporate", "earnings", "profit", "revenue", "losses",
        "company", "shareholder",
    ],
    "crime": [
        "murder", "killed", "shot dead", "arrested", "custody", "crime",
        "theft", "robbery", "fraud", "scam", "chargesheet", "fir ",
        "police case",
    ],
    "science_health": [
        "research", "study finds", "scientists", "lab ", "vaccine",
        "covid", "disease", "virus", "health", "hospital", "aiims",
        "genetic", "astronomy", "space", "nasa", "isro",
    ],
}

CATEGORY_FALLBACK = "misc"


# ----------------- DATA STRUCTURES ----------------- #

@dataclass
class ClusterMeta:
    label: int
    size: int
    centroid: List[float]
    medoid_id: str
    category: str  # high-level topic label


@dataclass
class ClusteringSummary:
    num_points: int
    num_clusters: int
    num_noise: int
    clusters: Dict[str, ClusterMeta]  # key: label as string


# ----------------- COMMON HELPERS ----------------- #

def ensure_collection(client: QdrantClient) -> None:
    collections = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in collections:
        print(f"⚠️ Collection '{COLLECTION_NAME}' not found. Creating empty...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"✅ Created empty collection '{COLLECTION_NAME}'.")
    else:
        info = client.get_collection(COLLECTION_NAME)
        size = info.config.params.vectors.size
        if size != VECTOR_SIZE:
            print(f"❗ Warning: collection dim={size}, expected {VECTOR_SIZE}.")
        print(f"✅ Collection '{COLLECTION_NAME}' exists.")


def run_hdbscan(vectors: np.ndarray) -> np.ndarray:
    print("🧠 Running HDBSCAN over topic space...")
    t0 = time.time()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(vectors)

    dur = time.time() - t0
    dur_min = dur / 60
    n_clusters = len(set(int(l) for l in labels if l != -1))
    n_noise = int(np.sum(labels == -1))

    print(f"✅ HDBSCAN done in {dur:.1f}s ({dur_min:.2f} min)")
    print(f"   → clusters (excluding noise): {n_clusters}")
    print(f"   → noise points: {n_noise}")

    return labels


def compute_centroid_and_medoid(
    cluster_vectors: np.ndarray,
    cluster_ids: List[str],
) -> Tuple[List[float], str]:
    centroid = cluster_vectors.mean(axis=0)
    # normalize for cosine
    norm = np.linalg.norm(centroid) + 1e-9
    centroid = (centroid / norm).astype(np.float32)

    diffs = cluster_vectors - centroid
    dists = np.linalg.norm(diffs, axis=1)
    medoid_idx = int(dists.argmin())
    medoid_id = cluster_ids[medoid_idx]
    return centroid.tolist(), medoid_id


# ----------------- CATEGORY INFERENCE ----------------- #

def score_text_categories(text: str) -> Dict[str, int]:
    """Return simple keyword-based scores per category for a title/text."""
    text_l = text.lower()
    scores: Dict[str, int] = {cat: 0 for cat in CATEGORY_KEYWORDS.keys()}

    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_l:
                scores[cat] += 1
    return scores


def infer_cluster_category(titles: List[str]) -> str:
    """Aggregate keyword scores over all titles in the cluster."""
    if not titles:
        return CATEGORY_FALLBACK

    agg_scores: Dict[str, int] = {cat: 0 for cat in CATEGORY_KEYWORDS.keys()}

    for t in titles:
        s = score_text_categories(t)
        for cat, val in s.items():
            agg_scores[cat] += val

    best_cat = max(agg_scores.items(), key=lambda kv: kv[1])
    if best_cat[1] <= 0:
        return CATEGORY_FALLBACK
    return best_cat[0]


# ----------------- FULL FETCH (INITIAL) ----------------- #

def fetch_all_points(
    client: QdrantClient,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Fetch ALL points' ids, vectors, and titles from Qdrant.
    """
    print("🔎 Fetching ALL points from Qdrant (ids + vectors + titles)...")
    ids: List[str] = []
    vecs: List[List[float]] = []
    titles: List[str] = []

    offset = None
    page_idx = 0

    while True:
        page_idx += 1
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        for p in points:
            ids.append(str(p.id))
            vecs.append(p.vector)
            payload = p.payload or {}
            titles.append(payload.get("title", "") or "")

        print(f"  📦 Page {page_idx}: {len(points)} points (total {len(ids)})")
        if offset is None:
            break

    if not ids:
        print("🛑 No points found. Nothing to cluster.")
        return [], np.empty((0, VECTOR_SIZE), dtype=np.float32), []

    arr = np.array(vecs, dtype=np.float32)
    print(f"✅ Total points: {len(ids)}, dim={arr.shape[1]}")
    return ids, arr, titles


def build_full_summary(
    ids: List[str],
    vectors: np.ndarray,
    titles: List[str],
    labels: np.ndarray,
) -> ClusteringSummary:
    print("📊 Building full clustering summary with topic categories...")
    meta: Dict[str, ClusterMeta] = {}

    unique_labels = sorted({int(l) for l in labels if l != -1})
    num_points = len(ids)
    num_clusters = len(unique_labels)
    num_noise = int(np.sum(labels == -1))

    for label in unique_labels:
        mask = (labels == label)
        idxs = np.where(mask)[0]
        cluster_ids = [ids[i] for i in idxs]
        cluster_vecs = vectors[mask]
        cluster_titles = [titles[i] for i in idxs]

        centroid, medoid_id = compute_centroid_and_medoid(cluster_vecs, cluster_ids)
        category = infer_cluster_category(cluster_titles)

        cm = ClusterMeta(
            label=label,
            size=len(cluster_ids),
            centroid=centroid,
            medoid_id=medoid_id,
            category=category,
        )
        meta[str(label)] = cm
        print(
            f"   • cluster {label:>3}: size={cm.size:>4}, "
            f"category={cm.category:<12}, medoid_id={cm.medoid_id[:8]}..."
        )

    summary = ClusteringSummary(
        num_points=num_points,
        num_clusters=num_clusters,
        num_noise=num_noise,
        clusters=meta,
    )
    print(f"✅ Full summary → clusters={num_clusters}, noise={num_noise}")
    return summary


# ----------------- NOISE FETCH (OUTLIERS ONLY) ----------------- #

def fetch_noise_points(
    client: QdrantClient,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Fetch points that currently have cluster == -1 (or no cluster).
    """
    print("🔎 Fetching NOISE / unassigned points (cluster == -1 or missing)...")
    ids: List[str] = []
    vecs: List[List[float]] = []
    titles: List[str] = []

    offset = None
    page_idx = 0

    while True:
        page_idx += 1
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=SCROLL_PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        for p in points:
            payload = p.payload or {}
            label = payload.get("cluster", -1)
            if label == -1:
                ids.append(str(p.id))
                vecs.append(p.vector)
                titles.append(payload.get("title", "") or "")

        print(
            f"  📦 Page {page_idx}: scanned {len(points)} points "
            f"(noise so far {len(ids)})"
        )
        if offset is None:
            break

    if not ids:
        print("🛑 No noise/unassigned points found.")
        return [], np.empty((0, VECTOR_SIZE), dtype=np.float32), []

    arr = np.array(vecs, dtype=np.float32)
    print(f"✅ Noise points: {len(ids)}, dim={arr.shape[1]}")
    return ids, arr, titles


# ----------------- CLUSTERS.JSON I/O ----------------- #

def load_existing_summary() -> Optional[ClusteringSummary]:
    if not CLUSTERS_JSON.exists():
        return None

    with open(CLUSTERS_JSON, "r", encoding="utf8") as f:
        data = json.load(f)

    clusters_dict: Dict[str, ClusterMeta] = {}
    for label, meta in data.get("clusters", {}).items():
        clusters_dict[label] = ClusterMeta(
            label=meta["label"],
            size=meta["size"],
            centroid=meta["centroid"],
            medoid_id=meta["medoid_id"],
            category=meta.get("category", CATEGORY_FALLBACK),
        )

    return ClusteringSummary(
        num_points=data["num_points"],
        num_clusters=data["num_clusters"],
        num_noise=data["num_noise"],
        clusters=clusters_dict,
    )


def save_summary(summary: ClusteringSummary) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_points": summary.num_points,
        "num_clusters": summary.num_clusters,
        "num_noise": summary.num_noise,
        "clusters": {
            label: asdict(meta)
            for label, meta in summary.clusters.items()
        },
    }
    with open(CLUSTERS_JSON, "w", encoding="utf8") as f:
        json.dump(payload, f, indent=2)
    print(f"💾 Wrote cluster metadata to {CLUSTERS_JSON}")


# ----------------- WRITE LABELS TO QDRANT ----------------- #

def write_labels_to_qdrant(
    client: QdrantClient,
    ids: List[str],
    labels: np.ndarray,
    summary: ClusteringSummary,
) -> None:
    """
    Update Qdrant payload with:
      - cluster: int
      - category: str (topic)
    Noise (cluster=-1) gets category="noise".
    """
    print("📝 Writing cluster labels + categories into Qdrant payloads...")

    label_to_ids: Dict[int, List[str]] = {}
    for point_id, label in zip(ids, labels):
        label_to_ids.setdefault(int(label), []).append(point_id)

    for label, group_ids in label_to_ids.items():
        if not group_ids:
            continue

        if label == -1:
            category = "noise"
        else:
            meta = summary.clusters.get(str(label))
            category = meta.category if meta is not None else CATEGORY_FALLBACK

        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"cluster": label, "category": category},
            points=group_ids,
        )
        print(
            f"   • cluster={label:>3}, category={category:<12} "
            f"→ updated {len(group_ids)} points"
        )

    print("✅ Finished updating Qdrant payloads.")


# ----------------- OUTLIER RECLUSTER MERGE ----------------- #

def extend_summary_with_noise_clusters(
    prev: ClusteringSummary,
    noise_ids: List[str],
    noise_vectors: np.ndarray,
    noise_titles: List[str],
    noise_labels: np.ndarray,
) -> Tuple[ClusteringSummary, np.ndarray]:
    """
    Given previous summary + HDBSCAN labels over noise subset,
    create new global cluster labels and update summary.
    """
    print("📊 Extending summary with new topic clusters from noise subset...")

    if len(noise_ids) == 0:
        print("🛑 No noise points to extend.")
        return prev, np.array([], dtype=np.int32)

    unique_local = sorted({int(l) for l in noise_labels if l != -1})
    if not unique_local:
        print("ℹ️ HDBSCAN on noise produced only noise again. No new clusters.")
        new_global_labels = np.full_like(noise_labels, -1, dtype=np.int32)
        new_summary = ClusteringSummary(
            num_points=prev.num_points,
            num_clusters=prev.num_clusters,
            num_noise=int(np.sum(new_global_labels == -1)),
            clusters=prev.clusters,
        )
        return new_summary, new_global_labels

    existing_labels = [int(k) for k in prev.clusters.keys()] if prev.clusters else []
    start_label = (max(existing_labels) + 1) if existing_labels else 0

    local_to_global: Dict[int, int] = {}
    for offset_idx, local_label in enumerate(unique_local):
        local_to_global[local_label] = start_label + offset_idx

    new_clusters: Dict[str, ClusterMeta] = dict(prev.clusters) if prev.clusters else {}

    for local_label in unique_local:
        mask = (noise_labels == local_label)
        idxs = np.where(mask)[0]
        cluster_ids = [noise_ids[i] for i in idxs]
        cluster_vecs = noise_vectors[mask]
        cluster_titles = [noise_titles[i] for i in idxs]

        centroid, medoid_id = compute_centroid_and_medoid(cluster_vecs, cluster_ids)
        category = infer_cluster_category(cluster_titles)

        global_label = local_to_global[local_label]
        cm = ClusterMeta(
            label=global_label,
            size=len(cluster_ids),
            centroid=centroid,
            medoid_id=medoid_id,
            category=category,
        )
        new_clusters[str(global_label)] = cm
        print(
            f"   • new cluster {global_label:>3}: size={cm.size:>4}, "
            f"category={cm.category:<12}, medoid_id={cm.medoid_id[:8]}..."
        )

    global_labels = []
    for lbl in noise_labels:
        if lbl == -1:
            global_labels.append(-1)
        else:
            global_labels.append(local_to_global[int(lbl)])
    global_labels = np.array(global_labels, dtype=np.int32)

    still_noise = int(np.sum(global_labels == -1))
    new_num_clusters = prev.num_clusters + len(unique_local)

    new_summary = ClusteringSummary(
        num_points=prev.num_points,
        num_clusters=new_num_clusters,
        num_noise=still_noise,
        clusters=new_clusters,
    )

    print(f"✅ Extended summary → total clusters={new_num_clusters}, noise={still_noise}")
    return new_summary, global_labels


# ----------------- MAIN ----------------- #

def main() -> None:
    print(
        f"\n🌌 Topic cluster maintenance for '{COLLECTION_NAME}' "
        f"on {QDRANT_HOST}:{QDRANT_PORT}"
    )
    t0 = time.time()

    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(client)

    existing_summary = load_existing_summary()

    # MODE 1: INITIAL FULL CLUSTERING
    if existing_summary is None:
        print("🚀 No clusters.json found → running INITIAL full topic clustering...")
        ids, vectors, titles = fetch_all_points(client)
        if len(ids) == 0:
            return

        labels = run_hdbscan(vectors)
        summary = build_full_summary(ids, vectors, titles, labels)

        save_summary(summary)
        write_labels_to_qdrant(client, ids, labels, summary)

        dur = time.time() - t0
        dur_min = dur / 60
        print(f"\n🎉 Initial topic clustering complete in {dur:.1f}s ({dur_min:.2f} min)")
        print(f"   points:   {summary.num_points}")
        print(f"   clusters: {summary.num_clusters}")
        print(f"   noise:    {summary.num_noise}")
        print("🌠 Done.\n")
        return

    # MODE 2: OUTLIER-ONLY RE-CLUSTERING
    print("🔁 clusters.json found → topic-clustering NOISE points only...")
    noise_ids, noise_vectors, noise_titles = fetch_noise_points(client)
    if len(noise_ids) == 0:
        print("✅ No unassigned/noise points. Nothing to do.")
        return

    noise_labels_local = run_hdbscan(noise_vectors)
    new_summary, noise_labels_global = extend_summary_with_noise_clusters(
        existing_summary,
        noise_ids,
        noise_vectors,
        noise_titles,
        noise_labels_local,
    )

    write_labels_to_qdrant(client, noise_ids, noise_labels_global, new_summary)
    save_summary(new_summary)

    dur = time.time() - t0
    dur_min = dur / 60
    print(f"\n🎉 Noise topic-clustering complete in {dur:.1f}s ({dur_min:.2f} min)")
    print(f"   points:   {new_summary.num_points}")
    print(f"   clusters: {new_summary.num_clusters}")
    print(f"   noise:    {new_summary.num_noise}")
    print("🌠 Done.\n")


if __name__ == "__main__":
    main()
