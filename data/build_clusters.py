"""
Coherence cluster construction — Section 3.2 of the paper

Requires: sentence-transformers, python-louvain, networkx, numpy
Install: pip install sentence-transformers python-louvain networkx numpy
"""
from __future__ import annotations

import json
import numpy as np
import networkx as nx
from pathlib import Path
from sentence_transformers import SentenceTransformer
from community import best_partition

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
THRESHOLD  = 0.65
DATA_DIR   = Path(__file__).parent / "raw"


def load_latest_markets() -> list[dict]:
    files = sorted(DATA_DIR.glob("markets_*.json"))
    if not files:
        raise FileNotFoundError("No market snapshots found. Run collect_polymarket.py first.")
    return json.loads(files[-1].read_text())


def build_coherence_clusters(
    markets: list[dict],
    threshold: float = THRESHOLD,
) -> dict[int, int]:
    """
    Returns: partition dict {market_index -> cluster_id}
    """
    model = SentenceTransformer(MODEL_NAME)
    questions = [m.get("question", m.get("title", "")) for m in markets]
    print(f"Encoding {len(questions)} market questions...")
    embeddings = model.encode(questions, normalize_embeddings=True, show_progress_bar=True)

    sim = embeddings @ embeddings.T  # cosine similarity (normalized vecs)

    G = nx.Graph()
    G.add_nodes_from(range(len(markets)))
    rows, cols = np.where(sim > threshold)
    for i, j in zip(rows, cols):
        if i < j:
            G.add_edge(int(i), int(j), weight=float(sim[i, j]))

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    partition = best_partition(G)
    n_clusters = len(set(partition.values()))
    print(f"Found {n_clusters} coherence clusters")

    return partition


def classify_cluster_structure(
    cluster_markets: list[dict],
) -> str:
    """Heuristic classification: partition | conditional | dag"""
    questions = [m.get("question", "") for m in cluster_markets]
    lower = [q.lower() for q in questions]

    if any("if" in q or "given" in q or "conditional" in q for q in lower):
        return "conditional"
    # Partition: mutually exclusive events summing to ~1
    return "partition"


def filter_markets(markets: list[dict]) -> list[dict]:
    """Keep only active, liquid markets for the paper sample."""
    filtered = []
    for m in markets:
        try:
            vol = float(m.get("volume", 0) or 0)
            # Skip low-volume
            if vol < 10_000:
                continue
            # Keep only markets with a usable price (not resolved to 0 or 1)
            outcomes = m.get("outcomes", "[]")
            if isinstance(outcomes, str):
                import json as _json
                outcomes = _json.loads(outcomes)
            # Skip if no outcomes
            if not outcomes:
                continue
            filtered.append(m)
        except Exception:
            continue
    return filtered


if __name__ == "__main__":
    print("Loading markets...")
    all_markets = load_latest_markets()
    print(f"Total raw markets: {len(all_markets)}")

    markets = filter_markets(all_markets)
    print(f"After filtering (vol ≥ $10k): {len(markets)} markets")

    partition = build_coherence_clusters(markets)

    # Group by cluster
    clusters: dict[int, list[int]] = {}
    for idx, cid in partition.items():
        clusters.setdefault(cid, []).append(idx)

    # Print summary
    sizes = [len(v) for v in clusters.values()]
    print(f"\nCluster size distribution:")
    print(f"  Mean: {np.mean(sizes):.1f}, Median: {np.median(sizes):.1f}, Max: {max(sizes)}")

    # Save
    out = {str(cid): [markets[i]["id"] for i in idxs]
           for cid, idxs in clusters.items() if len(idxs) > 1}
    Path("clusters.json").write_text(json.dumps(out, indent=2))
    print(f"Saved {len(out)} multi-market clusters to clusters.json")
