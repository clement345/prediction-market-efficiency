"""
Bayesian Coherence Index computation — Section 3.4 of the paper

For each cluster of related markets, compute:
  BCI(C) = KL(P_market || P_coherent)
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from scipy.special import rel_entr  # element-wise KL: p log(p/q)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) in bits."""
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return float(np.sum(rel_entr(p, q)) / np.log(2))


def coherent_projection(prices: np.ndarray) -> np.ndarray:
    """
    For a partition cluster: prices should sum to 1.
    P_coherent = prices / sum(prices)  (re-normalization = closest coherent dist)
    """
    total = prices.sum()
    if abs(total - 1.0) < 1e-9:
        return prices.copy()
    return prices / total


def compute_bci_partition(prices: list[float]) -> float:
    """
    BCI for a partition cluster.

    prices: list of mid-prices for each outcome in the partition
    Returns: BCI in bits
    """
    p = np.array(prices)
    p_coherent = coherent_projection(p)
    return kl_divergence(p, p_coherent)


def compute_bci_for_all_clusters(
    clusters: dict[str, list[str]],
    market_prices: dict[str, float],
) -> dict[str, float]:
    """
    clusters: {cluster_id -> [market_id, ...]}
    market_prices: {market_id -> mid_price}
    Returns: {cluster_id -> BCI}
    """
    results = {}
    for cid, market_ids in clusters.items():
        prices = [market_prices.get(mid) for mid in market_ids]
        prices = [p for p in prices if p is not None]
        if len(prices) < 2:
            continue
        bci = compute_bci_partition(prices)
        results[cid] = bci
    return results


def aggregate_bci(
    bci_by_cluster: dict[str, float],
    volumes: dict[str, float],
) -> float:
    """Volume-weighted average BCI."""
    total_vol = 0.0
    weighted_sum = 0.0
    for cid, bci in bci_by_cluster.items():
        vol = volumes.get(cid, 1.0)
        weighted_sum += bci * vol
        total_vol += vol
    return weighted_sum / total_vol if total_vol > 0 else 0.0


if __name__ == "__main__":
    # Example usage
    example_prices = {
        # US Election partition: Biden + Trump + Other should sum to 1
        "biden_wins": 0.45,
        "trump_wins": 0.52,
        "other_wins": 0.08,   # sum = 1.05 → BCI > 0
    }

    prices = list(example_prices.values())
    bci = compute_bci_partition(prices)
    print(f"Example partition prices: {prices}")
    print(f"Sum: {sum(prices):.4f}")
    print(f"BCI: {bci:.6f} bits")

    # Real data
    if Path("clusters.json").exists():
        clusters = json.loads(Path("clusters.json").read_text())
        print(f"\nLoaded {len(clusters)} clusters")
        # TODO: load real prices from database and compute
