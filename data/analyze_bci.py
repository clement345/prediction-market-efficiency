"""
BCI Analysis — Main empirical results
Section 4 of the paper

Key metric: Partition Coherence Deviation (PCD)
  PCD = Σp_i - 1  for partition clusters (should be 0 if coherent)
  PCD > 0 → overcounting → Dutch book: sell all outcomes
  PCD < 0 → undercounting → Dutch book: buy all outcomes
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_FILE = Path(__file__).parent / "raw" / "markets_20260401_215624.json"

PARTITION_KEYWORDS = [
    "winner", "win the", "champion", "elected", "president",
    "win 2025", "win 2024", "world series", "super bowl",
    "gold medal", "first place", "tournament winner",
]


def load_data() -> list[dict]:
    print("Loading market data...")
    return json.loads(DATA_FILE.read_text())


def get_yes_price(m: dict) -> float | None:
    try:
        prices_raw = m.get("outcomePrices", "[]")
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
        outcomes_raw = m.get("outcomes", "[]")
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        if not prices or not outcomes:
            return None
        for o, p in zip(outcomes, prices):
            if str(o).lower() not in ("no", "false"):
                return float(p)
    except Exception:
        return None


def build_event_clusters(markets: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for m in markets:
        ev = m.get("events")
        if ev and isinstance(ev, list):
            slug = ev[0].get("slug") or ev[0].get("id")
            if slug:
                groups[slug].append(m)
    return {s: ms for s, ms in groups.items() if len(ms) >= 2}


def is_partition_cluster(slug: str, ms: list[dict]) -> bool:
    ev_title = ""
    if ms[0].get("events"):
        ev_title = ms[0]["events"][0].get("title", "").lower()
    return any(kw in ev_title or kw in slug.lower() for kw in PARTITION_KEYWORDS)


def analyze_cluster(slug: str, ms: list[dict]) -> dict | None:
    prices = [get_yes_price(m) for m in ms]
    prices = [p for p in prices if p is not None and 0.001 < p < 0.999]
    if len(prices) < 2:
        return None
    p = np.array(prices)
    total = float(p.sum())
    pcd = total - 1.0        # Partition Coherence Deviation
    vol = sum(float(m.get("volume") or 0) for m in ms)
    # Dollar value of Dutch book: |PCD| × synthetic portfolio value ≈ 1
    dutch_book_pct = abs(pcd)
    return {
        "slug": slug,
        "pcd": pcd,
        "sum_prices": total,
        "dutch_book_pct": dutch_book_pct,
        "n_markets": len(prices),
        "total_vol": vol,
        "is_partition": is_partition_cluster(slug, ms),
    }


def print_summary(label: str, results: list[dict]) -> None:
    pcd  = np.array([r["pcd"] for r in results])
    sums = np.array([r["sum_prices"] for r in results])
    vols = np.array([r["total_vol"] for r in results])
    db   = np.array([r["dutch_book_pct"] for r in results])

    vw_pcd = np.average(pcd, weights=np.maximum(vols, 1))
    vw_db  = np.average(db,  weights=np.maximum(vols, 1))

    print(f"\n{'='*60}")
    print(f" {label}  (N={len(results)})")
    print(f"{'='*60}")
    print(f"  Mean PCD (Σp - 1):        {np.mean(pcd):+.4f}")
    print(f"  Median PCD:               {np.median(pcd):+.4f}")
    print(f"  Vol-weighted PCD:         {vw_pcd:+.4f}")
    print(f"  Mean |Dutch book| %:      {np.mean(db)*100:.2f}%")
    print(f"  Vol-weighted |DB| %:      {vw_db*100:.2f}%")
    print(f"  Overcounted (PCD > 0.05): {np.mean(pcd > 0.05)*100:.1f}%")
    print(f"  Undercounted (PCD <-0.05):{np.mean(pcd <-0.05)*100:.1f}%")
    print(f"  Approx coherent (|PCD|<0.02): {np.mean(np.abs(pcd)<0.02)*100:.1f}%")


def main() -> None:
    markets = load_data()
    clusters = build_event_clusters(markets)
    print(f"Total multi-market event clusters: {len(clusters)}")

    results = []
    for slug, ms in clusters.items():
        r = analyze_cluster(slug, ms)
        if r:
            results.append(r)

    partition     = [r for r in results if r["is_partition"]]
    non_partition = [r for r in results if not r["is_partition"]]

    print_summary("PARTITION clusters  (single winner)", partition)
    print_summary("NON-PARTITION clusters (multi-winner)", non_partition)

    # Top Dutch book opportunities
    top = sorted(results, key=lambda x: -x["dutch_book_pct"])[:10]
    print(f"\n{'='*60}")
    print(" Top 10 Dutch Book Opportunities")
    print(f"{'='*60}")
    for r in top:
        tag = "PARTITION" if r["is_partition"] else "non-part."
        print(f"  [{tag}] {r['slug'][:45]:45s} PCD={r['pcd']:+.3f}  vol=${r['total_vol']:>10,.0f}")

    # Save results
    out_path = Path(__file__).parent / "bci_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} cluster results → {out_path}")


if __name__ == "__main__":
    main()
