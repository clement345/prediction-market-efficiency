"""
Full empirical analysis for the paper.
Generates:
  - calibration stats + calibration curve data
  - PCD distribution data
  - cluster-level summary table
  - all numbers cited in the paper
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_FILE = Path(__file__).parent / "raw" / "markets_20260401_215624.json"
OUT_DIR   = Path(__file__).parent.parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

PARTITION_KEYWORDS = [
    "winner", "win the", "champion", "elected", "president",
    "win 2025", "win 2024", "world series", "super bowl",
    "gold medal", "first place", "tournament winner",
]


def get_yes_price(m: dict) -> float | None:
    try:
        prices  = json.loads(m["outcomePrices"]) if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices", [])
        outcomes = json.loads(m["outcomes"])      if isinstance(m.get("outcomes"), str)      else m.get("outcomes", [])
        for o, p in zip(outcomes, prices):
            if str(o).lower() not in ("no", "false"):
                return float(p)
    except Exception:
        return None


def is_resolved(m: dict) -> tuple[bool, int | None]:
    """Returns (is_resolved, outcome_index 0=NO 1=YES)"""
    try:
        prices = json.loads(m["outcomePrices"]) if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices", [])
        closed = m.get("closed", False)
        if not closed:
            return False, None
        prices_f = [float(p) for p in prices]
        if prices_f[0] >= 0.99:
            return True, 0   # NO resolved
        if prices_f[0] <= 0.01:
            return True, 1   # YES resolved
        # check last trade
        ltp = m.get("lastTradePrice")
        if ltp is not None:
            ltp = float(ltp)
            if ltp >= 0.99: return True, 1
            if ltp <= 0.01: return True, 0
    except Exception:
        pass
    return False, None


def main():
    print("Loading data...")
    data = json.loads(DATA_FILE.read_text())
    print(f"  Total markets: {len(data):,}")

    # ── 1. CALIBRATION ANALYSIS ────────────────────────────────────────────
    print("\n[1] Calibration analysis...")
    cal_pairs = []   # (predicted_prob, outcome)

    for m in data:
        resolved, outcome = is_resolved(m)
        if not resolved or outcome is None:
            continue
        p = get_yes_price(m)
        if p is None or not (0.01 < p < 0.99):
            continue
        cal_pairs.append((p, outcome))

    print(f"  Resolved markets for calibration: {len(cal_pairs):,}")

    probs   = np.array([x[0] for x in cal_pairs])
    outcomes = np.array([x[1] for x in cal_pairs])

    # Brier score
    brier = float(np.mean((probs - outcomes) ** 2))
    # Log score
    eps = 1e-9
    log_score = float(np.mean(outcomes * np.log(probs + eps) + (1 - outcomes) * np.log(1 - probs + eps)))
    # Baseline (predict 0.5 always)
    brier_baseline = float(np.mean((0.5 - outcomes) ** 2))

    print(f"  Brier score:    {brier:.4f}  (baseline: {brier_baseline:.4f})")
    print(f"  Log score:      {log_score:.4f}")
    print(f"  Skill score:    {1 - brier/brier_baseline:.4f}")

    # Calibration curve: 20 bins
    bins = np.linspace(0, 1, 21)
    bin_centers, bin_freqs, bin_counts = [], [], []
    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum() < 5:
            continue
        bin_centers.append(float((bins[i] + bins[i+1]) / 2))
        bin_freqs.append(float(outcomes[mask].mean()))
        bin_counts.append(int(mask.sum()))

    cal_data = {
        "bin_centers": bin_centers,
        "bin_freqs": bin_freqs,
        "bin_counts": bin_counts,
        "brier": brier,
        "log_score": log_score,
        "skill_score": 1 - brier / brier_baseline,
        "n_resolved": len(cal_pairs),
    }

    # Favorite-longshot bias: check 5-15% bin
    flb_mask = (probs >= 0.05) & (probs <= 0.15)
    if flb_mask.sum() > 0:
        flb_expected = float(probs[flb_mask].mean())
        flb_actual   = float(outcomes[flb_mask].mean())
        cal_data["flb_expected"] = flb_expected
        cal_data["flb_actual"]   = flb_actual
        print(f"  FLB (5-15%): predicted {flb_expected:.3f}, actual {flb_actual:.3f}")

    # ── 2. COHERENCE / PCD ANALYSIS ───────────────────────────────────────
    print("\n[2] Coherence / PCD analysis...")

    event_groups: dict[str, list[dict]] = defaultdict(list)
    for m in data:
        ev = m.get("events")
        if ev and isinstance(ev, list):
            slug = ev[0].get("slug") or ev[0].get("id")
            if slug:
                event_groups[slug].append(m)

    clusters = {s: ms for s, ms in event_groups.items() if len(ms) >= 2}

    def is_partition(slug, ms):
        title = ""
        if ms[0].get("events"):
            title = ms[0]["events"][0].get("title", "").lower()
        return any(kw in title or kw in slug.lower() for kw in PARTITION_KEYWORDS)

    all_results = []
    for slug, ms in clusters.items():
        prices = [get_yes_price(m) for m in ms]
        prices = [p for p in prices if p is not None and 0.001 < p < 0.999]
        if len(prices) < 2:
            continue
        p = np.array(prices)
        total = float(p.sum())
        pcd   = total - 1.0
        vol   = sum(float(m.get("volume") or 0) for m in ms)
        all_results.append({
            "slug":      slug,
            "pcd":       pcd,
            "sum":       total,
            "n":         len(prices),
            "vol":       vol,
            "partition": is_partition(slug, ms),
        })

    part     = [r for r in all_results if r["partition"]]
    nonpart  = [r for r in all_results if not r["partition"]]

    def vw_stat(lst, key, weight_key="vol"):
        vals = np.array([r[key] for r in lst])
        wts  = np.maximum(np.array([r[weight_key] for r in lst]), 1)
        return float(np.average(vals, weights=wts))

    pcd_all   = np.array([r["pcd"] for r in all_results])
    pcd_part  = np.array([r["pcd"] for r in part])
    pcd_npart = np.array([r["pcd"] for r in nonpart])

    coh_data = {
        "n_clusters_total":     len(all_results),
        "n_partition":          len(part),
        "n_nonpartition":       len(nonpart),
        # Partition
        "part_mean_pcd":        float(np.mean(pcd_part)),
        "part_median_pcd":      float(np.median(pcd_part)),
        "part_vw_pcd":          vw_stat(part, "pcd"),
        "part_mean_abs_db":     float(np.mean(np.abs(pcd_part))),
        "part_vw_abs_db":       abs(vw_stat(part, "pcd")),
        "part_pct_overcounted": float(np.mean(pcd_part > 0.05)),
        "part_pct_undercounted":float(np.mean(pcd_part < -0.05)),
        "part_pct_coherent":    float(np.mean(np.abs(pcd_part) < 0.02)),
        # Non-partition
        "npart_mean_pcd":       float(np.mean(pcd_npart)),
        "npart_median_pcd":     float(np.median(pcd_npart)),
        "npart_vw_pcd":         vw_stat(nonpart, "pcd"),
        "npart_mean_abs_db":    float(np.mean(np.abs(pcd_npart))),
        "npart_vw_abs_db":      abs(vw_stat(nonpart, "pcd")),
        "npart_pct_overcounted":float(np.mean(pcd_npart > 0.05)),
        "npart_pct_coherent":   float(np.mean(np.abs(pcd_npart) < 0.02)),
        # Top Dutch book
        "top_dutch_book": sorted(all_results, key=lambda x: -abs(x["pcd"]))[:10],
        # PCD distribution (for histogram)
        "pcd_all_hist": {
            "values": pcd_all.tolist(),
            "partition": pcd_part.tolist(),
            "nonpartition": pcd_npart.tolist(),
        },
    }

    print(f"  Total clusters:      {len(all_results):,}")
    print(f"  Partition clusters:  {len(part):,}")
    print(f"  Non-part. clusters:  {len(nonpart):,}")
    print(f"  Part. vol-wtd PCD:   {coh_data['part_vw_pcd']:+.4f}")
    print(f"  Non-part VW PCD:     {coh_data['npart_vw_pcd']:+.4f}")

    # ── 3. SAVE ALL RESULTS ───────────────────────────────────────────────
    results = {"calibration": cal_data, "coherence": coh_data}
    out = Path(__file__).parent / "paper_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nAll results saved → {out}")
    return results


if __name__ == "__main__":
    main()
