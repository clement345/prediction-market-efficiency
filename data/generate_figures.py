"""
Generate all paper figures from the BCI/PCD analysis.
Outputs: figures/fig1_pcd_distribution.pdf
         figures/fig2_pcd_by_size.pdf
         figures/fig3_top_dutch_books.pdf
         figures/fig4_active_prices.pdf
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

matplotlib.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.labelsize":  12,
    "axes.titlesize":  13,
    "legend.fontsize": 10,
    "figure.dpi":      150,
    "text.usetex":     False,
})

DATA_FILE = Path(__file__).parent / "raw" / "markets_20260401_215624.json"
FIG_DIR   = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

PARTITION_KEYWORDS = [
    "winner","win the","champion","elected","president",
    "win 2025","win 2024","world series","super bowl",
    "gold medal","first place","tournament winner",
]

def get_yes_price(m):
    try:
        prices   = json.loads(m["outcomePrices"]) if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices",[])
        outcomes = json.loads(m["outcomes"])       if isinstance(m.get("outcomes"),str)       else m.get("outcomes",[])
        for o, p in zip(outcomes, prices):
            if str(o).lower() not in ("no","false"):
                return float(p)
    except Exception:
        return None

def is_partition(slug, ms):
    title = ms[0].get("events",[{}])[0].get("title","").lower() if ms[0].get("events") else ""
    return any(kw in title or kw in slug.lower() for kw in PARTITION_KEYWORDS)

def main():
    print("Loading data...")
    data = json.loads(DATA_FILE.read_text())

    # Build clusters
    groups = defaultdict(list)
    for m in data:
        ev = m.get("events")
        if ev and isinstance(ev, list):
            slug = ev[0].get("slug") or ev[0].get("id")
            if slug: groups[slug].append(m)

    results = []
    for slug, ms in groups.items():
        if len(ms) < 2: continue
        prices = [get_yes_price(m) for m in ms]
        prices = [p for p in prices if p is not None and 0.001 < p < 0.999]
        if len(prices) < 2: continue
        total = sum(prices)
        pcd   = total - 1.0
        vol   = sum(float(m.get("volume") or 0) for m in ms)
        results.append({
            "slug":      slug,
            "pcd":       pcd,
            "n":         len(prices),
            "vol":       vol,
            "partition": is_partition(slug, ms),
        })

    part    = [r for r in results if r["partition"]]
    nonpart = [r for r in results if not r["partition"]]

    # ── FIGURE 1: PCD Distribution ─────────────────────────────────────────
    print("Generating Figure 1: PCD distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, subset, label, color, xlim in [
        (axes[0], part,    "Partition clusters\n(single-winner markets)",   "#2563EB", (-2, 2)),
        (axes[1], nonpart, "Non-partition clusters\n(multi-winner markets)", "#DC2626", (-1, 10)),
    ]:
        pcd_vals = np.array([r["pcd"] for r in subset])
        pcd_clip = np.clip(pcd_vals, xlim[0], xlim[1])

        bins = np.linspace(xlim[0], xlim[1], 40)
        ax.hist(pcd_clip, bins=bins, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax.axvline(0,  color="black", linewidth=1.5, linestyle="-",  label="Coherent (PCD=0)")
        ax.axvline(np.median(pcd_vals), color=color, linewidth=2, linestyle="--",
                   label=f"Median = {np.median(pcd_vals):+.3f}")
        vw = np.average(pcd_vals, weights=np.maximum([r["vol"] for r in subset], 1))
        ax.axvline(vw, color="darkred", linewidth=2, linestyle=":",
                   label=f"Vol-wtd mean = {vw:+.3f}")

        ax.set_xlabel("Partition Coherence Deviation  (Σp − 1)", fontsize=12)
        ax.set_ylabel("Number of clusters", fontsize=12)
        ax.set_title(label, fontsize=13, pad=10)
        ax.legend(framealpha=0.9, fontsize=9)
        ax.set_xlim(xlim)

        # Annotation box
        n_over  = np.mean(pcd_vals >  0.05) * 100
        n_under = np.mean(pcd_vals < -0.05) * 100
        txt = f"N = {len(subset)}\nOvercounted: {n_over:.0f}%\nUndercounted: {n_under:.0f}%"
        ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
                fontsize=9)

    fig.suptitle("Figure 1 — Partition Coherence Deviation (PCD) across Polymarket clusters",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path1 = FIG_DIR / "fig1_pcd_distribution.pdf"
    fig.savefig(path1, bbox_inches="tight")
    fig.savefig(str(path1).replace(".pdf",".png"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {path1}")

    # ── FIGURE 2: PCD vs Cluster Size ──────────────────────────────────────
    print("Generating Figure 2: PCD vs cluster size...")
    fig, ax = plt.subplots(figsize=(9, 6))

    for subset, label, color, marker in [
        (part,    "Partition",     "#2563EB", "o"),
        (nonpart, "Non-partition", "#DC2626", "^"),
    ]:
        ns   = np.array([r["n"]   for r in subset])
        pcds = np.array([r["pcd"] for r in subset])
        vols = np.array([r["vol"] for r in subset])
        sizes = np.clip(np.sqrt(vols / 5_000), 20, 300)
        ax.scatter(ns, pcds, s=sizes, c=color, marker=marker, alpha=0.5,
                   label=label, edgecolors="white", linewidths=0.3)

    ax.axhline(0, color="black", linewidth=1.5, linestyle="-")
    ax.axhline(0.05,  color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(-0.05, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Number of related markets in cluster", fontsize=12)
    ax.set_ylabel("Partition Coherence Deviation  (Σp − 1)", fontsize=12)
    ax.set_title("Figure 2 — PCD vs. cluster size\n(bubble size ∝ total volume)",
                 fontsize=13, pad=10)
    ax.legend(fontsize=10)
    ax.set_xlim(1, max(r["n"] for r in results) + 2)
    plt.tight_layout()
    path2 = FIG_DIR / "fig2_pcd_vs_size.pdf"
    fig.savefig(path2, bbox_inches="tight")
    fig.savefig(str(path2).replace(".pdf",".png"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {path2}")

    # ── FIGURE 3: Top Dutch Book Opportunities ─────────────────────────────
    print("Generating Figure 3: Top Dutch book opportunities...")
    top15 = sorted(results, key=lambda x: -abs(x["pcd"]))[:15]
    labels  = [r["slug"][:35] + ("…" if len(r["slug"]) > 35 else "") for r in top15]
    pcd_v   = [r["pcd"]  for r in top15]
    colors  = ["#2563EB" if r["partition"] else "#DC2626" for r in top15]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(top15)), pcd_v, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel("Partition Coherence Deviation  (Σp − 1)", fontsize=12)
    ax.set_title("Figure 3 — Top 15 Coherence Violations\n(blue = partition, red = non-partition)",
                 fontsize=13, pad=10)
    ax.invert_yaxis()

    patch1 = mpatches.Patch(color="#2563EB", alpha=0.85, label="Partition cluster")
    patch2 = mpatches.Patch(color="#DC2626", alpha=0.85, label="Non-partition cluster")
    ax.legend(handles=[patch1, patch2], fontsize=10)

    # Volume annotations
    for i, r in enumerate(top15):
        vol_m = r["vol"] / 1e6
        ax.text(pcd_v[i] + 0.05 if pcd_v[i] >= 0 else pcd_v[i] - 0.05,
                i, f"${vol_m:.1f}M", va="center",
                ha="left" if pcd_v[i] >= 0 else "right", fontsize=7.5)
    plt.tight_layout()
    path3 = FIG_DIR / "fig3_top_dutch_books.pdf"
    fig.savefig(path3, bbox_inches="tight")
    fig.savefig(str(path3).replace(".pdf",".png"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {path3}")

    # ── FIGURE 4: Active market price distribution ─────────────────────────
    print("Generating Figure 4: Active market price distribution...")
    active = [m for m in data
              if m.get("active") and not m.get("closed")
              and float(m.get("volume") or 0) >= 10_000]
    prices_active = [get_yes_price(m) for m in active]
    prices_active = [p for p in prices_active if p is not None and 0 < p < 1]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(prices_active, bins=bins, color="#7C3AED", alpha=0.8,
            edgecolor="white", linewidth=0.4)
    ax.axvline(0.5, color="black", linewidth=1.5, linestyle="--",
               label="p = 0.5 (maximum uncertainty)")
    ax.set_xlabel("Market mid-price (YES probability)", fontsize=12)
    ax.set_ylabel("Number of active markets", fontsize=12)
    ax.set_title(f"Figure 4 — Price distribution of active Polymarket markets\n"
                 f"(N = {len(prices_active):,}, volume ≥ $10k)",
                 fontsize=13, pad=10)
    ax.legend(fontsize=10)

    # Annotate favorite-longshot range
    ax.axvspan(0.05, 0.15, alpha=0.12, color="red", label="FLB zone (5–15%)")
    ax.axvspan(0.85, 0.95, alpha=0.12, color="red")
    txt = (f"Mean:   {np.mean(prices_active):.3f}\n"
           f"Median: {np.median(prices_active):.3f}\n"
           f"N:      {len(prices_active):,}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
            fontsize=9)
    plt.tight_layout()
    path4 = FIG_DIR / "fig4_active_prices.pdf"
    fig.savefig(path4, bbox_inches="tight")
    fig.savefig(str(path4).replace(".pdf",".png"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {path4}")

    # ── Save summary stats ─────────────────────────────────────────────────
    def vw(lst, key):
        vals = np.array([r[key] for r in lst])
        wts  = np.maximum(np.array([r["vol"] for r in lst]), 1)
        return float(np.average(vals, weights=wts))

    pcd_part  = np.array([r["pcd"] for r in part])
    pcd_npart = np.array([r["pcd"] for r in nonpart])

    stats = {
        "n_markets_total":   len(data),
        "n_active":          len(active),
        "n_clusters":        len(results),
        "n_partition":       len(part),
        "n_nonpartition":    len(nonpart),
        "part_mean_pcd":     float(np.mean(pcd_part)),
        "part_median_pcd":   float(np.median(pcd_part)),
        "part_vw_pcd":       vw(part, "pcd"),
        "part_pct_over":     float(np.mean(pcd_part >  0.05) * 100),
        "part_pct_under":    float(np.mean(pcd_part < -0.05) * 100),
        "part_pct_coherent": float(np.mean(np.abs(pcd_part) < 0.02) * 100),
        "npart_mean_pcd":    float(np.mean(pcd_npart)),
        "npart_median_pcd":  float(np.median(pcd_npart)),
        "npart_vw_pcd":      vw(nonpart, "pcd"),
        "npart_pct_over":    float(np.mean(pcd_npart >  0.05) * 100),
        "npart_pct_coherent":float(np.mean(np.abs(pcd_npart) < 0.02) * 100),
        "active_price_mean": float(np.mean(prices_active)),
        "active_price_median":float(np.median(prices_active)),
        "active_price_n":    len(prices_active),
    }
    Path(__file__).parent.parent / "figures"
    (Path(__file__).parent / "stats_summary.json").write_text(json.dumps(stats, indent=2))
    print("\nSummary stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
