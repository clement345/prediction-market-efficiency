"""
Analysis: Calibration + Information Half-Life
Generates:
  - figures/fig5_calibration_comparison.pdf
  - figures/fig6_halflife_distribution.pdf
  - data/calibration_results.json
  - data/halflife_results.json
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

matplotlib.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize":10,
    "figure.dpi":     150,
})

DATA_DIR = Path(__file__).parent
FIG_DIR  = DATA_DIR.parent / "figures"

# ── 1. CALIBRATION ────────────────────────────────────────────────────────────

def brier_score(preds, outcomes):
    p = np.array(preds); o = np.array(outcomes)
    return float(np.mean((p - o)**2))

def calibration_curve(preds, outcomes, n_bins=10):
    p = np.array(preds); o = np.array(outcomes)
    bins = np.linspace(0, 1, n_bins + 1)
    centers, freqs, counts = [], [], []
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i+1])
        if mask.sum() >= 3:
            centers.append((bins[i] + bins[i+1]) / 2)
            freqs.append(float(o[mask].mean()))
            counts.append(int(mask.sum()))
    return centers, freqs, counts

def compute_calibration_stats(data, name):
    preds   = [d["prediction"] for d in data]
    outcomes = [d["resolution"] for d in data]
    bs  = brier_score(preds, outcomes)
    bs_baseline = brier_score([0.5]*len(preds), outcomes)
    skill = 1 - bs / bs_baseline
    centers, freqs, counts = calibration_curve(preds, outcomes, n_bins=10)
    return {
        "name":      name,
        "n":         len(data),
        "brier":     round(bs, 4),
        "skill":     round(skill, 4),
        "centers":   centers,
        "freqs":     freqs,
        "counts":    counts,
        "preds":     preds,
        "outcomes":  outcomes,
    }


def load_manifold():
    path = DATA_DIR / "manifold_resolved.json"
    if not path.exists(): return []
    return json.loads(path.read_text())


def load_polymarket_calibration():
    """
    Extract calibration pairs from CLOB history.
    For each market: use price 7 days before last point as prediction,
    last price as resolution proxy (0 or 1 if it converges).
    """
    index = json.loads((DATA_DIR / "clob_history_index.json").read_text())
    hist_dir = DATA_DIR / "raw" / "clob_history"
    pairs = []

    for mid, info in index.items():
        path = hist_dir / f"{mid}.json"
        if not path.exists(): continue
        history = json.loads(path.read_text())
        if len(history) < 20: continue

        prices = [h["p"] for h in history]
        last_p = prices[-1]

        # Only use markets that converged to near 0 or 1 (resolved)
        if last_p > 0.95:
            outcome = 1
        elif last_p < 0.05:
            outcome = 0
        else:
            continue  # not resolved yet

        # Prediction = price 7+ days before end (index -168 for hourly)
        pred_idx = max(0, len(prices) - 168)
        pred = prices[pred_idx]
        if not (0.02 < pred < 0.98):
            continue

        pairs.append({
            "prediction": pred,
            "resolution": outcome,
            "volume": info["volume"],
        })

    return pairs


# ── 2. INFORMATION HALF-LIFE ─────────────────────────────────────────────────

def exp_adjustment(t, theta, p0, lam):
    """Price path model: p(t) = p0 + (theta - p0)(1 - exp(-t/lam))"""
    return p0 + (theta - p0) * (1 - np.exp(-t / np.maximum(lam, 0.01)))


def estimate_halflife(prices: list[float], timestamps: list[int]) -> float | None:
    """
    Estimate half-life λ by fitting the exponential adjustment model
    to each price series, treating the start as p0 and end as θ.
    """
    if len(prices) < 15:
        return None
    p = np.array(prices, dtype=float)
    t_raw = np.array(timestamps, dtype=float)
    t = (t_raw - t_raw[0]) / 3600  # convert to hours

    p0    = float(p[0])
    theta = float(p[-1])

    if abs(theta - p0) < 0.05:   # no meaningful movement
        return None

    try:
        # Fit only to the adjustment window (first 72 hours)
        mask = t <= 72
        if mask.sum() < 10:
            return None
        popt, _ = curve_fit(
            lambda t_, lam: exp_adjustment(t_, theta, p0, lam),
            t[mask], p[mask],
            p0=[12.0],
            bounds=(0.1, 500),
            maxfev=5000,
        )
        lam = float(popt[0])
        if 0.1 < lam < 200:
            return lam
    except Exception:
        pass
    return None


def load_halflife_data():
    index   = json.loads((DATA_DIR / "clob_history_index.json").read_text())
    hist_dir = DATA_DIR / "raw" / "clob_history"
    results = []

    for mid, info in index.items():
        path = hist_dir / f"{mid}.json"
        if not path.exists(): continue
        history = json.loads(path.read_text())
        if len(history) < 15: continue

        prices = [h["p"] for h in history]
        times  = [h["t"] for h in history]
        lam    = estimate_halflife(prices, times)
        if lam is None: continue

        results.append({
            "market_id": mid,
            "question":  info["question"][:80],
            "volume":    info["volume"],
            "n_points":  len(prices),
            "halflife_h": round(lam, 2),
            "price_range": round(abs(prices[-1] - prices[0]), 3),
        })

    return results


# ── 3. GENERATE FIGURES ───────────────────────────────────────────────────────

def fig5_calibration(poly_data, manifold_data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, data, name, color, marker in [
        (axes[0], poly_data,     "Polymarket (CLOB history)",  "#2563EB", "o"),
        (axes[1], manifold_data, "Manifold Markets",           "#16A34A", "s"),
    ]:
        if not data["centers"]:
            ax.text(0.5, 0.5, f"Insufficient data\n(N={data['n']})",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        centers = np.array(data["centers"])
        freqs   = np.array(data["freqs"])
        counts  = np.array(data["counts"])

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration", alpha=0.6)

        # Calibration curve
        ax.plot(centers, freqs, color=color, marker=marker, linewidth=2,
                markersize=6, label=f"{name}\nBrier = {data['brier']:.3f}")

        # Confidence bands (Wilson)
        for c, f, n in zip(centers, freqs, counts):
            se = np.sqrt(f * (1-f) / max(n, 1))
            ax.fill_between([c - 0.04, c + 0.04], [f - 1.96*se]*2, [f + 1.96*se]*2,
                            alpha=0.15, color=color)

        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Predicted probability", fontsize=12)
        ax.set_ylabel("Empirical frequency", fontsize=12)
        ax.set_title(f"Calibration — {name}", fontsize=12, pad=8)
        ax.legend(fontsize=9, loc="upper left")

        bs_txt = (f"Brier score: {data['brier']:.4f}\n"
                  f"Skill score: {data['skill']:.4f}\n"
                  f"N = {data['n']:,}")
        ax.text(0.97, 0.03, bs_txt, transform=ax.transAxes,
                va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
                fontsize=9)

    fig.suptitle("Figure 5 — Calibration curves: Polymarket vs.\ Manifold Markets",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    p = FIG_DIR / "fig5_calibration_comparison.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(str(p).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")


def fig6_halflife(hl_results):
    lams = np.array([r["halflife_h"] for r in hl_results])
    vols = np.array([r["volume"] for r in hl_results])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: histogram
    ax = axes[0]
    bins = np.logspace(np.log10(max(lams.min(), 0.1)), np.log10(lams.max()), 30)
    ax.hist(lams, bins=bins, color="#7C3AED", alpha=0.8, edgecolor="white")
    ax.axvline(float(np.median(lams)), color="darkred", linewidth=2, linestyle="--",
               label=f"Median = {np.median(lams):.1f}h")
    vw_lam = np.average(lams, weights=np.maximum(vols, 1))
    ax.axvline(vw_lam, color="orange", linewidth=2, linestyle=":",
               label=f"Vol-weighted = {vw_lam:.1f}h")
    ax.set_xscale("log")
    ax.set_xlabel("Information half-life λ (hours, log scale)", fontsize=12)
    ax.set_ylabel("Number of markets", fontsize=12)
    ax.set_title("Distribution of information half-life", fontsize=12)
    ax.legend(fontsize=10)
    ax.text(0.97, 0.97,
            f"N = {len(lams)}\nMean = {lams.mean():.1f}h\nMedian = {np.median(lams):.1f}h",
            transform=ax.transAxes, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
            fontsize=9)

    # Right: λ vs log(volume)
    ax2 = axes[1]
    log_vols = np.log10(np.maximum(vols, 1))
    ax2.scatter(log_vols, lams, alpha=0.5, s=30, color="#7C3AED", edgecolors="white", linewidths=0.3)
    # Regression line
    corr, pval = pearsonr(log_vols, np.log10(lams))
    z = np.polyfit(log_vols, np.log10(lams), 1)
    xr = np.linspace(log_vols.min(), log_vols.max(), 100)
    ax2.plot(xr, 10**np.polyval(z, xr), color="darkred", linewidth=2,
             label=f"r = {corr:.2f} (p={pval:.3f})")
    ax2.set_xlabel("log₁₀(Volume, USD)", fontsize=12)
    ax2.set_ylabel("Information half-life λ (hours)", fontsize=12)
    ax2.set_title("Half-life vs. market volume", fontsize=12)
    ax2.set_yscale("log")
    ax2.legend(fontsize=10)

    fig.suptitle("Figure 6 — Information half-life λ across Polymarket contracts",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    p = FIG_DIR / "fig6_halflife.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(str(p).replace(".pdf", ".png"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")

    return {"median_h": float(np.median(lams)),
            "mean_h":   float(lams.mean()),
            "vw_mean_h": float(vw_lam),
            "n":         len(lams),
            "corr_vol":  float(corr),
            "pval_vol":  float(pval)}


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("[1] Calibration analysis...")
    manifold_raw  = load_manifold()
    polymarket_raw = load_polymarket_calibration()

    print(f"  Manifold: {len(manifold_raw)} resolved markets")
    print(f"  Polymarket (resolved via CLOB): {len(polymarket_raw)} markets")

    manifold_stats  = compute_calibration_stats(manifold_raw,  "Manifold")
    polymarket_stats = compute_calibration_stats(polymarket_raw, "Polymarket")

    print(f"  Manifold  Brier={manifold_stats['brier']:.4f} skill={manifold_stats['skill']:.4f}")
    print(f"  Polymarket Brier={polymarket_stats['brier']:.4f} skill={polymarket_stats['skill']:.4f}")

    print("\n[2] Half-life analysis...")
    hl_results = load_halflife_data()
    print(f"  Markets with estimated λ: {len(hl_results)}")

    print("\n[3] Generating figures...")
    fig5_calibration(polymarket_stats, manifold_stats)
    hl_stats = fig6_halflife(hl_results)

    print(f"\n  Median λ = {hl_stats['median_h']:.1f}h")
    print(f"  Vol-weighted λ = {hl_stats['vw_mean_h']:.1f}h")
    print(f"  Corr(log_vol, log_λ) = {hl_stats['corr_vol']:.3f} (p={hl_stats['pval_vol']:.3f})")

    # Save all results
    cal_results = {
        "polymarket": {k: v for k, v in polymarket_stats.items()
                       if k not in ("preds","outcomes","centers","freqs","counts")},
        "manifold":   {k: v for k, v in manifold_stats.items()
                       if k not in ("preds","outcomes","centers","freqs","counts")},
    }
    (DATA_DIR / "calibration_results.json").write_text(json.dumps(cal_results, indent=2))
    (DATA_DIR / "halflife_results.json").write_text(
        json.dumps({"stats": hl_stats, "markets": hl_results}, indent=2))

    print("\nAll done. Results saved.")
    return cal_results, hl_stats


if __name__ == "__main__":
    main()
