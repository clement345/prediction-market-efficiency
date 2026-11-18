"""
Longitudinal calibration analysis.
Generates:
  - figures/fig7_calibration_by_horizon.pdf
  - figures/fig8_brier_vs_horizon.pdf
  - data/longitudinal_stats.json
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi":     150,
})

DATA_DIR = Path(__file__).parent
FIG_DIR  = DATA_DIR.parent / "figures"
HORIZONS = [7, 14, 30, 60, 90, 180]


def brier(preds, outcomes):
    p, o = np.array(preds), np.array(outcomes)
    return float(np.mean((p - o) ** 2))


def log_score(preds, outcomes, eps=1e-9):
    p, o = np.array(preds), np.array(outcomes)
    return float(np.mean(o * np.log(p + eps) + (1-o) * np.log(1-p + eps)))


def calibration_curve(preds, outcomes, n_bins=10):
    p, o = np.array(preds), np.array(outcomes)
    bins = np.linspace(0, 1, n_bins + 1)
    centers, freqs, counts = [], [], []
    for i in range(len(bins)-1):
        mask = (p >= bins[i]) & (p < bins[i+1])
        if mask.sum() >= 3:
            centers.append(float((bins[i]+bins[i+1])/2))
            freqs.append(float(o[mask].mean()))
            counts.append(int(mask.sum()))
    return centers, freqs, counts


def main():
    records = json.loads((DATA_DIR / "longitudinal_calibration.json").read_text())
    print(f"Loaded {len(records)} resolved markets")

    # ── Compute stats at each horizon ────────────────────────────────────────
    stats_by_horizon = {}
    for days in HORIZONS:
        k = str(days)
        pairs = [(r["predictions"][k], r["outcome"])
                 for r in records if k in r["predictions"]]
        if len(pairs) < 10:
            continue
        preds   = [x[0] for x in pairs]
        outcomes = [x[1] for x in pairs]
        bs = brier(preds, outcomes)
        ls = log_score(preds, outcomes)
        bs_base = 0.25
        skill = 1 - bs / bs_base
        centers, freqs, counts = calibration_curve(preds, outcomes, n_bins=10)

        # Mean absolute error
        mae = float(np.mean(np.abs(np.array(preds) - np.array(outcomes))))

        # Favourite-longshot bias: 5-15%
        flb_mask = [(p, o) for p, o in zip(preds, outcomes) if 0.05 <= p <= 0.15]
        flb_pred = np.mean([x[0] for x in flb_mask]) if flb_mask else None
        flb_act  = np.mean([x[1] for x in flb_mask]) if flb_mask else None

        stats_by_horizon[days] = {
            "n":          len(pairs),
            "brier":      round(bs, 4),
            "log_score":  round(ls, 4),
            "skill":      round(skill, 4),
            "mae":        round(mae, 4),
            "flb_pred":   round(flb_pred, 3) if flb_pred else None,
            "flb_actual": round(flb_act, 3)  if flb_act  else None,
            "centers":    centers,
            "freqs":      freqs,
            "counts":     counts,
        }
        print(f"  T-{days:3d}d  N={len(pairs):3d}  Brier={bs:.4f}  skill={skill:.3f}  MAE={mae:.3f}")

    # ── Figure 7: calibration curves at key horizons ─────────────────────────
    print("\nGenerating Figure 7: calibration curves by horizon...")
    horizons_to_plot = [h for h in [30, 90, 180] if h in stats_by_horizon]
    n_cols = len(horizons_to_plot)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 5.5))
    if n_cols == 1: axes = [axes]

    colors = ["#2563EB", "#DC2626", "#16A34A", "#7C3AED"]

    for ax, days, color in zip(axes, horizons_to_plot, colors):
        s = stats_by_horizon[days]
        centers = np.array(s["centers"])
        freqs   = np.array(s["freqs"])
        counts  = np.array(s["counts"])

        ax.plot([0,1],[0,1], "k--", lw=1.2, alpha=0.5, label="Perfect calibration")
        ax.plot(centers, freqs, color=color, marker="o", lw=2, ms=5,
                label=f"Polymarket T-{days}d\nBrier={s['brier']:.4f}")

        # Confidence bands
        for c, f, n in zip(centers, freqs, counts):
            se = np.sqrt(f*(1-f)/max(n,1))
            ax.fill_between([c-0.04, c+0.04], [f-1.96*se]*2, [f+1.96*se]*2,
                            alpha=0.2, color=color)

        # Superforecaster benchmark line
        ax.axhline(0.5, color="gray", lw=0.5, alpha=0.3)

        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Predicted probability", fontsize=11)
        ax.set_ylabel("Empirical frequency", fontsize=11)
        ax.set_title(f"T$-${days} days before resolution\n(N={s['n']})", fontsize=12)
        ax.legend(fontsize=9, loc="upper left")

        txt = (f"Brier:  {s['brier']:.4f}\n"
               f"Skill:  {s['skill']:.3f}\n"
               f"MAE:    {s['mae']:.3f}\n"
               f"N = {s['n']}")
        ax.text(0.97, 0.03, txt, transform=ax.transAxes,
                va="bottom", ha="right", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    fig.suptitle("Figure 7 — Polymarket calibration at multiple prediction horizons\n"
                 "(retrospective longitudinal analysis, N=438 resolved markets)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = FIG_DIR / "fig7_calibration_by_horizon.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(str(p).replace(".pdf",".png"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")

    # ── Figure 8: Brier score vs horizon (decay curve) ────────────────────────
    print("Generating Figure 8: Brier vs horizon...")
    h_vals  = sorted(stats_by_horizon.keys())
    bs_vals = [stats_by_horizon[h]["brier"] for h in h_vals]
    n_vals  = [stats_by_horizon[h]["n"]     for h in h_vals]
    sk_vals = [stats_by_horizon[h]["skill"] for h in h_vals]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Brier vs horizon
    ax = axes[0]
    ax.plot(h_vals, bs_vals, "o-", color="#2563EB", lw=2.5, ms=8, label="Polymarket (retrospective)")
    # Benchmark lines (from literature)
    ax.axhline(0.15, color="#DC2626", lw=1.5, linestyle="--", label="Superforecasters ~0.15 (Tetlock 2016)")
    ax.axhline(0.25, color="gray",    lw=1.0, linestyle=":",  label="Uninformative baseline (0.25)")
    ax.axhline(0.087,color="#16A34A", lw=1.5, linestyle="-.", label="Manifold Markets 0.087 (this paper)")
    ax.fill_between(h_vals, [b - 0.02 for b in bs_vals], [b + 0.02 for b in bs_vals],
                    alpha=0.15, color="#2563EB")

    for h, b, n in zip(h_vals, bs_vals, n_vals):
        ax.annotate(f"N={n}", (h, b), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)

    ax.set_xlabel("Days before resolution (prediction horizon)", fontsize=12)
    ax.set_ylabel("Brier score (lower = better)", fontsize=12)
    ax.set_title("Brier score vs. prediction horizon", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(-5, 195); ax.set_ylim(0, 0.28)

    # Right: Skill score vs horizon
    ax2 = axes[1]
    ax2.plot(h_vals, sk_vals, "s-", color="#7C3AED", lw=2.5, ms=8, label="Polymarket skill score")
    ax2.axhline(0, color="black", lw=1.0, linestyle="-")
    ax2.axhline(0.30, color="#DC2626", lw=1.5, linestyle="--", label="Superforecasters ~0.30")
    ax2.set_xlabel("Days before resolution", fontsize=12)
    ax2.set_ylabel("Brier skill score (higher = better)", fontsize=12)
    ax2.set_title("Skill score vs. prediction horizon", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.set_xlim(-5, 195)

    fig.suptitle("Figure 8 — Brier and skill scores by prediction horizon\n"
                 "Polymarket retrospective longitudinal study (N=438 resolved markets)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = FIG_DIR / "fig8_brier_vs_horizon.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(str(p).replace(".pdf",".png"), bbox_inches="tight")
    plt.close()
    print(f"  Saved {p}")

    # ── Save stats ────────────────────────────────────────────────────────────
    out = {str(k): {kk: vv for kk, vv in v.items()
                    if kk not in ("centers","freqs","counts")}
           for k, v in stats_by_horizon.items()}
    (DATA_DIR / "longitudinal_stats.json").write_text(json.dumps(out, indent=2))
    print(f"\nStats saved → data/longitudinal_stats.json")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'Horizon':>10} {'N':>6} {'Brier':>8} {'Skill':>8} {'MAE':>8}")
    print("="*65)
    for h in h_vals:
        s = stats_by_horizon[h]
        print(f"T-{h:3d}d    {s['n']:>6}  {s['brier']:>8.4f}  {s['skill']:>8.4f}  {s['mae']:>8.4f}")
    print("="*65)
    print(f"{'Superforecasters':>10}     --    ~0.1500   ~0.3000     --")
    print(f"{'Manifold Mkts':>10}    88    0.0865    0.6541     --")


if __name__ == "__main__":
    main()
