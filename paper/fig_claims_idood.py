#!/usr/bin/env python3
"""Revision figures, every value read from saved results (no typed numbers).
fig_idood: in-domain (audited test) vs nine-set mean ROC-AUC, one point per model, three-seed means
           (saved_predictions.Data for the models of Table 6, the tuned/LoRA runs, review/e1_clipft.json for CLIP-FT);
           arrows: shared recipe -> validation-tuned learning rate.
fig_claims: primary claims (review/stats.json), H1/H3 with the tuned VideoMAE-B (review/c45.json) and K1/K3 on the
           second XD-Violence sample (review/c6_confirmB.json): difference with 95% seed-aware interval.
fig_fusion_rules: fusion rules on the five untouched samples (review/h_partH.json)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[1]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
from make_figures import OUT, INK, INK2, GRID, COL1, style
import saved_predictions as RD
from saved_predictions import Data, SETS, UNSEEN, L
from sklearn.metrics import roc_auc_score
BLUE, ORANGE, PURPLE, GREY = "#2a78d6", "#eb6834", "#6a3d9a", "#8c8a84"
plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "pdf.fonttype": 42})
R = L / "review"; J = lambda f: json.load(open(R / f))


def fig_idood():
    RD.MODELS.update({"VideoMAE-B FT (tuned)": "A1_videomae_lr2e-05", "R(2+1)D-18 FT (tuned)": "A1_r2plus1d_18_lr2e-05",
                      "VideoMAE-B LoRA": "A1_videomae_lora_lr0.0001"})
    orig = RD._run_dir; RD._run_dir = lambda st, sd: (L / f"runs_review/{st}_seed{sd}") if "_lora" in st else orig(st, sd)
    D = Data()
    def pt(m):
        a = {s: np.mean([100 * roc_auc_score(D.sets[s]["y"][D.sets[s]["keep"]], p[D.sets[s]["keep"]]) for p in D.probs(m, s)]) for s in SETS}
        return a["Test"], np.mean([a[s] for s in UNSEEN])
    groups = {  # label: (model key, colour, marker)
        "Frozen features": [("LENS-Net-A", "LENS-Net-A"), ("CLIP probe", "CLIP probe"), ("LENS-Net", "LENS-Net"), ("w/o motion", "LENS-Net without motion"),
                            ("Motion probe", "Motion probe"), ("CLIP zero-shot", "CLIP zero-shot"), ("VadCLIP", "VadCLIP")],
        "Fine-tuned, shared recipe": [("VideoMAE-B FT", "VideoMAE-B"), ("Video Swin-T FT", "Swin-T"), ("R(2+1)D-18 FT", "R(2+1)D-18"),
                                      ("X-CLIP FT", "X-CLIP"), ("CNN-BiGRU", "CNN-BiGRU")],
        "Tuned or adapted": [("VideoMAE-B FT (tuned)", "VideoMAE-B (tuned lr)"), ("R(2+1)D-18 FT (tuned)", "R(2+1)D-18 (tuned lr)"),
                             ("VideoMAE-B LoRA", "VideoMAE-B LoRA"), ("CLIP-FT", "CLIP-FT")]}
    sty = {"Frozen features": (BLUE, "o"), "Fine-tuned, shared recipe": (ORANGE, "s"), "Tuned or adapted": (PURPLE, "D")}
    e1 = J("e1_clipft.json")["auc"]["CLIP-FT"]
    P = {}
    for g, ms in groups.items():
        for k, lab in ms: P[k] = (e1["Test"], e1["mean9"]) if k == "CLIP-FT" else pt(k)
    json.dump({k: [float(v[0]), float(v[1])] for k, v in P.items()}, open(OUT / "fig_idood_data.json", "w"), indent=1)
    fig, ax = plt.subplots(figsize=(COL1 * 1.1, 3.0))
    for a, b in (("VideoMAE-B FT", "VideoMAE-B FT (tuned)"), ("R(2+1)D-18 FT", "R(2+1)D-18 FT (tuned)")):
        ax.annotate("", xy=P[b], xytext=P[a], arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.6, shrinkA=5, shrinkB=4, mutation_scale=5.5))
    for g, ms in groups.items():
        c, mk = sty[g]
        xs = [P[k][0] for k, _ in ms]; ys = [P[k][1] for k, _ in ms]
        ax.scatter(xs, ys, s=15, c=c, marker=mk, edgecolors="white", linewidths=0.7, zorder=3 if g != "Frozen features" else 3.5, label=g)
    lab_at = {  # label position in data units (x, y, ha); a thin leader joins label and point
        "LENS-Net-A": (98.42, 88.2, "left"), "CLIP probe": (96.80, 87.3, "right"), "CLIP-FT": (96.80, 85.6, "right"),
        "VideoMAE-B LoRA": (96.80, 84.1, "right"), "LENS-Net": (98.42, 85.2, "left"), "VadCLIP": (98.42, 83.5, "left"),
        "VideoMAE-B FT (tuned)": (98.42, 81.9, "left"), "R(2+1)D-18 FT (tuned)": (96.50, 82.0, "right"),
        "CLIP zero-shot": (94.70, 82.6, "left"), "Motion probe": (98.42, 80.3, "left"), "w/o motion": (98.42, 79.0, "left"),
        "VideoMAE-B FT": (98.42, 77.4, "left"), "Video Swin-T FT": (96.80, 77.0, "right"), "R(2+1)D-18 FT": (95.95, 75.5, "right"),
        "X-CLIP FT": (95.30, 73.6, "left"), "CNN-BiGRU": (96.30, 66.6, "left")}
    name = {k: lab for ms in groups.values() for k, lab in ms}
    for k, (tx, ty, ha) in lab_at.items():
        ax.annotate(name[k], P[k], xytext=(tx, ty), textcoords="data", fontsize=5.4, color=INK, ha=ha, va="center", zorder=4,
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.35, shrinkA=1, shrinkB=2.5))
    ax.set_xlim(94.3, 99.75); ax.set_ylim(63.5, 89.5)
    style(ax, "both"); ax.set_xlabel("In-domain ROC-AUC, audited test (%)"); ax.set_ylabel("Mean ROC-AUC, nine unseen sets (%)")
    ax.legend(loc="lower right", frameon=False, fontsize=5.6, handletextpad=0.3, borderaxespad=0.2, markerscale=0.9)
    fig.tight_layout(pad=0.3); fig.savefig(OUT / "fig_idood.pdf"); plt.close(fig)


def fig_claims():
    st = J("stats.json")["primary"]; c45 = J("c45.json")["C4"]["claims"]; c6 = J("c6_confirmB.json")["claims"]
    order = [("H1", "H1 CLIP probe - VideoMAE-B FT (nine-set mean)", "Probe − VideoMAE-B, nine-set mean"),
             ("H2", "H2 CLIP probe - LENS-Net (nine-set mean)", "Probe − LENS-Net, nine-set mean"),
             ("H3", "H3 LENS-Net - VideoMAE-B FT (nine-set mean)", "LENS-Net − VideoMAE-B, nine-set mean"),
             ("H4", "H4 LENS-Net - w/o motion on SCFD (better)", "Motion benefit, SCFD"),
             ("H5", "H5 LENS-Net - w/o motion on SCVD (better)", "Motion benefit, SCVD"),
             ("K1", "K1 LENS-Net-A - LENS-Net on XD-train (better)", "Fusion − LENS-Net, XD sample"),
             ("K2", "K2 LENS-Net-A - LENS-Net on AIE (non-inferior)", "Fusion − LENS-Net, AIE (n.i.)"),
             ("K3", "K3 LENS-Net-A - CLIP probe on XD-train (non-inferior)", "Fusion − probe, XD sample (n.i.)"),
             ("K4", "K4 LENS-Net-A - CLIP probe on AIE (better)", "Fusion − probe, AIE")]
    rows = []
    for tag, key, lab in order:
        r = st[key]; rows.append((f"{tag}  {lab}", r["diff"], r["ci"], bool(r.get("holds_after_holm", r.get("holds"))), r.get("delta0", 0.0)))
        if tag in ("H1", "H3"):
            t = c45[{"H1": "H1 CLIP probe - VideoMAE-B FT", "H3": "H3 LENS-Net - VideoMAE-B FT"}[tag]]["tuned"]
            rows.append((f"      with tuned VideoMAE-B", t["diff"], t["ci"], False, 0.0))
        if tag in ("K1", "K3"):
            k = c6[{"K1": "K1 LENS-Net-A - LENS-Net (better)", "K3": "K3 LENS-Net-A - CLIP probe (non-inferior)"}[tag]]
            rows.append((f"      second XD sample", k["diff"], k["ci"], bool(k["holds_after_holm"]), -1.0 if tag == "K3" else 0.0))
    fig, ax = plt.subplots(figsize=(COL1, 2.9)); y = np.arange(len(rows))[::-1]
    ax.axvline(0, color=INK2, lw=0.6, zorder=1)
    for yi, (lab, d, ci, holds, d0) in zip(y, rows):
        ax.plot(ci, [yi, yi], color=BLUE, lw=1.0, solid_capstyle="round", zorder=2)
        ax.scatter([d], [yi], s=16, marker="o", zorder=3, linewidths=0.9, edgecolors=BLUE, facecolors=BLUE if holds else "white")
        if d0 < 0: ax.plot([d0, d0], [yi - 0.35, yi + 0.35], color=INK2, lw=0.6, ls=(0, (2, 1.5)))
        ax.text(1.015, yi, "holds" if holds else "fails", transform=ax.get_yaxis_transform(), fontsize=5.6, va="center", color=INK if holds else INK2)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=5.6); ax.tick_params(axis="y", length=0)
    style(ax, "x"); ax.spines["left"].set_visible(False); ax.set_xlabel("$\\Delta$ROC-AUC (points), 95% interval")
    fig.tight_layout(pad=0.3, rect=(0, 0, 0.9, 1)); fig.savefig(OUT / "fig_claims.pdf"); plt.close(fig)


def fig_diversity():
    e6 = J("e6_diversity.json"); pts = e6["points"]; pairs = e6["pairs"]
    name = {"CLIP probe + LENS-Net": "+ LENS-Net", "CLIP probe + w/o motion": "+ without motion", "CLIP probe + Motion probe": "+ motion probe",
            "CLIP probe + VideoMAE-B FT (tuned)": "+ VideoMAE-B (tuned)", "CLIP probe + VideoMAE-B LoRA": "+ VideoMAE-B LoRA",
            "CLIP probe + CLIP-FT": "+ CLIP-FT", "CLIP probe + VadCLIP": "+ VadCLIP",
            "LENS-Net + VideoMAE-B FT (tuned)": "LENS-Net + VideoMAE-B (tuned)", "LENS-Net + VideoMAE-B LoRA": "LENS-Net + LoRA"}
    fig, ax = plt.subplots(figsize=(COL1, 2.7))
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    ax.scatter([p["D1"] for p in pts], [p["gain"] for p in pts], s=6, c=GREY, alpha=0.55, linewidths=0, zorder=2)
    order = sorted(pairs, key=lambda k: -pairs[k]["gain"]); ys = np.linspace(4.3, -8.6, len(order))
    for k, ly in zip(order, ys):
        v = pairs[k]
        ax.scatter([v["D1"]], [v["gain"]], s=18, c=BLUE, edgecolors="white", linewidths=0.6, zorder=3)
        ax.annotate(name[k], (v["D1"], v["gain"]), xytext=(0.62, ly), textcoords="data", fontsize=5.4, color=INK, ha="left", va="center",
                    zorder=4, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.35, shrinkA=1, shrinkB=2.5))
    style(ax, "both"); ax.set_xlabel("Member diversity, 1 $-$ Spearman $\\rho$ of scores"); ax.set_ylabel("Fusion gain over better member (AUC points)")
    ax.set_ylim(-16.5, 5)
    fig.tight_layout(pad=0.3); fig.savefig(OUT / "fig_diversity.pdf"); plt.close(fig)


def fig_fusion_rules():
    """Fusion rules on the five untouched samples (review/h_partH.json): per rule the five video-level AUCs (grey),
    their mean (coloured) and range; dashed line at LENS-Net-A. Rules and groups as in the paper's fusion-rule table."""
    h = J("h_partH.json"); A = h["untouched_auc_max"]; S5 = ["XD sample A", "AIE", "XD sample B", "UCF-train", "HMDB51"]
    groups = [("Anchored averages", BLUE, [("LENS-Net-A (anchor + LENS-Net)", "LENS-Net-A"), ("AES-bal (1/2, 1/4, 1/4)", "AES-bal"),
                                            ("AES (anchor + LoRA + tuned, equal)", "AES")]),
              ("Standard ensembles", ORANGE, [("Mean of all eight candidates", "Mean of all eight"), ("Caruana selection with replacement", "Caruana selection"),
                                               ("Stacking of all eight (logistic)", "Stacking"), ("Majority vote at 0.5", "Majority vote"),
                                               ("Best single candidate (source val)", "Best single model")]),
              ("Information-fusion baselines", PURPLE, [("Calibrated averaging (S1 anchor + LENS-Net)", "Calibrated, anchor + LENS-Net"),
                                                        ("Calibrated averaging (S3 all eight)", "Calibrated, all eight"),
                                                        ("Dempster-Shafer, discounted (S1 anchor + LENS-Net)", "Dempster–Shafer, anchor + LENS-Net"),
                                                        ("Dempster-Shafer, discounted (S3 all eight)", "Dempster–Shafer, all eight"),
                                                        ("Confidence-weighted (S1 anchor + LENS-Net)", "Confidence-weighted"),
                                                        ("TENT LENS-Net + anchor", "TENT LENS-Net + anchor"),
                                                        ("TENT w/o motion + anchor", "TENT, without motion + anchor")])]
    rows = [(k, lab, c) for _, c, rs in groups for k, lab in rs]
    fig, ax = plt.subplots(figsize=(COL1, 3.35))
    la = np.mean([A["LENS-Net-A (anchor + LENS-Net)"][s] for s in S5])
    ax.axvline(la, color=BLUE, lw=0.6, ls=(0, (3, 2)), zorder=1)
    y = 0; ticks, labels, gy = [], [], []
    for gname, c, rs in groups:
        gy.append((y, gname, c)); y += 1
        for k, lab in rs:
            v = np.array([A[k][s] for s in S5])
            ax.plot([v.min(), v.max()], [y, y], color=c, lw=1.0, alpha=0.45, solid_capstyle="round", zorder=2)
            ax.scatter(v, [y] * 5, s=7, c=GREY, alpha=0.8, linewidths=0, zorder=3)
            ax.scatter([v.mean()], [y], s=24, c=c, edgecolors="white", linewidths=0.6, zorder=4)
            ticks.append(y); labels.append(lab); y += 1
        y += 0.35
    for yy, gname, c in gy: ax.text(77.1, yy, gname, fontsize=5.6, fontweight="bold", color=c, va="center", ha="left")
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=5.6); ax.invert_yaxis()
    ax.set_xlim(76.5, 99); ax.tick_params(axis="y", length=0)
    style(ax, "x"); ax.set_xlabel("Video-level ROC-AUC (%)")
    ax.text(la + 0.25, -0.65, "LENS-Net-A mean", fontsize=5.2, color=BLUE, ha="left", va="bottom")
    ax.set_ylim(y - 0.6, -1.0)
    fig.tight_layout(pad=0.3); fig.savefig(OUT / "fig_fusion_rules.pdf"); plt.close(fig)


def fig_complementarity():
    """Part I (review/i_partI.json): anchor against LENS-Net video scores on the five untouched samples (seed means, max over
    kept windows), and the share of (violent, non-violent) pairs ranked correctly by one member only, with the share of
    these that LENS-Net-A ranks correctly."""
    d = J("i_partI.json"); S5 = ["XD sample A", "AIE", "XD sample B", "UCF-train", "HMDB51"]
    names = {"XD sample A": "XD sample A", "AIE": "AIE", "XD sample B": "XD sample B", "UCF-train": "UCF-Crime train", "HMDB51": "HMDB51"}
    short = {"XD sample A": "XD A", "AIE": "AIE", "XD sample B": "XD B", "UCF-train": "UCF", "HMDB51": "HMDB"}
    fig = plt.figure(figsize=(7.0, 1.72))
    gs = fig.add_gridspec(1, 7, width_ratios=[1, 1, 1, 1, 1, 0.42, 1.3], wspace=0.32)
    for j, sn in enumerate(S5):
        ax = fig.add_subplot(gs[0, j]); sc = d["scatter"][sn]; y = np.array(sc["label"])
        a, b = np.array(sc["anchor"]), np.array(sc["lens"])
        ax.plot([0, 1], [0, 1], color=GRID, lw=0.6, zorder=1)
        ax.scatter(a[y == 0], b[y == 0], s=2.2, c=BLUE, alpha=0.45, linewidths=0, zorder=2, label="Non-violent")
        ax.scatter(a[y == 1], b[y == 1], s=2.2, c=ORANGE, alpha=0.45, linewidths=0, zorder=3, label="Violent")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.tick_params(labelsize=5.6, length=2)
        if j: ax.set_yticklabels([])
        else: ax.set_ylabel("LENS-Net score", fontsize=6.2)
        ax.set_xlabel("Anchor score", fontsize=6.2); ax.set_title(names[sn], fontsize=6.4, pad=2.5)
        ax.text(0.98, 0.03, f"$\\rho$ = {d['spearman'][sn]:.2f}", fontsize=5.6, va="bottom", ha="right", color=INK)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        if j == 0: ax.legend(loc="upper left", fontsize=5.0, frameon=False, handletextpad=0.1, markerscale=3, borderaxespad=0.1)
    ax = fig.add_subplot(gs[0, 6])
    yy = np.arange(len(S5))
    ao = np.array([100 * d["pairs"][s]["anchor_only"] for s in S5]); lo = np.array([100 * d["pairs"][s]["lens_only"] for s in S5])
    ax.barh(yy, ao, height=0.62, color=BLUE, label="Anchor only")
    ax.barh(yy, lo, left=ao + 0.25, height=0.62, color=PURPLE, label="LENS-Net only")
    for k, s in enumerate(S5):
        ax.text(ao[k] + lo[k] + 0.9, k, f"{100 * d['pairs'][s]['resolved']:.0f}", fontsize=5.6, va="center", ha="left", color=INK)
    ax.set_yticks(yy); ax.set_yticklabels([short[s] for s in S5], fontsize=5.6); ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 30); ax.set_xticks([0, 10, 20, 30]); ax.tick_params(axis="x", labelsize=5.6, length=2)
    ax.set_xlabel("Share of video pairs (%)", fontsize=6.2)
    ax.set_title("Pairs right by one member only", fontsize=6.4, pad=2.5)
    ax.legend(loc="lower right", fontsize=5.0, frameon=False, handlelength=0.8, handletextpad=0.3, borderaxespad=0.05)
    style(ax, "x")
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.22, top=0.88); fig.savefig(OUT / "fig_complementarity.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_idood(); fig_claims(); fig_diversity(); fig_fusion_rules(); fig_complementarity(); print("written")
