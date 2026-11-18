#!/usr/bin/env python3
"""Journal figure: per-category results on the external benchmarks (logs/error_analysis.json and logs/clip_methods_results.json).
(a) ROC-AUC of each category's violent windows against all non-violent windows of its benchmark; (b) recall at each
model's validation threshold. Three-seed means. LENS-Net (original prompts), LENS-Net with the broad prompt set (recall
only), fine-tuned VideoMAE-B."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from make_figures import L, OUT, BLUE, ORANGE, INK, INK2, GRID, COL1, COL2, style
EA, TC = json.load(open(L / "logs/error_analysis.json")), json.load(open(L / "logs/clip_methods_results.json"))
K = ["XD-test|Fighting", "XD-test|Abuse", "XD-test|Riot", "XD-test|Shooting", "XD-test|Explosion", "XD-test|Car accident",
     "UCF-test|Fighting", "UCF-test|Robbery", "UCF-test|Shooting"]
E = EA["external_per_category"]; R = TC["category_recall"]
lab = [f"{k.split('|')[1]} ({'XD' if k.startswith('XD') else 'UCF'}, $n$={E['LENS-Net'][k]['n_pos']})" for k in K]
y = np.arange(len(K))[::-1]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL1, 4.8), gridspec_kw=dict(hspace=0.42, top=0.855))                # single column: panels stacked
for ax in (a1, a2): style(ax, "x"); ax.axhline(2.5, color=GRID, lw=0.8)
for m, c, mk, dy in (("LENS-Net", BLUE, "o", 0.12), ("VideoMAE-B FT", ORANGE, "s", -0.12)):
    a1.scatter([E[m][k]["auc"] for k in K], y + dy, c=c, marker=mk, s=22, zorder=3, edgecolors="white", linewidths=0.5)
for i, k in enumerate(K):
    a1.plot([E["LENS-Net"][k]["auc"], E["VideoMAE-B FT"][k]["auc"]], [y[i] + 0.12, y[i] - 0.12], color="#bbbbbb", lw=0.7, zorder=1)
a1.set_xlim(50, 100); a1.set_xlabel("ROC-AUC (category vs. all non-violent)"); a1.set_title("(a) Ranking quality", fontsize=7.5)
for ax_ in (a1, a2): ax_.set_yticks(y); ax_.set_yticklabels(lab, fontsize=6.3)
series = (("LENS-Net", [R["LENS-Net"][k] for k in K], BLUE, "o", "full", 0.2), ("LENS-Net, broad prompts", [R["LENS-Net (broad)"][k] for k in K], BLUE, "o", "none", 0.0),
          ("VideoMAE-B FT", [E["VideoMAE-B FT"][k]["recall"] for k in K], ORANGE, "s", "full", -0.2))
for name, v, c, mk, fill, dy in series:
    a2.scatter(v, y + dy, marker=mk, s=22, zorder=3, facecolors=c if fill == "full" else "white", edgecolors=c, linewidths=0.9 if fill == "none" else 0.5)
a2.set_xlim(0, 100); a2.set_xlabel("recall at validation threshold (%)"); a2.set_title("(b) Detection at the operating point", fontsize=7.5)
h = [Line2D([], [], marker="o", ls="", color=BLUE, label="LENS-Net"), Line2D([], [], marker="o", ls="", mfc="white", mec=BLUE, label="LENS-Net, broad prompts (b only)"),
     Line2D([], [], marker="s", ls="", color=ORANGE, label="VideoMAE-B FT")]
fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, fontsize=6.3, bbox_to_anchor=(0.45, 0.995), columnspacing=0.8, handletextpad=0.3)
fig.savefig(OUT / "fig_categories.pdf", dpi=400); fig.savefig(OUT / "fig_categories.png", dpi=400); plt.close(fig); print("ok")
