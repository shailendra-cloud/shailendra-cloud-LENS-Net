#!/usr/bin/env python3
"""Journal figure: t-SNE of (a) raw CLIP clip embeddings u and (b) LENS-Net's PPS descriptor phi, for windows from
corpora never used in training (XD, UCF, SCFD, SCVD) plus the audited test split. Colour = label, marker = corpus.
Silhouette scores (cosine; label vs corpus) quantify the visual impression."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).parent)); from make_figures import *
sys.path[:0] = [str(L.parent / "src" / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]; import lensnet as V
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
rng = np.random.default_rng(0)
man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["split", "dataset", "binary_label"])
ext = pd.read_csv(ROOT / "data_tables/manifests/external_windows.csv")
pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True)
fx = pd.read_csv(L / "fresh/near_duplicate_exclusion.csv")
parts = []
b = V.Compact("bench"); idx = np.where(man.split == "test")[0]
parts.append((b, idx, man.binary_label.values[idx], np.array(["audited test"] * len(idx))))
e = V.Compact("external")
for bb, nm in [("XD-test", "XD-Violence"), ("UCF-test", "UCF-Crime")]:
    i = np.where((pool.excluded_precaution == 0) & (pool.benchmark == bb))[0]; parts.append((e, i, pool.binary_label.values[i], np.array([nm] * len(i))))
f = V.Compact("fresh")
for bb in ["SCFD", "SCVD"]:
    i = np.where((fx.excluded_near_duplicate == 0) & (fx.benchmark == bb))[0]; parts.append((f, i, fx.binary_label.values[i], np.array([bb] * len(i))))
U, P, Y, D = [], [], [], []
for F_, i, y, d in parts:
    k = rng.choice(len(i), min(len(i), 700), replace=False); r = torch.as_tensor(i[k])
    U.append(torch.nn.functional.normalize(F_.cls[r], dim=-1).numpy()); P.append(V.pps_features(F_.S[r], "V2b").numpy()); Y.append(y[k]); D.append(d[k])
U, P, Y, D = np.concatenate(U), np.concatenate(P), np.concatenate(Y), np.concatenate(D)
P = (P - P.mean(0)) / (P.std(0) + 1e-8)
sil = {n: {"by_label": float(silhouette_score(X, Y, metric="cosine")), "by_corpus": float(silhouette_score(X, D, metric="cosine"))} for n, X in [("raw CLIP u", U), ("PPS phi", P)]}
print(json.dumps(sil, indent=1))
json.dump(sil, open(OUT / "fig_tsne_silhouette.json", "w"), indent=1)
emb = {n: TSNE(2, perplexity=40, init="pca", random_state=0, metric="cosine").fit_transform(X) for n, X in [("raw CLIP u", U), ("PPS phi", P)]}
MK = {"audited test": "o", "XD-Violence": "s", "UCF-Crime": "^", "SCFD": "D", "SCVD": "v"}
fig, axs = plt.subplots(1, 2, figsize=(COL2 * 0.78, 2.4))
for ax, (n, Z), title in zip(axs, emb.items(), ["(a) raw CLIP embedding $u$", "(b) LENS-Net PPS descriptor $\\phi$"]):
    for d, mk in MK.items():
        for lab, c in [(0, ORANGE), (1, BLUE)]:
            m = (D == d) & (Y == lab); ax.scatter(Z[m, 0], Z[m, 1], s=4, marker=mk, color=c, alpha=0.55, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title(f"{title}\nsilhouette: label {sil[n]['by_label']:.2f}, corpus {sil[n]['by_corpus']:.2f}", fontsize=7, color=INK)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="violent"), Line2D([], [], marker="o", ls="", color=ORANGE, ms=4, label="non-violent")] + \
    [Line2D([], [], marker=mk, ls="", color=GREY1, ms=4, label=d) for d, mk in MK.items()]
fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=7, frameon=False, fontsize=6, handletextpad=0.2, columnspacing=0.8)
fig.savefig(OUT / "fig_tsne.pdf"); print("saved")
