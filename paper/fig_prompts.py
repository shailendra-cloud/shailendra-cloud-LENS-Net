#!/usr/bin/env python3
"""Journal figure: which fixed prompts separate violent from non-violent windows. For each prompt and test set: Cohen's d
between violent and non-violent windows of the PPS 'grid max, mean over frames' feature (max similarity over the 16 grid
cells, averaged over the 16 frames). Original prompts, frozen CLIP ViT-L/14; no model training involved."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from make_figures import L, OUT, INK, INK2, COL2
sys.path[:0] = [str(L.parent / "src" / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]; import lensnet_setup as LN
F = L.parent / "workspace/features/l14"; ROOT = L.parent
def feat(name): return np.load(F / name / "compact.npz")["S"].astype(np.float32)[:, :, 1:, :].max(2).mean(1)   # (N,10)
man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv"); te = (man.split == "test").values
ext = pd.read_csv(ROOT / "data_tables/manifests/external_windows.csv")
pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True)
fw = pd.read_csv(L / "fresh/near_duplicate_exclusion.csv"); cw = pd.read_csv(L / "conf/conf_windows.csv"); cw = cw[cw.decode_failed == 0].reset_index(drop=True)
fb, fe, ff, fc = feat("bench"), feat("external"), feat("fresh"), feat("conf")
SETS = [("Audited\ntest", fb[te], man.binary_label.values[te]),
        ("XD-\nViolence", fe[((pool.benchmark == "XD-test") & (pool.excluded_precaution == 0)).values], pool.binary_label.values[((pool.benchmark == "XD-test") & (pool.excluded_precaution == 0)).values]),
        ("UCF-\nCrime", fe[((pool.benchmark == "UCF-test") & (pool.excluded_precaution == 0)).values], pool.binary_label.values[((pool.benchmark == "UCF-test") & (pool.excluded_precaution == 0)).values])]
for b in ("SCFD", "SCVD"):
    k = (fw.excluded_near_duplicate.values == 0) & (fw.benchmark == b).values; SETS.append((b, ff[k], fw.binary_label.values[k]))
for b in ("VioPeru", "AIRTLab"):
    k = (cw.benchmark == b).values; SETS.append((b, fc[k], cw.binary_label.values[k]))
def d(x, y):
    a, b = x[y == 1], x[y == 0]; s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)); return (a.mean() - b.mean()) / s
prompts = LN.PROMPT_SETS["orig"][0] + LN.PROMPT_SETS["orig"][1]
D = np.array([[d(x[:, p], y) for _, x, y in SETS] for p in range(10)])
json.dump(dict(prompts=prompts, sets=[s[0].replace("\n", " ") for s in SETS], cohens_d=D.round(6).tolist()), open(L / "logs/fig_prompts_values.json", "w"), indent=2)
fig, ax = plt.subplots(figsize=(COL2 * 0.62, 3.0))
vm = np.abs(D).max(); im = ax.imshow(D, cmap="RdBu_r", norm=colors.TwoSlopeNorm(vmin=-vm, vcenter=0, vmax=vm), aspect="auto")
for i in range(10):
    for j in range(len(SETS)):
        ax.text(j, i, f"{D[i, j]:.2f}", ha="center", va="center", fontsize=6.2, color="white" if abs(D[i, j]) > 0.6 * vm else INK)
ax.set_xticks(range(len(SETS))); ax.set_xticklabels([s[0] for s in SETS], fontsize=6.6)
ax.set_yticks(range(10)); ax.set_yticklabels([f"“{p}”" for p in prompts], fontsize=6.6)
ax.axhline(4.5, color="white", lw=1.6)
for yy, lab, col in ((2.0, "non-violent\nprompts", "#2166ac"), (7.0, "violent\nprompts", "#b2182b")):
    ax.text(len(SETS) - 0.35, yy, lab, rotation=0, ha="left", va="center", fontsize=6.6, color=col, fontweight="bold")
ax.tick_params(length=0); [sp.set_visible(False) for sp in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.17); cb.ax.tick_params(labelsize=6.2)
cb.set_label("Cohen's $d$ (violent $-$ non-violent windows)", fontsize=6.6)
fig.savefig(OUT / "fig_prompts.pdf", dpi=400); fig.savefig(OUT / "fig_prompts.png", dpi=400); plt.close(fig)
print(np.round(D, 2))
