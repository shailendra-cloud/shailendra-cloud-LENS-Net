#!/usr/bin/env python3
"""Journal figure: accuracy vs inference cost. y = mean ROC-AUC over the six unseen test sets (XD, UCF, SCFD, SCVD,
VioPeru, AIRTLab; P1 checkpoints, three-seed means); x = latency per 1.5-s window (logs/latency.json); marker area =
trained parameters (log scale). Every value is read from saved results."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from make_figures import L, OUT, BLUE, ORANGE, GREY1, GREY2, INK, INK2, COL1, style
J = lambda n: json.load(open(L / "logs" / n))
TC, A1, FR, CF, SF, LT = J("clip_methods_results.json"), J("A1_results.json"), J("fresh_results.json"), J("conf_results.json"), J("stageF_results.json"), J("latency.json")
SETS = ["XD", "UCF", "SCFD", "SCVD", "VioPeru", "AIRTLab"]
auc = lambda y, p: 100 * roc_auc_score(y, p)
ext = pd.read_csv(L.parent / "data_tables/manifests/external_windows.csv")
POOL = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True)
fw = pd.read_csv(L / "fresh/near_duplicate_exclusion.csv"); keep = fw.excluded_near_duplicate.values == 0
def lite(tag, rdir):                                        # LENS-Net variants: from their prediction files
    v = {s: [] for s in SETS}
    for sd in (42, 43, 44):
        e = pd.read_csv(L / rdir / f"v2_audited_LENS-M_vrex_seed{sd}/predictions_external.csv")
        for b, n in (("XD-test", "XD"), ("UCF-test", "UCF")):
            k = ((e.benchmark == b) & (e.excluded_precaution == 0)).values; v[n].append(auc(e.binary_label[k], e.p_violent[k]))
        p = pd.read_csv(L / f"fresh/preds/{tag}v2_audited_LENS-M_vrex_seed{sd}.csv").p_violent.values
        for n in ("SCFD", "SCVD"):
            k = keep & (fw.benchmark == n).values; v[n].append(auc(fw.binary_label[k], p[k]))
    g = CF["models"][f"{tag}v2_audited_LENS-M_vrex"]
    return np.mean([np.mean(v[s]) for s in SETS[:4]] + [g["VioPeru"]["mean"], g["AIRTLab"]["mean"]])
def other(xd, ucf, fk, ck): return np.mean([xd, ucf, FR["models"][fk]["SCFD"]["mean"], FR["models"][fk]["SCVD"]["mean"], CF["models"][ck]["VioPeru"]["mean"], CF["models"][ck]["AIRTLab"]["mean"]])
tc = lambda m: np.mean([TC["P1"][m][s]["mean"] for s in SETS])
mast = SF["F1_F3_table"]["MAST-Net"]
M = [  # label, mean unseen AUC, latency ms, trained params, colour, marker, family
    ("LENS-Net", tc("LENS-Net"), LT["LENS-Net (L14, 16f)"], 31926, BLUE, "o", "lens"),
    ("L1 (8 fr.)", lite("f8_", "runs_f8"), LT["L1 (L14, 8f)"], 31926, BLUE, "o", "lens"),
    ("L2 (B/16)", lite("b16_", "runs_b16"), LT["L2 (B16, 16f)"], 31926, BLUE, "o", "lens"),
    ("L3 (B/16, 8 fr.)", lite("b16_f8_", "runs_b16_f8"), LT["L3 (B16, 8f)"], 31926, BLUE, "o", "lens"),
    ("VideoMAE-B FT", tc("VideoMAE-B FT"), LT["VideoMAE-B FT"], 86.2e6, ORANGE, "s", "o"),
    ("VadCLIP (L/14)", tc("VadCLIP (L14)"), LT["VadCLIP (L14)"], 21.0e6, GREY1, "P", "o"),
    ("X-CLIP FT", tc("X-CLIP FT (lr 5e-6)"), LT["X-CLIP FT"], 126.9e6, GREY1, "X", "o"),
    ("Video Swin-T FT", other(A1["A1_swin3d_t"]["xd_auc"]["mean"], A1["A1_swin3d_t"]["ucf_auc"]["mean"], "A1_swin3d_t", "A1_swin3d_t"), LT["Video Swin-T FT"], 27.9e6, GREY2, "D", "o"),
    ("R(2+1)D-18 FT", other(A1["A1_r2plus1d_18"]["xd_auc"]["mean"], A1["A1_r2plus1d_18"]["ucf_auc"]["mean"], "A1_r2plus1d_18", "A1_r2plus1d_18"), LT["R(2+1)D-18 FT"], 31.3e6, GREY2, "^", "o"),
    ("CNN–BiGRU", other(mast["xd_auc"]["mean"], mast["ucf_auc"]["mean"], "MAST-Net", "MAST-Net"), LT["CNN-BiGRU"], 6.1e6, GREY2, "v", "o")]
json.dump({m[0]: dict(mean_unseen_auc=float(m[1]), ms=float(m[2]), params=m[3]) for m in M}, open(L / "logs/fig_efficiency_values.json", "w"), indent=2)
fig, ax = plt.subplots(figsize=(COL1, 2.75)); style(ax, "both")
size = lambda p: 18 + 22 * np.log10(p / 1e4)
lx = [m for m in M if m[6] == "lens"]
ax.plot([m[2] for m in lx], [m[1] for m in lx], color=BLUE, lw=0.8, ls="--", zorder=1)
OFF = {"LENS-Net": (0, 9, "center"), "L1 (8 fr.)": (-3, 8, "right"), "L2 (B/16)": (0, -12, "center"), "L3 (B/16, 8 fr.)": (-4, 5, "right"),
       "VideoMAE-B FT": (6, -3, "left"), "VadCLIP (L/14)": (2, -11, "center"), "X-CLIP FT": (6, -2, "left"), "Video Swin-T FT": (6, 2, "left"),
       "R(2+1)D-18 FT": (6, -3, "left"), "CNN–BiGRU": (6, -3, "left")}
for lab, y, x, p, c, mk, fam in M:
    ax.scatter(x, y, s=size(p), c=c, marker=mk, edgecolors="white", linewidths=0.6, zorder=3)
    dx, dy, ha = OFF[lab]
    ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.3, ha=ha, va="center",
                color=INK if fam == "lens" else INK2, fontweight="bold" if lab == "LENS-Net" else "normal")
ax.set_xscale("log"); ax.set_xlim(right=150); ax.set_ylim(top=86.6); ax.set_xticks([10, 20, 50, 100]); ax.set_xticklabels(["10", "20", "50", "100"])
ax.set_xlabel("latency per 1.5-s window (ms, log scale)"); ax.set_ylabel("mean ROC-AUC, six unseen sets")
for p, lab in ((3.2e4, "32k"), (1e6, "1M"), (1e8, "100M")):
    ax.scatter([], [], s=size(p), c="#cfcfcf", edgecolors="white", label=lab)
ax.legend(title="trained params", title_fontsize=6.3, fontsize=6.3, loc="lower right", frameon=False, labelspacing=1.3, borderpad=0.2, handletextpad=0.4)
fig.savefig(OUT / "fig_efficiency.pdf", dpi=400); fig.savefig(OUT / "fig_efficiency.png", dpi=400); plt.close(fig)
for m in M: print(f"{m[0]:18s} unseen {m[1]:.2f}  {m[2]:.0f} ms")
