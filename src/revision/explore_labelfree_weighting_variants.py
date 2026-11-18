"""EXPLORATORY (development data only): stronger label-free fusion variants. Reuses the loaders and the estimator of explore_labelfree_weighting.py."""
import json, sys, runpy, io, contextlib
import numpy as np
sys.argv = ["x"]
src = open(__import__("pathlib").Path(__file__).resolve().parents[2] / "src/revision/explore_labelfree_weighting.py").read().split("auc = lambda")[0]
g = {"__file__": __file__}; exec(src, g)
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
z, sml, fuse, PRf, UNSEEN, D, ANCHOR = g["z"], g["sml_weights"], g["fuse"], g["probs"], g["UNSEEN"], g["D"], g["ANCHOR"]
auc = lambda y, p: 100 * roc_auc_score(y, p)
SETS_M = {"M4": [ANCHOR, "LENS-Net", "VideoMAE-B LoRA", "VideoMAE-B FT (tuned)"],
          "M5": [ANCHOR, "LENS-Net", "VideoMAE-B LoRA", "VideoMAE-B FT (tuned)", "Motion probe"],
          "M3v": [ANCHOR, "VideoMAE-B LoRA", "VideoMAE-B FT (tuned)"]}
mem = sorted({m for v in SETS_M.values() for m in v}); PR = {(m, s): PRf(m, s) for m in mem for s in UNSEEN}
Y = {s: D.sets[s]["y"][D.sets[s]["keep"]] for s in UNSEEN}
def variant(kind, P):
    M = len(P); u = np.full(M, 1 / M)
    if kind == "equal": return fuse(P, u)
    if kind == "equal ranks": return sum(z(p) for p in P)
    v, _ = sml(P); v = np.clip(v, 1e-6, None)
    if kind.startswith("sharp"): k = float(kind.split()[1]); return fuse(P, v ** k)
    if kind == "drop weakest": keep = np.argsort(v)[1:]; return fuse([P[i] for i in keep], np.ones(len(keep)))
    if kind == "consensus corr":
        Z = [z(p) for p in P]; c = np.array([np.corrcoef(Z[i], np.mean([Z[j] for j in range(M) if j != i], 0))[0, 1] for i in range(M)])
        return fuse(P, np.clip(c, 1e-6, None) ** 2)
    if kind == "drop least consensual":
        Z = [z(p) for p in P]; c = np.array([np.corrcoef(Z[i], np.mean([Z[j] for j in range(M) if j != i], 0))[0, 1] for i in range(M)])
        keep = np.argsort(c)[1:]; return fuse([P[i] for i in keep], np.ones(len(keep)))
KINDS = ["equal", "equal ranks", "sharp 2", "sharp 4", "drop weakest", "consensus corr", "drop least consensual"]
out = {}
print(f"{'set | variant':40s} mean9   min9  (sets better/worse than equal by >0.5)")
for name, ms in SETS_M.items():
    base = None
    for kd in KINDS:
        per = {s: float(np.mean([auc(Y[s], variant(kd, [PR[m, s][i] for m in ms])) for i in range(3)])) for s in UNSEEN}
        if kd == "equal": base = per
        b = sum(per[s] - base[s] > 0.5 for s in UNSEEN); w = sum(per[s] - base[s] < -0.5 for s in UNSEEN)
        out[f"{name} | {kd}"] = per; print(f"{name+' | '+kd:40s} {np.mean(list(per.values())):6.2f} {min(per.values()):6.2f}   +{b} / -{w}")
json.dump(out, open(str(__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace/review/explore_saaf2.json"), "w"), indent=1)
