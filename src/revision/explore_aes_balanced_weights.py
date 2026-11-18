"""EXPLORATORY, development sets only: do the two fixes suggested by the AES diagnosis help where the method was built?
(a) balanced evidence weights: anchor 1/2, each video member 1/4; (b) rank-normalised scores (equal weights);
(c) both. Confirmation sets are not used."""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
src = open(__import__("pathlib").Path(__file__).resolve().parents[2] / "src/revision/compare_fusion_rules.py").read().split("lg = lambda")[0]
g = {"__file__": __file__}; exec(src, g)
probs, D, UNSEEN, ANCHOR = g["probs"], g["D"], g["UNSEEN"], g["ANCHOR"]
auc = lambda y, p: 100 * roc_auc_score(y, p); z = lambda p: (rankdata(p) - 0.5) / len(p)
M = [ANCHOR, "VideoMAE-B LoRA", "VideoMAE-B FT (tuned)"]
V = {"AES (equal weights, as tested)": ([1/3, 1/3, 1/3], False), "(a) balanced 1/2, 1/4, 1/4": ([.5, .25, .25], False),
     "(b) rank-normalised, equal": ([1/3, 1/3, 1/3], True), "(c) balanced + rank-normalised": ([.5, .25, .25], True),
     "reference: anchor + LENS-Net (LENS-Net-A)": None}
base = None; OUT = {}
for name, cfg in V.items():
    per = {}
    for s in UNSEEN:
        y = D.sets[s]["y"][D.sets[s]["keep"]]; a = []
        for i in range(3):
            if cfg is None: f = (probs(ANCHOR, s)[i] + probs("LENS-Net", s)[i]) / 2
            else:
                w, rk = cfg; P = [probs(m, s)[i] for m in M]; f = sum(wi * (z(p) if rk else p) for wi, p in zip(w, P))
            a.append(auc(y, f))
        per[s] = np.mean(a)
    if base is None: base = per
    OUT[name] = {**{k: float(v) for k, v in per.items()}, "mean9": float(np.mean(list(per.values()))), "min9": float(min(per.values()))}
    d = {s: per[s] - base[s] for s in UNSEEN}
    print(f"{name:42s} mean9 {np.mean(list(per.values())):6.2f}  min9 {min(per.values()):6.2f}   vs AES: XD {d['XD']:+.2f} Movies {d['Movies']:+.2f} AIRTLab {d['AIRTLab']:+.2f} | UCF {d['UCF']:+.2f} CCTV {d['CCTV-F']:+.2f} UBI {d['UBI-Fights']:+.2f}")
import json; json.dump(OUT, open(str(__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace/review/explore_aes_fixes_dev.json"), "w"), indent=1)
