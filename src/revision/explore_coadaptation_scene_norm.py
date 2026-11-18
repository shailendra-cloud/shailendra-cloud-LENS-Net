"""EXPLORATORY (development data only): CVCA with scene normalisation - each window's features minus alpha x the mean
feature of its video (removes the camera/scene signature that a target head could use as a shortcut)."""
import sys, json, numpy as np
src = open(__import__("pathlib").Path(__file__).resolve().parents[2] / "src/revision/explore_cross_view_coadaptation.py").read().split("auc = lambda")[0]
g = {"__file__": __file__}; exec(src, g)
from sklearn.metrics import roc_auc_score
D, UNSEEN, views, cvca, pct, ANCHOR, L = g["D"], g["UNSEEN"], g["views"], g["cvca"], g["pct"], g["ANCHOR"], g["L"]
auc = lambda y, p: 100 * roc_auc_score(y, p)
def centre(X, cl, a):
    X = X.copy(); u, inv = np.unique(cl, return_inverse=True)
    mu = np.zeros((len(u), X.shape[1])); np.add.at(mu, inv, X); mu /= np.bincount(inv)[:, None]
    return X - a * mu[inv]
res = {}
for vname, vm in (("LoRA", "VideoMAE-B LoRA"), ("tuned", "VideoMAE-B FT (tuned)")):
    for a_ in (0.0, 0.5, 1.0):
        tag = f"{vname} q0.3 r2 centre{a_}"; per_b, per_c = {}, {}
        for s in UNSEEN:
            S = D.sets[s]; y = S["y"][S["keep"]]; cl = np.asarray(S["cl"])[S["keep"]]; sem, mot = views(s)
            sem, mot = centre(sem, cl, a_), centre(mot, cl, a_)
            PA = [p[S["keep"]] for p in D.probs(ANCHOR, s)]; PV = [p[S["keep"]] for p in D.probs(vm, s)]
            b, c = [], []
            for i in range(3):
                hs, hm = cvca(PA[i], PV[i], sem, mot, cl, 0.3, 2)
                b.append(auc(y, (PA[i] + PV[i]) / 2)); c.append(auc(y, pct(PA[i]) + pct(PV[i]) + pct(hs) + pct(hm)))
            per_b[s], per_c[s] = float(np.mean(b)), float(np.mean(c))
        res[tag] = per_c
        d = {s: per_c[s] - per_b[s] for s in UNSEEN}
        print(f"{tag:26s} mean9 {np.mean(list(per_c.values())):6.2f} (base {np.mean(list(per_b.values())):.2f})  min9 {min(per_c.values()):6.2f}  +{sum(v>0.5 for v in d.values())}/-{sum(v<-0.5 for v in d.values())}  " + " ".join(f"{s[:4]}{d[s]:+.1f}" for s in UNSEEN), flush=True)
json.dump(res, open(L / "review/explore_coadapt2.json", "w"), indent=1)
n = {s: len(np.unique(np.asarray(D.sets[s]["cl"])[D.sets[s]["keep"]])) for s in UNSEEN}; print("videos/clusters per set:", n)
