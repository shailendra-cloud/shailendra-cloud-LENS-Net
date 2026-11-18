"""EXPLORATORY (development data only): anchored ensemble selection. Start from the frozen anchor; greedily add the
candidate whose inclusion maximises the SOURCE-validation AUC of the equal-weight average of probabilities; stop when no
candidate improves it by more than `tol`. Weights are never fitted. The nine unseen sets are only scored."""
import json, sys
import numpy as np
from sklearn.metrics import roc_auc_score
src = open(__import__("pathlib").Path(__file__).resolve().parents[2] / "src/revision/compare_fusion_rules.py").read().split("lg = lambda")[0]
g = {"__file__": __file__}; exec(src, g)
val, yv, ANCHOR, MEMBERS, probs, D, UNSEEN, L = g["val"], g["yv"], g["ANCHOR"], g["MEMBERS"], g["probs"], g["D"], g["UNSEEN"], g["L"]
auc = lambda y, p: 100 * roc_auc_score(y, p)
VAL = {m: val(m) for m in [ANCHOR] + MEMBERS}
TGT = {(m, s): probs(m, s) for m in [ANCHOR] + MEMBERS for s in UNSEEN}
Y = {s: D.sets[s]["y"][D.sets[s]["keep"]] for s in UNSEEN}


def vscore(ms): return float(np.mean([auc(yv, np.mean([VAL[m][i] for m in ms], 0)) for i in range(3)]))


def nine(ms):
    per = {s: float(np.mean([auc(Y[s], np.mean([TGT[m, s][i] for m in ms], 0)) for i in range(3)])) for s in UNSEEN}
    return float(np.mean(list(per.values()))), float(min(per.values())), per


res = {}
for tol in (0.0, 0.05, 0.1):
    sel = [ANCHOR]; path = [(list(sel), vscore(sel), *nine(sel)[:2])]
    while True:
        cand = [m for m in MEMBERS if m not in sel]
        if not cand: break
        best = max(cand, key=lambda m: vscore(sel + [m]))
        if vscore(sel + [best]) <= vscore(sel) + tol: break
        sel.append(best); path.append((list(sel), vscore(sel), *nine(sel)[:2]))
    res[f"tol {tol}"] = [dict(members=p[0], source_val_auc=p[1], nine_mean=p[2], nine_min=p[3]) for p in path]
    print(f"tol {tol}:")
    for p in path: print(f"   {' + '.join(x.replace('VideoMAE-B ', 'VMAE ') for x in p[0]):70s} source val {p[1]:.2f}  nine-set mean {p[2]:.2f}  min {p[3]:.2f}")
m9, mn, per = nine([ANCHOR, "LENS-Net"]); print(f"reference LENS-Net-A (anchor + LENS-Net): nine-set mean {m9:.2f} min {mn:.2f}")
allm = nine([ANCHOR] + MEMBERS); print(f"reference: average of all eight models: nine-set mean {allm[0]:.2f} min {allm[1]:.2f}")
json.dump(res, open(L / "review/explore_anchored_selection.json", "w"), indent=1)
