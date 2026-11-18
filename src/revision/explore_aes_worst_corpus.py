"""EXPLORATORY (source data only for selection): AES with worst-source-corpus selection. The greedy criterion is the
minimum over source corpora of the source-validation AUC of the equal-weight average (instead of the pooled AUC).
Prints what it selects; the nine development sets are scored descriptively. No confirmation data are used."""
import sys, numpy as np
from sklearn.metrics import roc_auc_score
src = open(__import__("pathlib").Path(__file__).resolve().parents[2] / "src/revision/compare_fusion_rules.py").read().split("lg = lambda")[0]
g = {"__file__": __file__}; exec(src, g)
val, yv, va, ANCHOR, MEMBERS, probs, D, UNSEEN = g["val"], g["yv"], g["va"], g["ANCHOR"], g["MEMBERS"], g["probs"], g["D"], g["UNSEEN"]
auc = lambda y, p: 100 * roc_auc_score(y, p)
VAL = {m: val(m) for m in [ANCHOR] + MEMBERS}; grp = va.dataset.values
groups = [d for d in np.unique(grp) if len(np.unique(yv[grp == d])) == 2]
print("source corpora with both classes in validation:", {d: int((grp == d).sum()) for d in groups})
def worst(ms): return float(np.mean([min(auc(yv[grp == d], np.mean([VAL[m][i] for m in ms], 0)[grp == d]) for d in groups) for i in range(3)]))
def pooled(ms): return float(np.mean([auc(yv, np.mean([VAL[m][i] for m in ms], 0)) for i in range(3)]))
def nine(ms):
    per = [np.mean([auc(D.sets[s]["y"][D.sets[s]["keep"]], np.mean([probs(m, s)[i] for m in ms], 0)) for i in range(3)]) for s in UNSEEN]
    return float(np.mean(per)), float(min(per))
for m in [ANCHOR] + MEMBERS:
    print(f"  {m:24s} worst-corpus val {worst([m]):6.2f}  pooled val {pooled([m]):6.2f}  " +
          " ".join(f"{d[:6]} {np.mean([auc(yv[grp == d], VAL[m][i][grp == d]) for i in range(3)]):.1f}" for d in groups))
for crit, f in (("pooled", pooled), ("worst-corpus", worst)):
    sel = [ANCHOR]
    while True:
        cand = [m for m in MEMBERS if m not in sel]
        if not cand: break
        best = max(cand, key=lambda m: f(sel + [m]))
        if f(sel + [best]) <= f(sel) + 0.05: break
        sel.append(best)
    n9 = nine(sel); print(f"{crit:13s} selects {sel}  criterion {f(sel):.2f}  nine-set (dev) mean {n9[0]:.2f} min {n9[1]:.2f}")
