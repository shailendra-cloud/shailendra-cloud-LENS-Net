"""C1 (frozen part) and C3 analysis: AUCs of runs_review_heads on the audited test split and the nine unseen sets,
with the same rows, labels and filters as saved_predictions.Data (which reproduces Table 6)."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from saved_predictions import Data, SETS, UNSEEN, L
D = Data(); RH = L / "runs_review_heads"
def preds(run, s):
    S = D.sets[s]
    if s == "Test": return pd.read_csv(RH / run / "predictions_test.csv").p_violent.values
    if s in ("XD", "UCF"): return pd.read_csv(RH / run / "predictions_external.csv").p_violent.values[S["rows"]]
    return pd.read_csv(RH / run / f"predictions_{S['dir']}.csv").p_violent.values[S["rows"]]
def auc(s, p): k = D.sets[s]["keep"]; return 100 * roc_auc_score(D.sets[s]["y"][k], p[k])
out = {}
groups = {}
for d in sorted(RH.iterdir()):
    if not (d / "summary.json").exists() or "_audited_" not in d.name: continue
    groups.setdefault(re.sub(r"_seed\d+$", "", d.name), []).append(d.name)
for g, runs in sorted(groups.items()):
    if len(runs) < 3: continue
    per = {s: float(np.mean([auc(s, preds(r, s)) for r in runs])) for s in SETS}
    per["mean9"] = float(np.mean([per[s] for s in UNSEEN])); per["min9"] = float(np.min([per[s] for s in UNSEEN])); per["seeds"] = len(runs)
    out[g] = per
for ref, nm in (("LENS-Net", "published LENS-Net (beta=1)"), ("CLIP probe", "published CLIP probe")):
    per = {s: float(np.mean([auc(s, p) for p in D.probs(ref, s)])) for s in SETS}
    per["mean9"] = float(np.mean([per[s] for s in UNSEEN])); per["min9"] = float(np.min([per[s] for s in UNSEEN])); out[nm] = per
json.dump(out, open(L / "review/c3_dg.json", "w"), indent=1)
print(f"{'model':38s} " + " ".join(f"{s[:6]:>6s}" for s in SETS) + "  mean9  min9")
for g, v in out.items(): print(f"{g:38s} " + " ".join(f"{v[s]:6.2f}" for s in SETS) + f"  {v['mean9']:5.2f} {v['min9']:5.2f}")
