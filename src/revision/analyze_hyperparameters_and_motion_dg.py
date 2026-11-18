"""E2 (s, lambda_A sensitivity) and E3 (DG objectives on the motion probe) of PREREG_revision Part E: nine-set mean and
minimum AUC (mean of three seeds) of runs_review_heads, next to the published models. Writes review/e23.json."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
from saved_predictions import Data, SETS, UNSEEN, L
D = Data(); RH = L / "runs_review_heads"
def preds(run, s):
    S = D.sets[s]
    if s == "Test": return pd.read_csv(RH / run / "predictions_test.csv").p_violent.values
    if s in ("XD", "UCF"): return pd.read_csv(RH / run / "predictions_external.csv").p_violent.values[S["rows"]]
    return pd.read_csv(RH / run / f"predictions_{S['dir']}.csv").p_violent.values[S["rows"]]
def auc(s, p): k = D.sets[s]["keep"]; return 100 * roc_auc_score(D.sets[s]["y"][k], p[k])
pat = re.compile(r"^L14_audited_(LP_none|LENS-M_vrex1)_(s15|s60|la0|la0\.1|la2)$|^L14_audited_MP_(none|irm|gdro|vrex1)$")
groups = {}
for d in sorted(RH.iterdir()):
    g = re.sub(r"_seed\d+$", "", d.name)
    if (d / "summary.json").exists() and pat.match(g): groups.setdefault(g, []).append(d.name)
out = {}
for g, runs in sorted(groups.items()):
    assert len(runs) == 3, (g, runs)
    per = {s: float(np.mean([auc(s, preds(r, s)) for r in runs])) for s in SETS}
    per["mean9"] = float(np.mean([per[s] for s in UNSEEN])); per["min9"] = float(min(per[s] for s in UNSEEN)); out[g] = per
for ref in ("LENS-Net", "CLIP probe", "Motion probe"):
    per = {s: float(np.mean([auc(s, p) for p in D.probs(ref, s)])) for s in SETS}
    per["mean9"] = float(np.mean([per[s] for s in UNSEEN])); per["min9"] = float(min(per[s] for s in UNSEEN)); out["published " + ref] = per
json.dump(out, open(L / "review/e23.json", "w"), indent=1)
for g, v in out.items(): print(f"{g:34s} test {v['Test']:.2f} mean9 {v['mean9']:.2f} min9 {v['min9']:.2f}  Movies {v['Movies']:.2f}")
