"""C2 analysis (PREREG_revision.md): OpenCLIP ViT-L/14 (LAION-2B) in place of OpenAI CLIP ViT-L/14. Per-set and nine-set
AUCs of the LAION CLIP probe, model without motion and LENS-Net (seeds 42-44), and the seed-aware tests of
(LAION probe - VideoMAE-B FT), (LAION LENS-Net - VideoMAE-B FT) and LAION vs OpenAI for each model. Writes review/c2_laion.json."""
import json, sys
import numpy as np, pandas as pd
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
from saved_predictions import Data, SETS, UNSEEN, L
from seed_aware_bootstrap import Boot, summarize
D = Data(); RH = L / "runs_review_heads"
LAION = {"LAION CLIP probe": "LAION_audited_LP_none", "LAION w/o motion": "LAION_audited_V2b_vrex1", "LAION LENS-Net": "LAION_audited_LENS-M_vrex1"}
OPENAI = {"LAION CLIP probe": "CLIP probe", "LAION w/o motion": "w/o motion", "LAION LENS-Net": "LENS-Net"}

def raw(run, s):
    S = D.sets[s]
    if s in ("XD", "UCF"): return pd.read_csv(RH / run / "predictions_external.csv").p_violent.values[S["rows"]]
    d = pd.read_csv(RH / run / f"predictions_{S['dir']}.csv").p_violent.values; assert len(d) == S["nwin"]; return d[S["rows"]]

def probs(model, s):
    k = D.sets[s]["keep"]
    if model == "LAION LENS-Net-A": return [(a + b) / 2 for a, b in zip(probs("LAION LENS-Net", s), probs("LAION CLIP probe", s))]
    if model in LAION: return [raw(f"{LAION[model]}_seed{r}", s)[k] for r in (42, 43, 44)]
    return [p[k] for p in D.probs(model, s)]

evals = {s: (D.sets[s]["y"][D.sets[s]["keep"]], np.asarray(D.sets[s]["cl"])[D.sets[s]["keep"]]) for s in UNSEEN}
BT = Boot(evals); cache = {}
def H(m, s):
    if (m, s) not in cache: cache[(m, s)] = BT.hier(m, BT.aucs(s, probs(m, s)))
    return cache[(m, s)]
ok9 = np.all([BT.S[s]["ok"] for s in UNSEEN], 0)
models = list(LAION) + ["LAION LENS-Net-A", "CLIP probe", "w/o motion", "LENS-Net", "LENS-Net-A", "VideoMAE-B FT"]
out = {"auc": {}, "tests": {}}
for m in models:
    per = {s: float(H(m, s)[0]) for s in UNSEEN}; per["mean9"] = float(np.mean(list(per.values()))); per["min9"] = float(min(per[s] for s in UNSEEN)); out["auc"][m] = per
def nine(a, c): return summarize(np.mean([H(a, s) - H(c, s) for s in UNSEEN], 0), ok9)
for a, c in [("LAION CLIP probe", "VideoMAE-B FT"), ("LAION LENS-Net", "VideoMAE-B FT"), ("LAION LENS-Net-A", "VideoMAE-B FT"),
             ("LAION CLIP probe", "LAION LENS-Net"), ("LAION LENS-Net-A", "LAION LENS-Net")] + [(m, OPENAI[m]) for m in LAION] + [("LAION LENS-Net-A", "LENS-Net-A")]:
    out["tests"][f"{a} - {c} (nine-set mean)"] = nine(a, c)
json.dump(out, open(L / "review/c2_laion.json", "w"), indent=1)
print(f"{'model':20s} " + " ".join(f"{s[:6]:>6s}" for s in UNSEEN) + "  mean9  min9")
for m, v in out["auc"].items(): print(f"{m:20s} " + " ".join(f"{v[s]:6.2f}" for s in UNSEEN) + f"  {v['mean9']:5.2f} {v['min9']:5.2f}")
for k, r in out["tests"].items(): print(f"{k:55s} {r['diff']:+6.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] p={r['p_two_sided']:.4f}")
