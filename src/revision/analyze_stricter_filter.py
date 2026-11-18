"""C7 (PREREG_revision.md): test-side near-duplicate filter at max cosine >= 0.90 instead of >= 0.95 for the seven
unseen sets that carry the score; XD/UCF unchanged. Per-set AUCs, Kendall tau, H1-H3 at both thresholds.
Writes review/c7_filter090.json."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
from saved_predictions import Data, MODELS, SETS, UNSEEN, L
from seed_aware_bootstrap import Boot, summarize
D = Data(); keep95 = {s: D.sets[s]["keep"].copy() for s in SETS}; keep90 = dict(keep95)
for s in UNSEEN:
    S = D.sets[s]
    if "dir" not in S: continue
    c = pd.read_csv(L / S["dir"] / "near_duplicate_exclusion.csv").max_cos_to_train.values[S["rows"]]
    assert ((c >= 0.95) == ~keep95[s]).all(); keep90[s] = c < 0.90
models = list(MODELS) + ["LENS-Net-A"]
out = {"removed": {}, "auc": {}, "tau": {}, "claims": {}}
for s in UNSEEN:
    out["removed"][s] = dict(at095=int((~keep95[s]).sum()), at090=int((~keep90[s]).sum()), windows=int(len(keep95[s])))
    y = D.sets[s]["y"]; res = {}
    for m in models:
        ps = D.probs(m, s)
        res[m] = dict(at095=float(np.mean([100 * roc_auc_score(y[keep95[s]], p[keep95[s]]) for p in ps])),
                      at090=float(np.mean([100 * roc_auc_score(y[keep90[s]], p[keep90[s]]) for p in ps])))
    out["auc"][s] = res; out["tau"][s] = float(kendalltau([res[m]["at095"] for m in models], [res[m]["at090"] for m in models])[0])
out["mean9"] = {m: {t: float(np.mean([out["auc"][s][m][t] for s in UNSEEN])) for t in ("at095", "at090")} for m in models}
H1H3 = {"H1 CLIP probe - VideoMAE-B FT": ("CLIP probe", "VideoMAE-B FT"), "H2 CLIP probe - LENS-Net": ("CLIP probe", "LENS-Net"),
        "H3 LENS-Net - VideoMAE-B FT": ("LENS-Net", "VideoMAE-B FT")}
for tag, K in (("at095", keep95), ("at090", keep90)):
    BT = Boot({s: (D.sets[s]["y"][K[s]], np.asarray(D.sets[s]["cl"])[K[s]]) for s in UNSEEN}); cache = {}
    def H(m, s):
        if (m, s) not in cache: cache[(m, s)] = BT.hier(m, BT.aucs(s, [p[K[s]] for p in D.probs(m, s)]))
        return cache[(m, s)]
    ok = np.all([BT.S[s]["ok"] for s in UNSEEN], 0)
    for k, (a, c) in H1H3.items():
        out["claims"].setdefault(k, {})[tag] = summarize(np.mean([H(a, s) - H(c, s) for s in UNSEEN], 0), ok)
json.dump(out, open(L / "review/c7_filter090.json", "w"), indent=1)
for s in UNSEEN:
    r = out["removed"][s]; print(f"{s:10s} removed {r['at095']:4d} -> {r['at090']:4d} of {r['windows']:5d}  tau {out['tau'][s]:.3f}  "
          + "  ".join(f"{m[:10]} {out['auc'][s][m]['at095']:.2f}->{out['auc'][s][m]['at090']:.2f}" for m in ("LENS-Net", "LENS-Net-A", "CLIP probe", "VideoMAE-B FT")))
for m in models: print(f"{m:16s} mean9 {out['mean9'][m]['at095']:.2f} -> {out['mean9'][m]['at090']:.2f}")
for k, v in out["claims"].items():
    print(k, " | ".join(f"{t}: {v[t]['diff']:+.2f} [{v[t]['ci'][0]:+.2f}, {v[t]['ci'][1]:+.2f}] p={v[t]['p_two_sided']:.4f}" for t in v))
