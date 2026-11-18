#!/usr/bin/env python3
"""Confirmatory analysis on the fresh CCTV sets (PREREG deviation 6). AUC per benchmark after near-duplicate
exclusion; seed means; paired differences with a video-cluster bootstrap on seed-averaged predictions (2000, rng 0)."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace"); P = R / "fresh/preds"
w = pd.read_csv(R / "fresh/near_duplicate_exclusion.csv")
keep = w.excluded_near_duplicate.values == 0
auc = lambda y, p: 100 * roc_auc_score(y, p)
groups = {}
for f in sorted(P.glob("*.csv")):
    g = re.sub(r"_seed4[234]", "", f.stem); groups.setdefault(g, []).append(f)
res = {"n_windows": {b: int(((w.benchmark == b) & keep).sum()) for b in ["SCFD", "SCVD"]},
       "n_videos": {b: int(w[(w.benchmark == b) & keep].video.nunique()) for b in ["SCFD", "SCVD"]}, "models": {}}
avg = {}
for g, fs in groups.items():
    ps = [pd.read_csv(f).p_violent.values for f in fs]; avg[g] = np.mean(ps, 0)
    row = {"seeds": len(fs)}
    for b in ["SCFD", "SCVD"]:
        m = keep & (w.benchmark == b).values
        v = [auc(w.binary_label[m], p[m]) for p in ps]
        row[b] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
    res["models"][g] = row


def boot(a, b, bench):
    m = keep & (w.benchmark == bench).values
    y, pa, pb, v = w.binary_label.values[m], avg[a][m], avg[b][m], w.video.values[m]
    cl = pd.Series(np.arange(len(y))).groupby(v).apply(np.array).values
    rng = np.random.default_rng(0); ds = []
    for _ in range(2000):
        i = np.concatenate(cl[rng.integers(0, len(cl), len(cl))])
        if len(np.unique(y[i])) == 2: ds.append(auc(y[i], pa[i]) - auc(y[i], pb[i]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(diff=float(auc(y, pa) - auc(y, pb)), ci=[float(lo), float(hi)], excludes_zero=bool(lo > 0 or hi < 0))


pairs = [("v2_audited_LENS-M_vrex", "v2_audited_V2b"), ("v2_audited_LENS-M_vrex", "MAST-Net"), ("v2_audited_V2b", "MAST-Net")]
pairs += [("v2_audited_LENS-M_vrex", g) for g in groups if g.startswith("A1_")]
pairs += [("v2_std_std_rwf_LENS-M_vrex", g) for g in groups if g.startswith("A2_")] + [("v2_std_std_rwf_LENS-M_vrex", "v2_std_std_rwf_V2b")]
res["paired"] = {f"{a} - {b}": {bn: boot(a, b, bn) for bn in ["SCFD", "SCVD"]} for a, b in pairs if a in avg and b in avg}
json.dump(res, open(R / "logs/fresh_results.json", "w"), indent=2)
print("windows", res["n_windows"], "videos", res["n_videos"])
for g, v in res["models"].items(): print(f"{g:34s} seeds {v['seeds']}  SCFD {v['SCFD']['mean']:.2f}±{v['SCFD']['sd']:.2f}  SCVD {v['SCVD']['mean']:.2f}±{v['SCVD']['sd']:.2f}")
for k, v in res["paired"].items(): print(k, {b: f"{x['diff']:+.2f} [{x['ci'][0]:+.2f},{x['ci'][1]:+.2f}]{'*' if x['excludes_zero'] else ''}" for b, x in v.items()})
