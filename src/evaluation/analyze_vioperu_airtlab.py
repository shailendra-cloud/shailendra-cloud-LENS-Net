#!/usr/bin/env python3
"""Pre-registered analysis on the journal confirmation sets (PREREG_journal Sec. 3). AUC per benchmark (seed means)
after near-duplicate exclusion; paired differences on seed-averaged predictions with a cluster bootstrap (2000, rng 0).
Clusters: VioPeru = video, AIRTLab = (label, clip number), shared by cam1/cam2. Flags: better = CI > 0,
non-inferior = lower bound > -1.0 (the pre-registered margin)."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace"); P = R / "conf/preds"
w = pd.read_csv(R / "conf/conf_windows.csv"); w = w[w.decode_failed == 0].reset_index(drop=True)
x = pd.read_csv(R / "conf/near_duplicate_exclusion.csv")
assert (x.path.values == w.path.values).all() and (x.start_frame.values == w.start_frame.values).all()
keep = x.excluded_near_duplicate.values == 0
B = ["VioPeru", "AIRTLab"]; auc = lambda y, p: 100 * roc_auc_score(y, p)
groups = {}
for f in sorted(P.glob("*.csv")):
    groups.setdefault(re.sub(r"_seed4[234]", "", f.stem), []).append(f)
res = {"n_windows": {b: int(((w.benchmark == b) & keep).sum()) for b in B},
       "n_clusters": {b: int(w[(w.benchmark == b) & keep].cluster.nunique()) for b in B}, "models": {}}
avg = {}
for g, fs in groups.items():
    ps = [pd.read_csv(f).p_violent.values for f in fs]; assert all(len(p) == len(w) for p in ps); avg[g] = np.mean(ps, 0)
    row = {"seeds": len(fs)}
    for b in B:
        m = keep & (w.benchmark == b).values; v = [auc(w.binary_label[m], p[m]) for p in ps]
        row[b] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
    res["models"][g] = row


def boot(a, b, bench):
    m = keep & (w.benchmark == bench).values
    y, pa, pb, c = w.binary_label.values[m], avg[a][m], avg[b][m], w.cluster.values[m]
    cl = pd.Series(np.arange(len(y))).groupby(c).apply(np.array).values
    rng = np.random.default_rng(0); ds = []
    for _ in range(2000):
        i = np.concatenate(cl[rng.integers(0, len(cl), len(cl))])
        if len(np.unique(y[i])) == 2: ds.append(auc(y[i], pa[i]) - auc(y[i], pb[i]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(diff=float(auc(y, pa) - auc(y, pb)), ci=[float(lo), float(hi)], better=bool(lo > 0), worse=bool(hi < 0),
                non_inferior=bool(lo > -1.0))


M, Ms = "v2_audited_LENS-M_vrex", "v2_std_std_rwf_LENS-M_vrex"
prim = {"H1 LENS-AG - LENS-M (P1)": ("v2_audited_LENS-AG_vrex", M), "H1 LENS-AG - LENS-M (P3)": ("v2_std_std_rwf_LENS-AG_vrex", Ms)}
for t, nm in (("f8_", "L1"), ("b16_", "L2"), ("b16_f8_", "L3")):
    prim[f"H2 {nm} - LENS-M (P1)"] = (t + M, M); prim[f"H2 {nm} - LENS-M (P3)"] = (t + Ms, Ms)
sec = {f"LENS-M - {g}": (M, g) for g in groups if g.startswith("A1_")}
sec.update({f"LENS-M (P3) - {g}": (Ms, g) for g in groups if g.startswith("A2_")})
sec.update({"LENS-M - w/o motion": (M, "v2_audited_V2b"), "LENS-M - CNN-BiGRU": (M, "MAST-Net")})
res["primary"] = {k: {b: boot(a, c, b) for b in B} for k, (a, c) in prim.items() if a in avg and c in avg}
res["secondary"] = {k: {b: boot(a, c, b) for b in B} for k, (a, c) in sec.items() if a in avg and c in avg}
json.dump(res, open(R / "logs/conf_results.json", "w"), indent=2)
fmt = lambda d: f"{d['diff']:+.2f} [{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}]{' better' if d['better'] else ''}{' WORSE' if d['worse'] else ''}{' NI' if d['non_inferior'] else ''}"
print("windows", res["n_windows"], "clusters", res["n_clusters"])
for g, v in res["models"].items(): print(f"{g:40s} seeds {v['seeds']}  " + "  ".join(f"{b} {v[b]['mean']:.2f}±{v[b]['sd']:.2f}" for b in B))
for sec_name in ("primary", "secondary"):
    print(f"--- {sec_name}")
    for k, v in res[sec_name].items(): print(f"{k:44s}", "  ".join(f"{b} {fmt(d)}" for b, d in v.items()))
