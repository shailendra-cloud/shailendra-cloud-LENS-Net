#!/usr/bin/env python3
"""Pre-registered analysis on CCTV-Fights (PREREG_journal Sec. 7). Window ROC-AUC (3-seed mean +- SD) after near-duplicate
exclusion on three sets: CCTV (primary), All, Testing (official testing subset). Paired differences on seed-averaged
predictions, video-cluster bootstrap (2000, rng 0): better = CI > 0, worse = CI < 0. Writes logs/cfights_results.json."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace"); P = R / "cfights/preds"
w = pd.read_csv(R / "cfights/cfights_windows.csv"); w = w[w.decode_failed == 0].reset_index(drop=True)
x = pd.read_csv(R / "cfights/near_duplicate_exclusion.csv")
assert (x.path.values == w.path.values).all() and (x.start_frame.values == w.start_frame.values).all()
keep = x.excluded_near_duplicate.values == 0
SETS = {"CCTV": (w.category == "CCTV").values, "All": np.ones(len(w), bool), "Testing": (w.subset == "testing").values}
auc = lambda y, p: 100 * roc_auc_score(y, p)
groups = {}
for f in sorted(P.glob("*.csv")): groups.setdefault(re.sub(r"_seed4[234]$", "", f.stem), []).append(f)
res = {"n_windows": {s: int((keep & m).sum()) for s, m in SETS.items()},
       "n_videos": {s: int(w[keep & m].video.nunique()) for s, m in SETS.items()},
       "n_excluded_near_duplicate": {s: int((~keep & m).sum()) for s, m in SETS.items()},
       "positives": {s: int(w.binary_label.values[keep & m].sum()) for s, m in SETS.items()}, "models": {}}
avg = {}
for g, fs in groups.items():
    ps = [pd.read_csv(f).p_violent.values for f in fs]; assert all(len(p) == len(w) for p in ps); avg[g] = np.mean(ps, 0)
    row = {"seeds": len(fs)}
    for s, m in SETS.items():
        v = [auc(w.binary_label.values[keep & m], p[keep & m]) for p in ps]
        row[s] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
    res["models"][g] = row


def boot(a, b, s):
    m = keep & SETS[s]
    y, pa, pb, c = w.binary_label.values[m], avg[a][m], avg[b][m], w.cluster.values[m]
    cl = pd.Series(np.arange(len(y))).groupby(c).apply(np.array).values
    rng = np.random.default_rng(0); ds = []
    for _ in range(2000):
        i = np.concatenate(cl[rng.integers(0, len(cl), len(cl))])
        if len(np.unique(y[i])) == 2: ds.append(auc(y[i], pa[i]) - auc(y[i], pb[i]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(diff=float(auc(y, pa) - auc(y, pb)), ci=[float(lo), float(hi)], better=bool(lo > 0), worse=bool(hi < 0))


M, Ms = "v2_audited_LENS-M_vrex", "v2_std_std_rwf_LENS-M_vrex"
prim = {f"LENS-M (P1) - {c}": (M, c) for c in ("A1_videomae", "A1_swin3d_t", "A1_r2plus1d_18", "A1_xclip_lr5e-06",
                                               "VadCLIP-L14_audited_lr0.0001")}
prim.update({f"LENS-M (P3) - {c}": (Ms, c) for c in ("A2_videomae", "A2_swin3d_t", "A2_r2plus1d_18", "VadCLIP-L14_std_std_rwf_lr0.0001")})
sec = {"LENS-AG - LENS-M (P1)": ("v2_audited_LENS-AG_vrex", M), "LENS-M - w/o motion (P1)": (M, "v2_audited_V2b"),
       "LENS-M - CNN-BiGRU (P1)": (M, "MAST-Net"), "LENS-M - CLIP zero-shot": (M, "ZS")}
for t, nm in (("f8_", "L1"), ("b16_", "L2"), ("b16_f8_", "L3")): sec[f"{nm} - LENS-M (P1)"] = (t + M, M)
res["missing"] = sorted({v for a, b in list(prim.values()) + list(sec.values()) for v in (a, b) if v not in avg})
res["primary"] = {k: {s: boot(a, b, s) for s in SETS} for k, (a, b) in prim.items() if a in avg and b in avg}
res["secondary"] = {k: {s: boot(a, b, s) for s in SETS} for k, (a, b) in sec.items() if a in avg and b in avg}
json.dump(res, open(R / "logs/cfights_results.json", "w"), indent=2)
fmt = lambda d: f"{d['diff']:+.2f} [{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}]{' better' if d['better'] else ''}{' WORSE' if d['worse'] else ''}"
print("windows", res["n_windows"], "videos", res["n_videos"], "near-dup excluded", res["n_excluded_near_duplicate"])
if res["missing"]: print("MISSING predictions:", res["missing"])
for g, v in sorted(res["models"].items()): print(f"{g:42s} seeds {v['seeds']}  " + "  ".join(f"{s} {v[s]['mean']:.2f}±{v[s]['sd']:.2f}" for s in SETS))
for k in ("primary", "secondary"):
    print(f"--- {k}")
    for n, v in res[k].items(): print(f"{n:44s}", "  ".join(f"{s} {fmt(d)}" for s, d in v.items()))
