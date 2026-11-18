#!/usr/bin/env python3
"""Frame-level analysis (PREREG_journal Sec. 8). Each frame takes the score of the window that owns it (frames
own_from..end_frame); ground truth from the official annotations (frame f positive if a <= f <= b for an interval).
Because scores are constant within a window, frame-level AUC/AP are computed exactly from two weighted samples per
window (its positive and negative frame counts). A decode-failed window hands its frames to the previous window of the
same video (the count is reported). Paired differences: seed-averaged scores, video-cluster bootstrap (2000, rng 0).
Writes logs/framelevel_results.json."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import framelevel_xd_ucf as FLM
R = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace"); FL = R / "framelevel"
W = pd.read_csv(FL / "windows.csv"); nchunk = int(W.chunk.max()) + 1
ok = pd.concat([pd.read_csv(FL / f"chunks/fl{k}.csv") for k in range(nchunk)], ignore_index=True)
assert len(ok) == len(W) and (ok.path.values == W.path.values).all() and (ok.start_frame.values == W.start_frame.values).all()
W["decode_failed"] = ok.decode_failed.values
gt, cat = FLM.ground_truth()
# frames owned by each window, split into positive / negative counts
pos = np.zeros(len(W), np.int64); neg = np.zeros(len(W), np.int64)
for i, r in enumerate(W.itertuples()):
    lo, hi = r.own_from, min(r.end_frame, r.total_frames - 1)
    if hi < lo: continue
    n = hi - lo + 1; p = 0
    for a, b in gt.get((r.benchmark, r.video), []):
        p += max(0, min(hi, b) - max(lo, a) + 1)
    p = min(p, n); pos[i], neg[i] = p, n - p
# decode failures: give their frames to the previous window of the same video
own = np.arange(len(W))
for i in np.where(W.decode_failed.values == 1)[0]:
    j = i - 1
    while j >= 0 and W.video.values[j] == W.video.values[i] and W.decode_failed.values[j] == 1: j -= 1
    if j < 0 or W.video.values[j] != W.video.values[i]:
        j = i + 1
        while W.decode_failed.values[j] == 1: j += 1
    own[i] = j
okrow = np.where(W.decode_failed.values == 0)[0]; pos_ok = np.zeros(len(W), np.int64); neg_ok = np.zeros(len(W), np.int64)
np.add.at(pos_ok, own, pos); np.add.at(neg_ok, own, neg)
V = W.iloc[okrow].reset_index(drop=True); Vp, Vn = pos_ok[okrow], neg_ok[okrow]
# predictions: per chunk, concatenated in window order
groups = {}
for f in sorted((FL / "preds/fl0").glob("*.csv")): groups.setdefault(re.sub(r"_seed4[234]$", "", f.stem), []).append(f.name)
def load(name): return np.concatenate([pd.read_csv(FL / f"preds/fl{k}" / name).p_violent.values for k in range(nchunk)])
UCF_VIOL = set(("Normal",) + tuple(__import__("video_data.label_schema", fromlist=["x"]).UCF_VIOLENT))
subsets = {
    "UCF": (V.benchmark == "UCF").values, "XD": (V.benchmark == "XD").values,
    "UCF-violent": ((V.benchmark == "UCF") & V.category.isin(UCF_VIOL)).values,
    "UCF-clean": ((V.benchmark == "UCF") & (V.contaminated == 0)).values, "XD-clean": ((V.benchmark == "XD") & (V.contaminated == 0)).values}
def metrics(s, m):
    y = np.r_[np.ones(m.sum()), np.zeros(m.sum())]; x = np.r_[s[m], s[m]]; w = np.r_[Vp[m], Vn[m]]
    k = w > 0; y, x, w = y[k], x[k], w[k]
    return 100 * roc_auc_score(y, x, sample_weight=w), 100 * average_precision_score(y, x, sample_weight=w)
res = {"n_videos": {b: int(V[m].video.nunique()) for b, m in subsets.items()},
       "n_frames": {b: int(Vp[m].sum() + Vn[m].sum()) for b, m in subsets.items()},
       "positive_frame_share": {b: float(Vp[m].sum() / max(Vp[m].sum() + Vn[m].sum(), 1)) for b, m in subsets.items()},
       "decode_failed_windows": int(W.decode_failed.sum()), "plan": json.load(open(FL / "plan.json")), "models": {}}
avg = {}
for g, fs in groups.items():
    ps = [load(f) for f in fs]; assert all(len(p) == len(V) for p in ps); avg[g] = np.mean(ps, 0)
    row = {"seeds": len(fs)}
    for b, m in subsets.items():
        v = np.array([metrics(p, m) for p in ps])
        row[b] = {"auc": float(v[:, 0].mean()), "auc_sd": float(v[:, 0].std(ddof=1)) if len(v) > 1 else 0.0,
                  "ap": float(v[:, 1].mean()), "ap_sd": float(v[:, 1].std(ddof=1)) if len(v) > 1 else 0.0}
    res["models"][g] = row
def boot(a, b, bench, metric):
    m = subsets[bench]; idx = np.where(m)[0]; vids = V.cluster.values[idx]
    cl = pd.Series(idx).groupby(vids).apply(np.array).values; mi = 0 if metric == "auc" else 1
    rng = np.random.default_rng(0); ds = []
    for _ in range(2000):
        sel = np.concatenate(cl[rng.integers(0, len(cl), len(cl))]); mm = np.zeros(len(V), bool)
        cnt = np.bincount(sel, minlength=len(V)); mm = cnt > 0
        y = np.r_[np.ones(mm.sum()), np.zeros(mm.sum())]; w = np.r_[Vp[mm] * cnt[mm], Vn[mm] * cnt[mm]]
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0: continue
        f = (lambda s: roc_auc_score(y, np.r_[s[mm], s[mm]], sample_weight=w)) if mi == 0 else (lambda s: average_precision_score(y, np.r_[s[mm], s[mm]], sample_weight=w))
        ds.append(100 * (f(avg[a]) - f(avg[b])))
    lo, hi = np.percentile(ds, [2.5, 97.5]); d = metrics(avg[a], m)[mi] - metrics(avg[b], m)[mi]
    return dict(diff=float(d), ci=[float(lo), float(hi)], better=bool(lo > 0), worse=bool(hi < 0))
M = "v2_audited_LENS-M_vrex"
comps = ["A1_videomae", "A1_swin3d_t", "A1_r2plus1d_18", "A1_xclip_lr5e-06", "VadCLIP-L14_audited_lr0.0001", "MAST-Net", "ZS", "v2_audited_V2b", "b16_f8_v2_audited_LENS-M_vrex", "v2_audited_LENS-AG_vrex", "v2_audited_LENS-FG_vrex", "v2_audited_LENS-FGpost_vrex", "v2_audited_LP", "v2_audited_LENS-NetA"]
res["missing"] = [c for c in [M] + comps if c not in avg]
def paired(c):
    return {"UCF_auc": boot(M, c, "UCF", "auc"), "XD_ap": boot(M, c, "XD", "ap"), "XD_auc": boot(M, c, "XD", "auc"),
            "UCF-violent_auc": boot(M, c, "UCF-violent", "auc")}
import multiprocessing as mp
cl_ = [c for c in comps if c in avg and M in avg]
with mp.get_context("fork").Pool(len(cl_)) as pool:                    # same rng seed per comparison, so results do not depend on the pool
    res["paired"] = {f"LENS-M - {c}": r for c, r in zip(cl_, pool.map(paired, cl_))}
json.dump(res, open(R / "logs/framelevel_results.json", "w"), indent=2)
print("videos", res["n_videos"], "frames", res["n_frames"], "decode-failed windows", res["decode_failed_windows"])
if res["missing"]: print("MISSING", res["missing"])
for g, v in sorted(res["models"].items()):
    print(f"{g:40s} UCF AUC {v['UCF']['auc']:.2f}±{v['UCF']['auc_sd']:.2f}  XD AP {v['XD']['ap']:.2f}±{v['XD']['ap_sd']:.2f}  XD AUC {v['XD']['auc']:.2f}  UCF-viol AUC {v['UCF-violent']['auc']:.2f}  UCF-clean {v['UCF-clean']['auc']:.2f}  XD-clean AP {v['XD-clean']['ap']:.2f}")
fmt = lambda d: f"{d['diff']:+.2f} [{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}]{' better' if d['better'] else ''}{' WORSE' if d['worse'] else ''}"
for k, v in res["paired"].items(): print(f"{k:44s}", "  ".join(f"{n} {fmt(d)}" for n, d in v.items()))
