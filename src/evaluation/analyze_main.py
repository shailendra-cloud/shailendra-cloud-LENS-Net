#!/usr/bin/env python3
"""Stage F analysis. Every metric is recomputed from saved prediction CSVs with sklearn, independently of the
training code, and cross-checked against each run's summary.json. Differences use seed-averaged predictions
and a cluster bootstrap (2000 resamples, rng 0): components for benchmark rows, videos for XD/UCF.
Writes logs/stageF_results.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]); L = ROOT / "workspace"; R = L / "runs"
SEEDS = [42, 43, 44]; B = 2000
auc = lambda y, p: 100 * roc_auc_score(y, p)
man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "split", "component_id", "dataset"])
FVAL = set(man[man.split == "val"].segment_id)
res = {"crosscheck_max_abs_diff": 0.0}


def check(run, key, val):
    s = json.load(open(R / run / "summary.json"))
    ref = s[key]["overall"]["auc"] if key in s else None
    if ref is not None:
        res["crosscheck_max_abs_diff"] = max(res["crosscheck_max_abs_diff"], abs(ref - val))


def boot_diff(y, pa, pb, clusters):
    rng = np.random.default_rng(0)
    cl = pd.Series(np.arange(len(y))).groupby(np.asarray(clusters)).apply(np.array).values
    point = auc(y, pa) - auc(y, pb); ds = []
    for _ in range(B):
        idx = np.concatenate(cl[rng.integers(0, len(cl), len(cl))])
        if len(np.unique(y[idx])) < 2: continue
        ds.append(auc(y[idx], pa[idx]) - auc(y[idx], pb[idx]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(diff=point, ci=[float(lo), float(hi)], excludes_zero=bool(lo > 0 or hi < 0), n_boot=len(ds))


def ms(v): return dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, per_seed=[float(x) for x in v])


# ---------------- F1 + F3: audited test and external
def preds(run, f, key):
    d = pd.read_csv(R / run / f)
    return d.set_index(key)
EXT_KEY = ["path", "start_frame", "end_frame"]
def mast(seed, f):
    return pd.read_csv(ROOT / f"source_benchmark/eval/binary_hybrid_seed{seed}_cap8/{f}")

configs = {"MAST-Net": None, "ZS": ["audited_ZS_seed42"], "v1": [f"audited_LENS_seed{s}_v1" for s in SEEDS]}
for c, nm in [("LP", "LP"), ("V2a", "V2a"), ("LENS(V2b)", "V2b"), ("-G", "-G_vrex"), ("-T", "-T_vrex"), ("-Sp", "-Sp_vrex"), ("MP", "MP"), ("LENS-M", "LENS-M_vrex")]:
    configs[c] = [f"v2_audited_{nm}_seed{s}" for s in SEEDS]
tab, avg = {}, {}
for c, runs in configs.items():
    t, x, u, pt, pxd = [], [], [], [], []
    for i, s in enumerate(SEEDS if runs is None else range(len(runs))):
        if runs is None:
            te = mast(s, "predictions_test.csv"); ex = mast(s, "predictions_external.csv")
        else:
            rn = runs[i]
            if not (R / rn / "predictions_test.csv").exists(): break
            te = pd.read_csv(R / rn / "predictions_test.csv"); ex = pd.read_csv(R / rn / "predictions_external.csv")
            ex = ex.merge(pd.read_csv(ROOT / "data_tables/manifests/external_windows.csv", usecols=EXT_KEY + ["excluded_precaution"]),
                          on=EXT_KEY, how="left", suffixes=("", "_m")) if "excluded_precaution" not in ex else ex
        te = te.sort_values("segment_id").reset_index(drop=True)
        a = auc(te.binary_label, te.p_violent); t.append(a)
        if runs is not None: check(runs[i], "test", a)
        exu = ex[ex.excluded_precaution == 0].sort_values(EXT_KEY).reset_index(drop=True)
        x.append(auc(*exu[exu.benchmark == "XD-test"][["binary_label", "p_violent"]].values.T))
        u.append(auc(*exu[exu.benchmark == "UCF-test"][["binary_label", "p_violent"]].values.T))
        pt.append(te); pxd.append(exu)
    if not t: continue
    tab[c] = dict(test_auc=ms(t), xd_auc=ms(x), ucf_auc=ms(u))
    avg[c] = (pt[0].assign(p=np.mean([d.p_violent.values for d in pt], 0)), pxd[0].assign(p=np.mean([d.p_violent.values for d in pxd], 0)))
res["F1_F3_table"] = tab
res["F1_F3_paired"] = {}
for a_, b_ in [("LENS(V2b)", "MAST-Net"), ("LENS(V2b)", "LP"), ("LP", "MAST-Net"), ("LENS-M", "LENS(V2b)"), ("LENS-M", "MAST-Net"), ("LENS-M", "MP")]:
    if a_ not in avg or b_ not in avg: continue
    (ta, xa), (tb, xb) = avg[a_], avg[b_]
    assert (ta.segment_id.values == tb.segment_id.values).all()
    out = {"test": boot_diff(ta.binary_label.values, ta.p.values, tb.p.values, ta.segment_id.map(man.set_index("segment_id").component_id).values)}
    for bm in ["XD-test", "UCF-test"]:
        ma, mb = xa.benchmark == bm, xb.benchmark == bm
        assert (xa[ma].path.values == xb[mb].path.values).all()
        out[bm] = boot_diff(xa[ma].binary_label.values, xa[ma].p.values, xb[mb].p.values, xa[ma].video.values)
    res["F1_F3_paired"][f"{a_} - {b_}"] = out

# ---------------- F2: LODO (primary rows exclude D's frozen-val rows)
lodo = {}
for d in ["HockeyFights", "RLVS", "RWF2000", "BusViolence", "VID"]:
    m = pd.read_csv(ROOT / f"source_benchmark/runs/lodo_{d}/binary_hybrid_seed42/test_predictions.csv", usecols=["segment_id", "binary_label", "p_violent"])
    m = m[~m.segment_id.isin(FVAL)].sort_values("segment_id").reset_index(drop=True)
    row = {"n": len(m), "MAST-Net(seed42)": auc(m.binary_label, m.p_violent)}
    P = {}
    for c, nm in [("ZS", None), ("LP", "LP"), ("LENS(V2b)", "V2b"), ("MP", "MP"), ("LENS-M", "LENS-M_vrex")]:
        runs = [f"lodo_{d}_ZS_seed42"] if c == "ZS" else [f"v2_lodo_{d}_{nm}_seed{s}" for s in SEEDS]
        vals, ps = [], []
        for rn in runs:
            f = R / rn / "predictions_test.csv"
            if not f.exists(): continue
            p = pd.read_csv(f, usecols=["segment_id", "binary_label", "p_violent"])
            p = p[~p.segment_id.isin(FVAL)].sort_values("segment_id").reset_index(drop=True)
            assert (p.segment_id.values == m.segment_id.values).all(), (d, rn)
            vals.append(auc(p.binary_label, p.p_violent)); ps.append(p.p_violent.values)
        if vals: row[c] = ms(vals); P[c] = np.mean(ps, 0)
    comp = m.segment_id.map(man.set_index("segment_id").component_id).values
    for a_, b_ in [("LENS(V2b)", "MAST-Net"), ("LENS(V2b)", "LP"), ("LENS-M", "LENS(V2b)"), ("LENS-M", "MAST-Net")]:
        if a_ in P and (b_ in P or b_ == "MAST-Net"):
            row[f"{a_} - {b_}"] = boot_diff(m.binary_label.values, P[a_], m.p_violent.values if b_ == "MAST-Net" else P[b_], comp)
    lodo[d] = row
res["F2_LODO"] = lodo

# ---------------- F4: standard protocols (video-level accuracy and AUC)
rep = json.load(open(L / "splits/standard_splits_report.json")); std = {}
for proto, splits in [("RWF-2000 official", ["std_rwf"]), ("Hockey 5-fold", [f"std_hockey_f{k}" for k in range(5)]), ("RLVS 5-fold", [f"std_rlvs_f{k}" for k in range(5)])]:
    row = {"test_videos_linked_to_train_pct": float(np.mean([rep[s]["linked_pct"] for s in splits]))}
    for c in ["LP", "V2b", "MP", "LENS-M_vrex"]:
        acc, aucs = [], []
        for s in SEEDS:
            a1, a2 = [], []
            for sp in splits:
                f = R / f"v2_std_{sp}_{c}_seed{s}/summary.json"
                if not f.exists(): break
                tv = json.load(open(f))["test_video"]["overall"]; a1.append(tv["accuracy"]); a2.append(tv["auc"])
            if len(a1) == len(splits): acc.append(np.mean(a1)); aucs.append(np.mean(a2))
        if acc: row[c] = dict(video_accuracy=ms(acc), video_auc=ms(aucs))
    std[proto] = row
res["F4_standard"] = std
json.dump(res, open(L / "logs/stageF_results.json", "w"), indent=2, default=float)
print(json.dumps(res, indent=1, default=float)[:6000])
