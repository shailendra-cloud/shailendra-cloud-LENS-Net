#!/usr/bin/env python3
"""Protocol A1 (pooled audited training) for the fine-tuned baselines: audited test AUC and XD/UCF AUC (evaluate_cnn_bigru
exclusions), per seed and mean/sd. Writes logs/A1_results.json."""
import json, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace"); auc = lambda y, p: 100 * roc_auc_score(y, p)
res = {}
for d in sorted((R / "runs_baselines").glob("A1_*_seed*")):
    if not (d / "summary.json").exists(): continue
    k = re.sub(r"_seed4[234]$", "", d.name)
    te = pd.read_csv(d / "predictions_test.csv"); ex = pd.read_csv(d / "predictions_external.csv"); ex = ex[ex.excluded_precaution == 0]
    v = dict(test=auc(te.binary_label, te.p_violent), xd=auc(*ex[ex.benchmark == "XD-test"][["binary_label", "p_violent"]].values.T),
             ucf=auc(*ex[ex.benchmark == "UCF-test"][["binary_label", "p_violent"]].values.T))
    res.setdefault(k, {"per_seed": {}})["per_seed"][d.name] = v
for k, r in res.items():
    for m in ["test", "xd", "ucf"]:
        x = [s[m] for s in r["per_seed"].values()]
        r[f"{m}_auc"] = dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)) if len(x) > 1 else 0.0, n=len(x))
json.dump(res, open(R / "logs/A1_results.json", "w"), indent=2)
for k, r in res.items(): print(k, len(r["per_seed"]), {m: round(r[f"{m}_auc"]["mean"], 2) for m in ["test", "xd", "ucf"]})

# ---- paired: LENS-Net (LENS-M) vs each fine-tuned baseline, seed-averaged predictions, cluster bootstrap (2000, rng 0)
ROOT = R.parent
man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "component_id"]).set_index("segment_id").component_id
KEY = ["path", "start_frame", "end_frame"]
def load_avg(dirs):
    te = [pd.read_csv(d / "predictions_test.csv").sort_values("segment_id").reset_index(drop=True) for d in dirs]
    ex = [pd.read_csv(d / "predictions_external.csv").sort_values(KEY).reset_index(drop=True) for d in dirs]
    t = te[0][["segment_id", "binary_label"]].assign(p=np.mean([x.p_violent.values for x in te], 0))
    e = ex[0][KEY + ["benchmark", "video", "binary_label", "excluded_precaution"]].assign(p=np.mean([x.p_violent.values for x in ex], 0))
    return t, e[e.excluded_precaution == 0].reset_index(drop=True)
def boot(y, a, b, cl):
    groups = pd.Series(np.arange(len(y))).groupby(np.asarray(cl)).apply(np.array).values
    rng = np.random.default_rng(0); ds = []
    for _ in range(2000):
        i = np.concatenate(groups[rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[i])) == 2: ds.append(auc(y[i], a[i]) - auc(y[i], b[i]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(diff=float(auc(y, a) - auc(y, b)), ci=[float(lo), float(hi)], excludes_zero=bool(lo > 0 or hi < 0))
lt, le = load_avg([R / "runs" / f"v2_audited_LENS-M_vrex_seed{s}" for s in (42, 43, 44)])
res["paired_vs_LENS-Net"] = {}
for k in list(res):
    if not k.startswith("A1_"): continue
    bt, be = load_avg([d for d in sorted((R / "runs_baselines").glob(f"{k}_seed4*")) if (d / "summary.json").exists()])
    bt = bt[bt.segment_id.isin(set(lt.segment_id))].reset_index(drop=True)
    assert (bt.segment_id.values == lt.segment_id.values).all() and (be.path.values == le.path.values).all()
    out = {"test": boot(lt.binary_label.values, lt.p.values, bt.p.values, lt.segment_id.map(man).values)}
    for bm, n in [("XD-test", "xd"), ("UCF-test", "ucf")]:
        m = (le.benchmark == bm).values
        out[n] = boot(le.binary_label.values[m], le.p.values[m], be.p.values[m], le.video.values[m])
    res["paired_vs_LENS-Net"][k] = out
    print("LENS-Net -", k, {m: f"{v['diff']:+.2f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]{'*' if v['excludes_zero'] else ''}" for m, v in out.items()})
json.dump(res, open(R / "logs/A1_results.json", "w"), indent=2)
