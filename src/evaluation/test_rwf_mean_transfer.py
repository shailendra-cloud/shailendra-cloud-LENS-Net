#!/usr/bin/env python3
"""P3 significance of the mean-transfer difference (LENS-Net vs each fine-tuned baseline). Seed-averaged predictions on
the identical transfer windows of the six unseen corpora; each bootstrap resample draws clusters independently within
every corpus (near-duplicate components for source corpora, videos for XD/UCF) and recomputes the mean of six AUCs."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]; import lensnet_setup as L, lensnet as V
R = L.OUT.parent; ROOT = L.ROOT; auc = lambda y, p: 100 * roc_auc_score(y, p)
man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "dataset", "source_unit_id", "binary_label", "component_id"])
sp = pd.read_csv(R / "splits/std_rwf.csv", usecols=["segment_id", "split"]); m = man.merge(sp, on="segment_id", how="left")
tr = set(m[m.split == "train"].source_unit_id); e = pd.read_csv(ROOT / "data_tables/manifests/near_duplicate_edges.csv", usecols=["unit_a", "unit_b"])
linked = set(e[e.unit_a.isin(tr)].unit_b) | set(e[e.unit_b.isin(tr)].unit_a) | tr
tgt = man[m.split.isna().values & (man.dataset != "RWF2000").values & ~man.source_unit_id.isin(linked).values].sort_values("segment_id").reset_index(drop=True)
ext = pd.read_csv(ROOT / "data_tables/manifests/external_windows.csv")
pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True)
keep = (pool.excluded_precaution == 0).values
row_of = pd.Series(np.arange(len(man)), index=man.segment_id)
def lens_preds(cfg):
    fb, fe, pt, pe = V.Compact("bench"), V.Compact("external"), [], []
    for s in (42, 43, 44):
        mdl = V.V2(cfg.replace("_vrex", ""), L.text_anchors()); mdl.load_state_dict(torch.load(L.OUT / f"v2_std_std_rwf_{cfg}_seed{s}/best.pt")); mdl.eval()
        pt.append(V.predict(mdl, fb, row_of[tgt.segment_id].values)); pe.append(V.predict(mdl, fe, np.arange(len(pool))))
    return np.mean(pt, 0), np.mean(pe, 0)
def base_preds(name):
    pt, pe = [], []
    for d in sorted((R / "runs_baselines").glob(f"{name}_seed4*")):
        if not (d / "summary.json").exists(): continue
        t = pd.read_csv(d / "predictions_transfer.csv", usecols=["segment_id", "p_violent"]).set_index("segment_id").p_violent
        pt.append(t.loc[tgt.segment_id].values); pe.append(pd.read_csv(d / "predictions_external.csv").p_violent.values)
    return np.mean(pt, 0), np.mean(pe, 0), len(pt)
# six corpora: (labels, clusters, index into transfer or external arrays)
sets = []
for d in ["HockeyFights", "RLVS", "BusViolence", "VID"]:
    i = np.where(tgt.dataset == d)[0]; sets.append((d, "t", i, tgt.binary_label.values[i], tgt.component_id.values[i]))
for b in ["XD-test", "UCF-test"]:
    i = np.where(keep & (pool.benchmark == b).values)[0]; sets.append((b, "e", i, pool.binary_label.values[i], pool.video.values[i]))
Lt, Le = lens_preds("LENS-M_vrex")
res = {}
for name in ["A2_videomae", "A2_r2plus1d_18", "A2_swin3d_t"]:
    Bt, Be, ns = base_preds(name)
    def mean_diff(idx_sets):
        dl = []
        for (d, src, i, y, cl), ii in zip(sets, idx_sets):
            a, b = (Lt, Bt) if src == "t" else (Le, Be)
            yy = y[ii]; dl.append(auc(yy, a[i][ii]) - auc(yy, b[i][ii]))
        return float(np.mean(dl))
    point = mean_diff([np.arange(len(s[2])) for s in sets])
    rng = np.random.default_rng(0); groups = [pd.Series(np.arange(len(s[2]))).groupby(s[4]).apply(np.array).values for s in sets]
    boots = []
    for _ in range(2000):
        ii = [np.concatenate(g[rng.integers(0, len(g), len(g))]) for g in groups]
        if all(len(np.unique(s[3][x])) == 2 for s, x in zip(sets, ii)): boots.append(mean_diff(ii))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res[name] = dict(seeds=ns, diff=point, ci=[float(lo), float(hi)], excludes_zero=bool(lo > 0 or hi < 0), n_boot=len(boots))
    print(f"LENS-Net - {name} ({ns} seeds): mean-transfer diff {point:+.2f} [{lo:+.2f}, {hi:+.2f}]{'*' if lo > 0 or hi < 0 else ''}", flush=True)
json.dump(res, open(R / "logs/a2_mean_boot.json", "w"), indent=2)
