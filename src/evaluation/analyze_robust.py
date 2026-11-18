#!/usr/bin/env python3
"""Robustness analysis (PREREG_journal Sec. 5; descriptive). Window ROC-AUC per model x condition x set (seed means);
mean over the 12 corruptions and drop from clean; paired LENS-Net minus VideoMAE-B FT per condition on seed-averaged
predictions (cluster bootstrap, 2000, rng 0). Writes logs/robust_results.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace"); ROOT = R.parent; P = R / "robust/preds"
CONDS = [f"{k}{i}" for k in ("blur", "dark", "noise", "jpeg") for i in (1, 2, 3)]
MODELS = {"LENS-Net": "LENS-M", "w/o motion": "V2b", "Motion probe": "MP", "VideoMAE-B FT": "videomae",
          "Swin-T FT": "swin3d_t", "R(2+1)D-18 FT": "r2plus1d_18", "CNN-BiGRU": "CNN-BiGRU"}
auc = lambda y, p: 100 * roc_auc_score(y, p)
man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv"); te = man[man.split == "test"].reset_index(drop=True)
w = pd.read_csv(R / "conf/conf_windows.csv"); w = w[(w.decode_failed == 0) & (w.benchmark == "VioPeru")].reset_index(drop=True)
SETS = {"test": (te.binary_label.values, te.component_id.values), "viop": (w.binary_label.values, w.video.values)}
res = {}
for s, (y, cl) in SETS.items():
    groups = pd.Series(np.arange(len(y))).groupby(cl).apply(np.array).values
    def preds(cond, key): return [pd.read_csv(P / s / cond / f"{key}_seed{sd}.csv").p_violent.values for sd in (42, 43, 44)]
    tab = {}
    for name, key in MODELS.items():
        row = {c: float(np.mean([auc(y, p) for p in preds(c, key)])) for c in ["clean"] + CONDS}
        row["mean_corrupted"] = float(np.mean([row[c] for c in CONDS])); row["drop"] = row["clean"] - row["mean_corrupted"]
        tab[name] = row
    paired = {}
    for c in ["clean"] + CONDS:
        a, b = np.mean(preds(c, "LENS-M"), 0), np.mean(preds(c, "videomae"), 0)
        rng = np.random.default_rng(0); ds = []
        for _ in range(2000):
            i = np.concatenate(groups[rng.integers(0, len(groups), len(groups))])
            if len(np.unique(y[i])) == 2: ds.append(auc(y[i], a[i]) - auc(y[i], b[i]))
        lo, hi = np.percentile(ds, [2.5, 97.5])
        paired[c] = dict(diff=float(auc(y, a) - auc(y, b)), ci=[float(lo), float(hi)], excludes_zero=bool(lo > 0 or hi < 0))
    res[s] = dict(auc=tab, paired_LENS_minus_VideoMAE=paired)
json.dump(res, open(R / "logs/robust_results.json", "w"), indent=2)
for s in SETS:
    print(f"=== {s}")
    print(f"{'model':15s} {'clean':>6s} " + " ".join(f"{c:>6s}" for c in CONDS) + "   mean   drop")
    for n, r in res[s]["auc"].items():
        print(f"{n:15s} {r['clean']:6.2f} " + " ".join(f"{r[c]:6.2f}" for c in CONDS) + f"  {r['mean_corrupted']:6.2f} {r['drop']:6.2f}")
    print("LENS - VideoMAE:", " ".join(f"{c}:{v['diff']:+.1f}{'*' if v['excludes_zero'] else ''}" for c, v in res[s]["paired_LENS_minus_VideoMAE"].items()))
