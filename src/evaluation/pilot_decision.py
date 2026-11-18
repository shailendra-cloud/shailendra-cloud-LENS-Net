#!/usr/bin/env python3
"""Stage P go/no-go (PREREG Sec. 5). Reads only validation-derived predictions."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2])
RUNS = ROOT / "workspace/runs"
base = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "split"])
frozen_val = set(base[base.split == "val"].segment_id)
out = {"rows": {}}
for d in ["HockeyFights", "RLVS", "RWF2000"]:
    m = pd.read_csv(ROOT / f"source_benchmark/runs/lodo_{d}/binary_hybrid_seed42/test_predictions.csv", usecols=["segment_id", "binary_label", "p_violent"])
    m = m[m.segment_id.isin(frozen_val)]
    row = {"n": len(m), "MAST-Net": 100 * roc_auc_score(m.binary_label, m.p_violent)}
    for c in ["LENS", "LP"]:
        p = pd.read_csv(RUNS / f"lodo_{d}_{c}_seed42_pilot/predictions_heldout_val_rows.csv", usecols=["segment_id", "binary_label", "p_violent"])
        assert set(p.segment_id) == set(m.segment_id), d
        row[c] = 100 * roc_auc_score(p.binary_label, p.p_violent)
    out["rows"][d] = row
mean = {k: float(np.mean([r[k] for r in out["rows"].values()])) for k in ["MAST-Net", "LENS", "LP"]}
aud = json.load(open(RUNS / "audited_LENS_seed42/summary.json"))["val"]["overall"]["auc"]
out.update(mean_heldout_val_auc=mean, audited_val_auc_LENS=aud,
           criterion_1=f"mean LENS {mean['LENS']:.2f} >= MAST-Net {mean['MAST-Net']:.2f} + 5",
           criterion_1_pass=bool(mean["LENS"] >= mean["MAST-Net"] + 5.0),
           criterion_2=f"audited val LENS {aud:.2f} >= 94.69 (MAST-Net 3-seed mean 95.69 - 1.0)",
           criterion_2_pass=bool(aud >= 94.69))
out["decision"] = "GO" if out["criterion_1_pass"] and out["criterion_2_pass"] else "NO-GO"
json.dump(out, open(RUNS.parent / "logs/pilot_decision.json", "w"), indent=2)
print(json.dumps(out, indent=2))
