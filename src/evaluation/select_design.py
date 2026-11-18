#!/usr/bin/env python3
"""Deviation 2 selection rule: higher mean held-out-val AUC over pilot-LODO; ties (<=0.10) broken by audited val AUC."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace/runs")
out = {}
for c in ["V2a", "V2b"]:
    a = {d: json.load(open(R / f"v2_lodo_{d}_{c}_seed42_pilot/summary.json"))["heldout_val_rows"]["overall"]["auc"] for d in ["HockeyFights", "RLVS", "RWF2000"]}
    out[c] = dict(heldout_val=a, mean=float(np.mean(list(a.values()))),
                  audited_val=json.load(open(R / f"v2_audited_{c}_seed42/summary.json"))["val"]["overall"]["auc"])
d = out["V2a"]["mean"] - out["V2b"]["mean"]
out["selected"] = ("V2a" if d > 0 else "V2b") if abs(d) > 0.10 else ("V2a" if out["V2a"]["audited_val"] >= out["V2b"]["audited_val"] else "V2b")
json.dump(out, open(R.parent / "logs/design_selection.json", "w"), indent=2); print(json.dumps(out, indent=2))
