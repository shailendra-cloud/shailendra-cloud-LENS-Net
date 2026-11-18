#!/usr/bin/env python3
"""Near-duplicate exclusion for the journal confirmation sets (PREREG_journal Sec. 3), identical to score_scfd_scvd.dedup:
a conf window is excluded if its mean CLIP CLS (ViT-L/14) cosine to any audited-train or std_rwf-train window is >= 0.95.
Labels are not read. Writes conf/near_duplicate_exclusion.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import lensnet_setup as L, lensnet as V
assert L.BACKBONE == "L14" and L.FRAMES == 16
ROOT = L.ROOT; CF = ROOT / "workspace/conf"
man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "split"])
sp = pd.read_csv(ROOT / "workspace/splits/std_rwf.csv", usecols=["segment_id", "split"])
train = set(man[man.split == "train"].segment_id) | set(sp[sp.split == "train"].segment_id)
b = V.Compact("bench"); rows = np.where(man.segment_id.isin(train).values)[0]
tb = F.normalize(b.cls[rows], dim=-1); c = V.Compact("conf"); tc = F.normalize(c.cls, dim=-1)
mx = torch.cat([(tc @ tb[i:i + 4096].t()).max(1).values.unsqueeze(1) for i in range(0, len(tb), 4096)], 1).max(1).values
w = pd.read_csv(CF / "conf_windows.csv"); w = w[w.decode_failed == 0].reset_index(drop=True)
assert (c.index.path.values == w.path.values).all() and (c.index.start_frame.values == w.start_frame.values).all()
w = w.drop(columns=["binary_label"]).assign(max_cos_to_train=mx.numpy(), excluded_near_duplicate=(mx.numpy() >= 0.95).astype(int))
w.to_csv(CF / "near_duplicate_exclusion.csv", index=False)
print("excluded:", w.groupby("benchmark").excluded_near_duplicate.sum().to_dict(), "of", w.groupby("benchmark").size().to_dict(),
      "| max cos quantiles 50/99/100%:", np.quantile(mx.numpy(), [0.5, 0.99, 1.0]).round(3))
