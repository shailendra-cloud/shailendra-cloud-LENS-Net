#!/usr/bin/env python3
"""Part I (PREREG_revision.md): complementarity of the anchor (CLIP probe) and LENS-Net on the five untouched samples -
video-score scatter (I-1), pairwise disagreement and the share resolved by LENS-Net-A (I-2), Spearman correlation (I-3).
Descriptive. Uses the stored per-window probabilities of Parts G and H. Writes review/i_partI.json and the scatter data."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.argv = [sys.argv[0], "all"]
_gs = open(__import__("pathlib").Path(__file__).resolve().parents[2] / "src/revision/analyze_fusion_rules_untouched.py").read()
g = {"__file__": __file__}; exec(_gs.split("# ---- development sets")[0], g)
globals().update({k: v for k, v in g.items() if not k.startswith("__")})
exec("import confirm_anchor_fusion as CA, confirm_anchor_fusion_sample_b as CB\n" + _gs.split("import confirm_anchor_fusion as CA, confirm_anchor_fusion_sample_b as CB\n")[1].split('res["n_videos"]')[0], globals())
A, B = "CLIP probe", "LENS-Net"


def pairs(y, s):
    """Correctness of every (violent, non-violent) pair: 1 right, 0.5 tie, 0 wrong."""
    d = s[y == 1][:, None] - s[y == 0][None, :]
    return (d > 0) + 0.5 * (d == 0)


res = {"scatter": {}, "pairs": {}, "spearman": {}}
for sn in SAMPLES:
    ids, y, _ = META[sn]
    va, vb, vf, st = [], [], [], []
    for s in SEEDS:
        pa, pb = member(sn, A, s), member(sn, B, s)
        a = vscore(sn, pa, "max").loc[ids].values; b = vscore(sn, pb, "max").loc[ids].values
        f = vscore(sn, 0.5 * (pa + pb), "max").loc[ids].values
        va.append(a); vb.append(b); vf.append(f)
        ca, cb, cf = pairs(y, a), pairs(y, b), pairs(y, f)
        one = (ca == 1) & (cb == 0) | (ca == 0) & (cb == 1)
        st.append({"both": float(((ca == 1) & (cb == 1)).mean()), "anchor_only": float(((ca == 1) & (cb == 0)).mean()),
                   "lens_only": float(((ca == 0) & (cb == 1)).mean()), "neither": float(((ca == 0) & (cb == 0)).mean()),
                   "resolved": float(cf[one].mean()) if one.any() else float("nan")})
    res["pairs"][sn] = {k: float(np.mean([x[k] for x in st])) for k in st[0]}
    ma, mb = np.mean(va, 0), np.mean(vb, 0)
    res["spearman"][sn] = float(spearmanr(ma, mb).correlation)
    res["scatter"][sn] = {"anchor": ma.tolist(), "lens": mb.tolist(), "label": y.tolist()}
json.dump(res, open(L / "review/i_partI.json", "w"), indent=1)
for sn in SAMPLES:
    p = res["pairs"][sn]
    print(f"{sn:12s} n={len(META[sn][0]):4d} rho={res['spearman'][sn]:.2f} both {p['both']:.3f} anchor-only {p['anchor_only']:.3f} "
          f"LENS-only {p['lens_only']:.3f} neither {p['neither']:.3f} resolved {p['resolved']:.3f}")
