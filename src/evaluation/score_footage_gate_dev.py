#!/usr/bin/env python3
"""LENS-FG development scoring (PREREG_journal Sec. 9; exploratory). Writes predictions of
  LENS-FG-post = unchanged LENS-Net checkpoints with the footage gate applied at inference only
to the window-level sets, and LENS-FG / LENS-FG-post predictions to the frame-level chunks (PREREG_journal Sec. 8 protocol).
Prints window-level AUC on the audited test split and the external pool for all gate variants."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import lensnet_setup as L, lensnet as V, framelevel_xd_ucf as FLM
R = L.ROOT / "workspace"


def model(cfg, run):
    m = V.V2(cfg, L.text_anchors()); m.load_state_dict(torch.load(L.OUT / run / "best.pt", weights_only=True)); m.eval(); return m


def runs(seed):          # name -> (config used at inference, checkpoint)
    return {f"v2_audited_LENS-FG_vrex_seed{seed}": ("LENS-FG", f"v2_audited_LENS-FG_vrex_seed{seed}"),
            f"v2_audited_LENS-FGpost_vrex_seed{seed}": ("LENS-FG", f"v2_audited_LENS-M_vrex_seed{seed}"),
            f"v2_std_std_rwf_LENS-FGpost_vrex_seed{seed}": ("LENS-FG", f"v2_std_std_rwf_LENS-M_vrex_seed{seed}")}


for setname, pdir in (("fresh", "fresh/preds"), ("conf", "conf/preds"), ("cfights", "cfights/preds")):
    cf = V.Compact(setname); rows = np.arange(len(cf.index))
    for s in (42, 43, 44):
        for name, (cfg, ck) in runs(s).items():
            if "FGpost" in name: pd.DataFrame({"p_violent": V.predict(model(cfg, ck), cf, rows)}).to_csv(R / pdir / f"{name}.csv", index=False)
for k in range(16):
    cf = V.Compact(f"fl{k}"); rows = np.arange(len(cf.index))
    for s in (42, 43, 44):
        for name, (cfg, ck) in runs(s).items():
            if name.startswith("v2_audited"): pd.DataFrame({"p_violent": V.predict(model(cfg, ck), cf, rows)}).to_csv(FLM.FL / "preds" / f"fl{k}" / f"{name}.csv", index=False)
# window-level audited test + external pool
df, _ = L.load_split("audited"); te = df[df.split == "test"].reset_index(drop=True); b = V.Compact("bench")
ext = pd.read_csv(L.ROOT / "data_tables/manifests/external_windows.csv")
pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True); e = V.Compact("external")
assert (e.index.path.values == pool.path.values).all()
out = {}
for lab, cfg, pat in (("LENS-Net", "LENS-M", "v2_audited_LENS-M_vrex_seed{}"), ("w/o motion", "V2b", "v2_audited_V2b_seed{}"), ("LENS-AG", "LENS-AG", "v2_audited_LENS-AG_vrex_seed{}"),
                      ("LENS-FG", "LENS-FG", "v2_audited_LENS-FG_vrex_seed{}"), ("LENS-FG-post", "LENS-FG", "v2_audited_LENS-M_vrex_seed{}")):
    v = []
    for s in (42, 43, 44):
        m = model(cfg, pat.format(s)); pe = V.predict(m, e, np.arange(len(pool)))
        v.append([100 * roc_auc_score(te.binary_label, V.predict(m, b, te.feat_row.values))] + [100 * roc_auc_score(pool.binary_label[pool.benchmark == bb], pe[(pool.benchmark == bb).values]) for bb in ("XD-test", "UCF-test")])
    out[lab] = np.mean(v, 0).round(2).tolist(); print(f"{lab:14s} test {out[lab][0]:.2f}  XD {out[lab][1]:.2f}  UCF {out[lab][2]:.2f}", flush=True)
json.dump(out, open(R / "logs/fg_dev_window.json", "w"), indent=2)
