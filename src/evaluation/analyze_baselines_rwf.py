#!/usr/bin/env python3
"""Protocol A2 (PREREG deviation 4): train on RWF-2000 official train; report official-test video accuracy / AUC and
transfer AUC on other corpora (windows not linked to std_rwf train) and XD/UCF. Baselines are read from runs_baselines;
LENS (V2b) and LP std_rwf runs are re-scored here on the identical transfer sets from their saved best.pt."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import lensnet_setup as L, lensnet as V
from video_data import window_data as D
ROOT = L.ROOT; R = L.OUT.parent
auc = lambda y, p: 100 * roc_auc_score(y, p)
man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "video_id", "dataset", "source_unit_id", "binary_label"])
sp = pd.read_csv(R / "splits/std_rwf.csv", usecols=["segment_id", "split"])
m = man.merge(sp, on="segment_id", how="left")
tr_units = set(m[m.split == "train"].source_unit_id)
e = pd.read_csv(ROOT / "data_tables/manifests/near_duplicate_edges.csv", usecols=["unit_a", "unit_b"])
linked = set(e[e.unit_a.isin(tr_units)].unit_b) | set(e[e.unit_b.isin(tr_units)].unit_a) | tr_units
TGT = m[m.split.isna() & (m.dataset != "RWF2000") & ~m.source_unit_id.isin(linked)].segment_id
ext = pd.read_csv(ROOT / "data_tables/manifests/external_windows.csv")
POOL = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True)


def summarize(val, test, trans, extp):
    yv, pv = L.video_level(val, val.p_violent.values); yt, pt = L.video_level(test, test.p_violent.values)
    thr = D.select_threshold(yv, pv); acc = D.binary_metrics(yt, pt, thr)["accuracy"]
    out = {"rwf_official_video_acc": acc, "rwf_official_video_auc": auc(yt, pt), "n_test_videos": len(yt)}
    trans = trans[trans.segment_id.isin(set(TGT))]
    for d, g in trans.groupby("dataset"): out[d] = auc(g.binary_label, g.p_violent)
    k = extp[extp.excluded_precaution == 0]
    for b, g in k.groupby("benchmark"): out[b] = auc(g.binary_label, g.p_violent)
    return out


res = {"transfer_windows": {d: int(n) for d, n in m[m.segment_id.isin(set(TGT))].dataset.value_counts().items()},
       "excluded_linked_windows": int((m.split.isna() & (m.dataset != "RWF2000") & m.source_unit_id.isin(linked)).sum())}
for d in sorted((R / "runs_baselines").glob("A2_*_seed*")):
    if not (d / "summary.json").exists(): continue
    f = lambda n: pd.read_csv(d / n)
    res[d.name] = summarize(f("predictions_val.csv"), f("predictions_test.csv"), f("predictions_transfer.csv"), f("predictions_external.csv"))

feats, ex = V.Compact("bench"), V.Compact("external")
row_of = pd.Series(np.arange(len(man)), index=man.segment_id)
tg = man[man.segment_id.isin(set(TGT))].copy()
for c in ["V2b", "LP", "LENS-M_vrex", "MP"]:
    for s in [42, 43, 44]:
        d = L.OUT / f"v2_std_std_rwf_{c}_seed{s}"
        model = V.V2(c.replace("_vrex", ""), L.text_anchors()); model.load_state_dict(torch.load(d / "best.pt")); model.eval()
        val, test = pd.read_csv(d / "predictions_val.csv"), pd.read_csv(d / "predictions_test.csv")
        trans = tg.assign(p_violent=V.predict(model, feats, row_of[tg.segment_id].values))
        extp = POOL.assign(p_violent=V.predict(model, ex, np.arange(len(POOL))))
        res[f"A2_LENS-{c}_seed{s}"] = summarize(val, test, trans, extp)
json.dump(res, open(R / "logs/A2_results.json", "w"), indent=2, default=float)
cols = ["rwf_official_video_acc", "rwf_official_video_auc", "HockeyFights", "RLVS", "BusViolence", "VID", "XD-test", "UCF-test"]
print("windows", res["transfer_windows"], "excluded linked", res["excluded_linked_windows"])
print(f"{'model':28s}" + "".join(f"{c[:10]:>11s}" for c in cols))
for k, v in res.items():
    if isinstance(v, dict) and "rwf_official_video_acc" in v:
        print(f"{k:28s}" + "".join(f"{v.get(c, float('nan')):11.2f}" for c in cols))
