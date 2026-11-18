#!/usr/bin/env python3
"""F3 (PREREG Sec. 6): score audited-split LENS-Net runs on XD-test and UCF-test, with the same pool, exclusions and
metrics as video_data/evaluate_cnn_bigru.py and the threshold frozen from each run's audited validation set."""
import argparse, json, sys
from pathlib import Path

import numpy as np, pandas as pd, torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import lensnet_setup as L
from video_data import window_data as D

ROOT = L.ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directory names under workspace/runs (audited_*)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    feats = L.Tokens("external")
    ext = pd.read_csv(ROOT / "data_tables/manifests/external_windows.csv")
    if "excluded_precaution" not in ext: ext["excluded_precaution"] = 0
    pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True)
    idx = feats.index
    assert len(idx) == len(pool) and (idx.path.values == pool.path.values).all() and (idx.start_frame.values == pool.start_frame.values).all()
    pool["feat_row"] = np.arange(len(pool))
    keep = (pool.excluded_precaution == 0).values
    anchors = L.text_anchors()
    for name in a.runs:
        rd = L.OUT / name
        s = json.load(open(rd / "summary.json"))
        assert s["split"] == "audited", name
        model = L.LENS(s["config"], anchors).to(dev)
        if s["config"] != "ZS":
            model.load_state_dict(torch.load(rd / "best.pt", map_location=dev))
        pe_all = L.predict(model, feats, pool.feat_row.values, dev)
        thr = s["threshold_selected_on_val"]
        usable, pe = pool[keep].reset_index(drop=True), pe_all[keep]
        res = dict(note="evaluation only; threshold frozen from audited validation", threshold=thr)
        for b, g in usable.groupby("benchmark"):
            ii = g.index.values; gg = g.reset_index(drop=True); pp = pe[ii]
            res[b] = dict(overall=D.binary_metrics(gg.binary_label, pp, thr),
                          per_category=D.per_group(gg, pp, thr, "category"),
                          within_video_auc=D.within_group_auc(gg.binary_label, pp, gg.video))
            gp = pool[pool.benchmark == b]
            res[b]["overall_including_precautionary"] = D.binary_metrics(gp.binary_label, pe_all[gp.index.values], thr)
        pool.assign(p_violent=pe_all).drop(columns=["feat_row"]).to_csv(rd / "predictions_external.csv", index=False)
        json.dump(res, open(rd / "external.json", "w"), indent=2, default=float)
        print(name, {b: round(res[b]["overall"]["auc"], 2) for b in ("XD-test", "UCF-test") if b in res}, flush=True)


if __name__ == "__main__":
    main()
