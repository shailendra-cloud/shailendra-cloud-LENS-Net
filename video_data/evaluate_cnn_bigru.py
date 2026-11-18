#!/usr/bin/env python3
"""STEP 3: clean binary evaluation of any checkpoint under the leakage-controlled protocol.

  --kind v1_5class   v1 checkpoint; P(VIOLENCE) = 1 - P(NON_VIOLENCE). Windows from videos the v1 model
                     trained on (v1_split == train) are excluded from its primary metrics.
  --kind cnn_bigru   CNN-BiGRU checkpoint.

The decision threshold is chosen on the VALIDATION split only (F1-maximising) and then frozen.
External windows (XD-test / UCF-test, 1.5 s, contamination-excluded) are scored with that frozen
threshold and are reported only; they never influence any choice.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from video_data import cnn_bigru_model as mm
from video_data import window_data as D
from video_data.frame_store import FrameStore

MAN = ROOT / "data_tables/manifests"


def load(kind, ckpt, dev):
    if kind == "v1_5class":
        model, info = mm.load_mastnet(ckpt, dev, strict=True)
        classes = info["classes"]
        if classes is None or not str(classes[0]).upper().startswith("NON"):
            raise RuntimeError(f"expected NON_VIOLENCE at index 0, got {classes}")
        return model, (lambda prob: 1.0 - prob[:, 0]), None
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    arch = ck["arch"]
    model = mm.MASTNet(backbone_kind=arch["backbone_kind"], pretrained=False, num_classes=2,
                       use_motion=arch["use_motion"], motion_pretrained=arch["motion_pretrained"],
                       use_interaction=arch["use_interaction"], whiten=arch["whiten"])
    model.load_state_dict(ck["model_state"], strict=True)
    return model.to(dev).eval(), (lambda prob: prob[:, 1]), ck


def fp_fn(df, p, thr, k=25):
    d = df.assign(p_violent=p, pred=(np.asarray(p) >= thr).astype(int))
    cols = [c for c in ("segment_id", "window_id", "video", "video_id", "dataset", "benchmark", "category", "setting",
                        "production", "negative_kind", "label_status", "binary_label", "p_violent") if c in d]
    fps = d[(d.binary_label == 0) & (d.pred == 1)]; fns = d[(d.binary_label == 1) & (d.pred == 0)]
    key = "dataset" if "dataset" in d else "benchmark"
    rates = {}
    for col in [c for c in (key, "negative_kind", "setting", "category", "label_status") if c in d]:
        neg = d[d.binary_label == 0].groupby(col); pos = d[d.binary_label == 1].groupby(col)
        rates[col] = {"false_positive_rate_pct": (neg.pred.mean() * 100).round(2).to_dict(),
                      "false_negative_rate_pct": ((1 - pos.pred.mean()) * 100).round(2).to_dict()}
    return dict(n_false_positives=int(len(fps)), n_false_negatives=int(len(fns)), rates=rates,
                top_false_positives=fps.sort_values("p_violent", ascending=False)[cols].head(k).to_dict("records"),
                top_false_negatives=fns.sort_values("p_violent")[cols].head(k).to_dict("records"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--kind", choices=["v1_5class", "cnn_bigru"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-external", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-per-split", type=int, default=0, help="smoke test only: sample this many windows per split")
    a = ap.parse_args()
    dev = torch.device("cuda")
    out = ROOT / "source_benchmark/eval" / a.name; out.mkdir(parents=True, exist_ok=True)
    model, pv_fn, _ = load(a.kind, a.checkpoint, dev)
    df = pd.read_csv(MAN / "source_windows.csv", dtype={"v1_split": str}).fillna({"v1_split": ""})
    store = FrameStore(stores=("supp",), use_v1=True)
    res = dict(checkpoint=a.checkpoint, kind=a.kind)

    def score(frame, st, decode="seek"):
        dl = DataLoader(D.WindowDataset(frame, st, False, decode=decode), batch_size=a.batch_size, shuffle=False,
                        num_workers=a.workers, pin_memory=True)
        return D.predict(model, dl, dev, tta=True, p_violent_fn=pv_fn)

    splits = {}
    for sp in ("val", "test"):
        f = df[df.split == sp].reset_index(drop=True)
        if a.max_per_split: f = f.sample(min(a.max_per_split, len(f)), random_state=0).reset_index(drop=True)
        f["seen_by_v1_training"] = (f.v1_split == "train").astype(int)
        splits[sp] = (f, score(f, store))
    def clean(f, p):
        if a.kind != "v1_5class": return f, p
        m = f.seen_by_v1_training.values == 0
        return f[m].reset_index(drop=True), p[m]
    fv, pv = clean(*splits["val"])
    thr = D.select_threshold(fv.binary_label, pv)
    res["threshold_selected_on_val"] = thr
    for sp, (f, p) in splits.items():
        fc, pc = clean(f, p)
        per = D.per_group(fc, pc, thr, "dataset")
        res[sp] = dict(overall=D.binary_metrics(fc.binary_label, pc, thr), per_dataset=per,
                       macro_dataset_auc=D.macro_auc(per)[0], worst_dataset_auc=D.macro_auc(per)[1],
                       within_group_auc=D.within_group_auc(fc.binary_label, pc, fc.component_id),
                       per_negative_kind=D.per_group(fc, pc, thr, "negative_kind"),
                       excluded_seen_by_v1=int(len(f) - len(fc)), fp_fn=fp_fn(fc, pc, thr))
        if a.kind == "v1_5class":
            res[sp]["all_windows_including_v1_seen"] = D.binary_metrics(f.binary_label, p, thr)
        f.assign(p_violent=p).to_csv(out / f"predictions_{sp}.csv", index=False)

    ext_csv = MAN / "external_windows.csv"
    if not a.no_external and ext_csv.exists():
        ext = pd.read_csv(ext_csv)
        if "excluded_precaution" not in ext: ext["excluded_precaution"] = 0
        pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True)
        if a.max_per_split: pool = pool.groupby("benchmark", group_keys=False).apply(
            lambda g: g.sample(min(a.max_per_split, len(g)), random_state=0)).reset_index(drop=True)
        pe_all = score(pool, FrameStore(stores=("external",), use_v1=False), decode="seq")
        keep = (pool.excluded_precaution == 0).values
        usable, pe = pool[keep].reset_index(drop=True), pe_all[keep]
        res["external"] = dict(note="evaluation only; threshold frozen from validation",
                               excluded_contamination_windows=int((ext.excluded_contamination == 1).sum()),
                               excluded_precaution_windows=int(((ext.excluded_precaution == 1) & (ext.excluded_contamination == 0)).sum()))
        for b, g in usable.groupby("benchmark"):
            idx = g.index.values; gg = g.reset_index(drop=True); pp = pe[idx]
            res["external"][b] = dict(overall=D.binary_metrics(gg.binary_label, pp, thr),
                                      per_category=D.per_group(gg, pp, thr, "category"),
                                      per_label_status=D.per_group(gg, pp, thr, "label_status"),
                                      within_video_auc=D.within_group_auc(gg.binary_label, pp, gg.video),
                                      fp_fn=fp_fn(gg, pp, thr))
            gp = pool[pool.benchmark == b]
            res["external"][b]["overall_including_precautionary"] = D.binary_metrics(gp.binary_label, pe_all[gp.index.values], thr)
        pool.assign(p_violent=pe_all).to_csv(out / "predictions_external.csv", index=False)
    json.dump(res, open(out / "metrics.json", "w"), indent=2, default=float)
    print(json.dumps({k: (v["overall"] if isinstance(v, dict) and "overall" in v else v)
                      for k, v in res.items() if k in ("val", "test", "threshold_selected_on_val")}, indent=1, default=float))
    if "external" in res:
        for b in ("XD-test", "UCF-test"):
            if b in res["external"]: print(b, json.dumps(res["external"][b]["overall"], default=float))


if __name__ == "__main__":
    main()
