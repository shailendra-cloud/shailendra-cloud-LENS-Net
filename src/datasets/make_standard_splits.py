#!/usr/bin/env python3
"""Standard-protocol splits for the SOTA comparison (PREREG Sec. 6, F4). Writes workspace/splits/<name>.csv
(segment_id, video_id, split) plus a leak report: the share of test videos with a verified near-duplicate
link to a training video under each protocol.

  std_rwf          official RWF-2000 train/val (val -> 'test'); 10 % of official-train videos -> 'val' (stratified, seed 0)
  std_hockey_f{k}  stratified 5-fold video-level CV (seed 0); 10 % of the training fold -> 'val'
  std_rlvs_f{k}    same for RLVS
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2])
OUT = ROOT / "workspace/splits"; OUT.mkdir(parents=True, exist_ok=True)
seg = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv",
                  usecols=["segment_id", "video_id", "source_unit_id", "dataset", "binary_label"])
edges = pd.read_csv(ROOT / "data_tables/manifests/near_duplicate_edges.csv", usecols=["unit_a", "unit_b"])
unit_of_video = seg.groupby("video_id").source_unit_id.first()
nbr = {}
for a, b in zip(edges.unit_a, edges.unit_b):
    nbr.setdefault(a, set()).add(b); nbr.setdefault(b, set()).add(a)


def leak_rate(vsplit):
    """Share of test videos whose source unit has a verified edge to a train-video source unit."""
    tr_units = set(unit_of_video[[v for v, s in vsplit.items() if s == "train"]])
    te = [v for v, s in vsplit.items() if s == "test"]
    linked = sum(bool(nbr.get(unit_of_video[v], set()) & tr_units) for v in te)
    return linked, len(te)


def write(name, vsplit):
    g = seg[seg.video_id.isin(vsplit.keys())].copy()
    g["split"] = g.video_id.map(vsplit)
    g[["segment_id", "video_id", "split"]].to_csv(OUT / f"{name}.csv", index=False)
    l, n = leak_rate(vsplit)
    vl = pd.Series(vsplit)
    return dict(windows=len(g), videos={s: int((vl == s).sum()) for s in ("train", "val", "test")},
                test_videos_linked_to_train=l, test_videos=n, linked_pct=100 * l / max(n, 1))


report = {}
off = pd.read_csv(ROOT / "data_tables/splits/rwf_official_split_videos.csv")
off = off[off.official.isin(["train", "val"])]
lab = seg.groupby("video_id").binary_label.max()
tr_v = off[off.official == "train"].video_id.values
tr_v, va_v = train_test_split(tr_v, test_size=0.1, stratify=lab[tr_v].values, random_state=0)
vs = {**{v: "train" for v in tr_v}, **{v: "val" for v in va_v}, **{v: "test" for v in off[off.official == "val"].video_id}}
report["std_rwf"] = write("std_rwf", vs)
report["std_rwf"]["official_videos_missing_from_benchmark"] = 2000 - len(off)

for ds, short in [("HockeyFights", "hockey"), ("RLVS", "rlvs")]:
    vids = np.array(sorted(seg[seg.dataset == ds].video_id.unique()))
    y = lab[vids].values
    for k, (tri, tei) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(vids, y)):
        tr_v, va_v = train_test_split(vids[tri], test_size=0.1, stratify=y[tri], random_state=0)
        vs = {**{v: "train" for v in tr_v}, **{v: "val" for v in va_v}, **{v: "test" for v in vids[tei]}}
        report[f"std_{short}_f{k}"] = write(f"std_{short}_f{k}", vs)
json.dump(report, open(OUT / "standard_splits_report.json", "w"), indent=2)
for k, v in report.items():
    print(k, v["videos"], f"linked {v['test_videos_linked_to_train']}/{v['test_videos']} ({v['linked_pct']:.1f}%)")
