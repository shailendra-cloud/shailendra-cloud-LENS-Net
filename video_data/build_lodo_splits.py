#!/usr/bin/env python3
"""Leave-one-dataset-out (LODO) manifests built from the frozen source manifest. Read-only on the frozen files.

For each held-out dataset D:
  train  = frozen TRAIN rows of the other datasets
  val    = frozen VAL rows of the other datasets        (model selection and threshold, as in the baseline)
  test   = ALL rows of D (train + val + test)           (D is never seen during training or selection)
Any train/val source unit with a VERIFIED near-duplicate link (the fine-gated edges the builder kept) to a D unit
is removed, so no footage shared with D can reach training or selection. The rule is direct links, not whole
components: components chain transitively, and one chain (two RLVS videos -> one VID recording group, visually
confirmed as false matches in Phase 4) would otherwise remove all of VID when RLVS is held out. Everything else
(windows, labels, weighting inputs) is unchanged.
Writes source_benchmark/lodo/<D>/source_windows.csv and source_benchmark/lodo/lodo_summary.csv.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data_tables/manifests/source_windows.csv"
OUT = ROOT / "source_benchmark/lodo"
DATASETS = ["BusViolence", "HockeyFights", "RLVS", "RWF2000", "VID"]


def main():
    df = pd.read_csv(BASE)
    edges = pd.read_csv(ROOT / "data_tables/manifests/near_duplicate_edges.csv", usecols=["unit_a", "unit_b"])
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in DATASETS:
        held = df[df.dataset == d].copy()
        comps = set(held.component_id)
        hu = set(held.source_unit_id)
        linked = (set(edges[edges.unit_a.isin(hu)].unit_b) | set(edges[edges.unit_b.isin(hu)].unit_a)) - hu
        others = df[(df.dataset != d)]
        leak = others.source_unit_id.isin(linked)
        comp_rule = int((others.component_id.isin(comps) & others.split.isin(["train", "val"])).sum())
        tr = others[(others.split == "train") & ~leak]
        va = others[(others.split == "val") & ~leak]
        te = held.assign(split="test")
        m = pd.concat([tr, va, te], ignore_index=True)[df.columns]
        assert not (set(tr.source_unit_id) | set(va.source_unit_id)) & linked, "linked unit left in train/val"
        assert not (set(tr.dataset) | set(va.dataset)) & {d}, "held-out dataset leaked into train/val"
        (OUT / d).mkdir(exist_ok=True)
        m.to_csv(OUT / d / "source_windows.csv", index=False)
        rows.append(dict(held_out=d, train=len(tr), val=len(va), test=len(te),
                         test_violent=int(te.binary_label.sum()), test_non_violent=int((te.binary_label == 0).sum()),
                         test_videos=te.video_id.nunique(), test_components=te.component_id.nunique(),
                         removed_for_shared_footage=int((leak & others.split.isin(["train", "val"])).sum()),
                         removed_units=len(linked & set(others.source_unit_id)),
                         would_remove_under_component_rule=comp_rule,
                         train_datasets="+".join(sorted(set(tr.dataset)))))
    S = pd.DataFrame(rows); S.to_csv(OUT / "lodo_summary.csv", index=False)
    print(S.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
