"""C1 (PREREG_revision.md): random (R) and source-controlled (S) video-level splits of RWF-2000 and Hockey Fights,
seed 42, test 20% / val 10% of videos. Writes review/splits/C1_{corpus}_{R,S}.csv (segment_id, video_id, split)."""
import json, numpy as np, pandas as pd
from pathlib import Path
ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]); OUT = ROOT / "workspace/review/splits"
man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "video_id", "component_id", "dataset", "binary_label", "source_unit_id"])
E = pd.read_csv(ROOT / "data_tables/manifests/near_duplicate_edges.csv", usecols=["segment_a", "segment_b"])
seg2vid = dict(zip(man.segment_id, man.video_id))
rep = {}
for corpus in ("RWF2000", "HockeyFights"):
    m = man[man.dataset == corpus]
    vids = m.groupby("video_id").agg(y=("binary_label", "max"), comp=("component_id", "first")).reset_index()
    n = len(vids); n_te, n_va = round(0.2 * n), round(0.1 * n)
    rng = np.random.default_rng(42)
    # R: stratified random by video label
    lab = {}
    for y, g in vids.groupby("y"):
        v = g.video_id.values.copy(); rng.shuffle(v); k_te, k_va = round(0.2 * len(v)), round(0.1 * len(v))
        lab.update({x: "test" for x in v[:k_te]}); lab.update({x: "val" for x in v[k_te:k_te + k_va]}); lab.update({x: "train" for x in v[k_te + k_va:]})
    R = lab
    # S: whole components, shuffled; a component goes to test/val only if it fits the remaining room
    rng = np.random.default_rng(42); comps = vids.groupby("comp").video_id.apply(list); order = rng.permutation(len(comps)); S = {}
    room = {"test": n_te, "val": n_va}
    for i in order:
        vs = comps.iloc[i]; tgt = next((s for s in ("test", "val") if room[s] >= len(vs)), "train")
        if tgt != "train": room[tgt] -= len(vs)
        S.update({x: tgt for x in vs})
    for name, sp in (("R", R), ("S", S)):
        d = m[["segment_id", "video_id"]].assign(split=m.video_id.map(sp)); d.to_csv(OUT / f"C1_{corpus}_{name}.csv", index=False)
        tr = set(d[d.split == "train"].video_id); te = set(d[d.split == "test"].video_id)
        ed = E[E.segment_a.isin(set(m.segment_id)) & E.segment_b.isin(set(m.segment_id))]
        pairs = {(seg2vid[a], seg2vid[b]) for a, b in zip(ed.segment_a, ed.segment_b)}
        linked = {b for a, b in pairs if a in tr and b in te} | {a for a, b in pairs if b in tr and a in te}
        vs = vids.set_index("video_id")
        rep[f"{corpus}_{name}"] = dict(videos={s: int(sum(1 for v in sp.values() if v == s)) for s in ("train", "val", "test")},
                                      test_violent_share=round(float(vs.loc[list(te), "y"].mean()), 3), test_linked_pct=round(100 * len(linked) / len(te), 1))
json.dump(rep, open(OUT / "C1_splits_report.json", "w"), indent=1); print(json.dumps(rep, indent=1))
