#!/usr/bin/env python3
"""Same-source overlap under the public evaluation splits (Section 4, Fig. 3, Table 8c). CPU only, about 1 min.

    python src/audit/audit_public_splits.py            # writes results/public_split_audit.json

A: RWF-2000 official train/val split. B: RWF-2000 file-name prefix as a candidate source identifier.
C: random 80/20 file-level splits of RLVS, Hockey Fights and RWF-2000 (1,000 draws, rng 0): the share of test
videos with a verified same-source link to a training video (median 24.4 %, 48.2 % and 55.6 %).
All counts use the verified same-source edges (data_tables/manifests/near_duplicate_edges.csv), so they count
detected links only. Protocol, fixed before any count was computed: PROTOCOL_public_split_audit.md.

The matching of RWF-2000 videos to the official split (by original file name, through the file listing of the
Kaggle mirror in data_tables/audit/rwf2000_kaggle_files.json) is stored in data_tables/audit/
rwf_official_split_videos.csv; the intermediate file-name manifests it was built from are not redistributed.
"""
import collections, json, os
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
M = pd.read_csv(REPO / "data_tables/manifests/source_windows.csv")
E = pd.read_csv(REPO / "data_tables/manifests/near_duplicate_edges.csv")
vids = pd.read_csv(REPO / "data_tables/audit/rwf_official_split_videos.csv")
OUT = Path(os.environ.get("AUDIT_OUT") or REPO / "results"); OUT.mkdir(exist_ok=True)
res = {}

rwf = M[M.dataset == "RWF2000"]
res["A_matching"] = vids.official.value_counts().to_dict()
res["A_label_check"] = pd.crosstab(vids.orig_label.fillna("?"), vids.binary_label).to_dict()


def relatedness(seg_split, vid_split, restrict):
    """Detected relatedness between 'train' and the evaluation split under a window->split map."""
    e = E[E.segment_a.isin(restrict) & E.segment_b.isin(restrict)].copy()
    e["sa"] = e.segment_a.map(seg_split); e["sb"] = e.segment_b.map(seg_split)
    cross = e[e.sa != e.sb]
    tr_ev = cross[cross.sa.eq("train") | cross.sb.eq("train")]
    ev_segs = set(tr_ev.segment_a[tr_ev.sa != "train"]) | set(tr_ev.segment_b[tr_ev.sb != "train"])
    sub = M[M.segment_id.isin(restrict)]
    ev = sub[sub.segment_id.map(seg_split) != "train"]
    comp_span = int((sub.assign(s=sub.segment_id.map(seg_split)).groupby("component_id").s.nunique() > 1).sum())
    return dict(cross_split_pairs=int(len(cross)), cross_split_opposite_label=int((cross.label_a != cross.label_b).sum()),
                eval_windows=int(len(ev)), eval_windows_linked_to_train=int(ev.segment_id.isin(ev_segs).sum()),
                eval_videos=int(ev.video_id.nunique()),
                eval_videos_linked_to_train=int(ev[ev.segment_id.isin(ev_segs)].video_id.nunique()),
                components_spanning=comp_span)


# ---------------------------------------------------------------- A. official split
ok = vids[vids.official.isin(["train", "val"])]
seg_off = rwf[rwf.video_id.isin(ok.video_id)].set_index("segment_id").video_id.map(dict(zip(ok.video_id, ok.official)))
restrict = set(seg_off.index)
res["A_official"] = relatedness(seg_off.to_dict(), None, restrict)
seg_aud = M.set_index("segment_id").split
res["A_audited_same_videos"] = relatedness({s: ("train" if seg_aud[s] == "train" else "eval") for s in restrict}, None, restrict)
res["A_official_videos"] = ok.official.value_counts().to_dict()

# ---------------------------------------------------------------- B. file-name prefix
pv = vids.dropna(subset=["prefix"])
size = pv.groupby("prefix").size()
res["B_prefix"] = dict(videos=int(len(pv)), prefixes=int(len(size)),
                       videos_per_prefix=size.value_counts().sort_index().astype(int).to_dict(),
                       prefixes_multi=int((size > 1).sum()))
multi = pv[pv.prefix.isin(size[size > 1].index)]
res["B_prefix"]["span_official"] = int((multi[multi.official.isin(["train", "val"])].groupby("prefix").official.nunique() > 1).sum())
res["B_prefix"]["span_audited"] = int((multi.groupby("prefix").split.nunique() > 1).sum())
res["B_prefix"]["span_audited_videos"] = int(multi[multi.prefix.isin(
    multi.groupby("prefix").split.nunique().loc[lambda s: s > 1].index)].video_id.nunique())
res["B_prefix"]["span_audited_by_pair"] = collections.Counter(
    "/".join(sorted(g.split.unique())) for _, g in multi.groupby("prefix") if g.split.nunique() > 1)
# agreement between prefix and detected links (video pairs within RWF-2000)
v_of_seg = dict(zip(rwf.segment_id, rwf.video_id))
er = E[E.segment_a.isin(v_of_seg) & E.segment_b.isin(v_of_seg)]
linked = {tuple(sorted((v_of_seg[a], v_of_seg[b]))) for a, b in zip(er.segment_a, er.segment_b) if v_of_seg[a] != v_of_seg[b]}
pre = dict(zip(pv.video_id, pv.prefix))
same_prefix = set()
for _, g in multi.groupby("prefix"):
    v = sorted(g.video_id)
    same_prefix |= {(a, b) for i, a in enumerate(v) for b in v[i + 1:]}
comp = dict(zip(vids.video_id, vids.component_id))
res["B_agreement"] = dict(
    same_prefix_pairs=len(same_prefix),
    same_prefix_pairs_directly_linked=sum(p in linked for p in same_prefix),
    same_prefix_pairs_same_component=sum(comp[a] == comp[b] for a, b in same_prefix),
    linked_video_pairs=len(linked),
    linked_pairs_same_prefix=sum(pre.get(a) is not None and pre.get(a) == pre.get(b) for a, b in linked))

# ---------------------------------------------------------------- C. random 80/20 file-level splits
rng = np.random.default_rng(0)
res["C_random_80_20"] = {}
for ds in ("RLVS", "HockeyFights", "RWF2000"):
    d = M[M.dataset == ds]
    vl = d.drop_duplicates("video_id")[["video_id", "binary_label"]]
    restrict = set(d.segment_id)
    e = E[E.segment_a.isin(restrict) & E.segment_b.isin(restrict)]
    va, vb = e.segment_a.map(v_of := dict(zip(d.segment_id, d.video_id))).values, e.segment_b.map(v_of).values
    seg_v = d.video_id.values
    stats = collections.defaultdict(list)
    for _ in range(1000):
        test = set()
        for _, g in vl.groupby("binary_label"):
            ids = g.video_id.values
            test |= set(ids[rng.permutation(len(ids))[: int(round(0.2 * len(ids)))]])
        ta = np.fromiter((v in test for v in va), bool, len(va)); tb = np.fromiter((v in test for v in vb), bool, len(vb))
        cross = ta != tb
        linked_test = set(va[cross & ta]) | set(vb[cross & tb])
        stats["cross_pairs"].append(int(cross.sum()))
        stats["test_videos_linked_pct"].append(100 * len(linked_test) / len(test))
    res["C_random_80_20"][ds] = {k: dict(median=float(np.median(v)), p2_5=float(np.percentile(v, 2.5)),
                                         p97_5=float(np.percentile(v, 97.5))) for k, v in stats.items()}
    res["C_random_80_20"][ds]["videos"] = int(len(vl))

json.dump(res, open(OUT / "public_split_audit.json", "w"), indent=1, default=int)
print(f"Wrote {OUT / 'public_split_audit.json'}")
