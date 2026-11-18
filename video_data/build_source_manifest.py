#!/usr/bin/env python3
"""STEP 2: build the binary VIOLENCE / NON_VIOLENCE segment manifest.

Inputs are read-only: the v1 manifests, the v1 frame cache (memory-mapped, for perceptual hashes of
reused windows), the VID-Detect master manifest (for the BusViolence violent clips and the VID
classroom/domestic/other clips that v1 dropped), and the XD-test / UCF file listings (for leakage
exclusions). Nothing in v1 is modified.

Pipeline
  1. candidates  : v1 videos + VID-Detect videos v1 dropped only because they fit no 5-class label
  2. rules       : inclusion rules and exclusions from video_data/label_schema.py; every exclusion is written
                   to removed_rows.csv with a reason code
  3. probing     : existence, readability, frame count, fps; exact-duplicate removal by MD5
  4. windows     : v1 windows reused verbatim for v1 videos (cached frames stay reusable); new videos
                   segmented with the v1 rule (1.5 s, <= 8 windows)
  5. provenance  : source_unit_id per label_schema.source_unit_id
  6. phash       : 3-frame dHash per window through the cache's resize + JPEG pipeline
  7. near-dups   : source units linked by near-duplicate windows are merged into components, so
                   visually identical material can never be split across train/val/test
  8. split       : component-level; v1 split preserved where it is already a valid unit split
                   (keeps the v1 checkpoint's training videos out of the test split); otherwise
                   deterministic greedy assignment toward 70/15/15 per (dataset, label)
  9. weights     : sample_weight = equal mass per independent evidence unit within each label and
                   split; sample_weight_dataset_balanced adds equal mass per dataset first
No temporal labels are created and no quiet-negative windows are generated.
"""
import argparse, collections, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from video_data import label_schema as S
from video_data import video_io as IO

P = dict(
    v1_videos=ROOT / "raw_data/FINAL_DATASET/manifest_5class.csv",
    v1_clips=ROOT / "raw_data/FINAL_DATASET/manifest_5class_clips.csv",
    v1_dropped=ROOT / "raw_data/FINAL_DATASET/dropped_videos.json",
    viddetect=(__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[1]) / "data/manifest.csv"),
    peliculas=(__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[1]) / "data/ARCHIVE_PELICULAS/manifest_before_peliculas_removal.csv"),
    v1_cache=(__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[1]) / "raw_data/frame_cache/clip_T32_S256_N31279_v1.npz"),
    xd_test_dir=ROOT / "raw_data/xd_violence/data/video/test_videos",
    ucf_ann=ROOT / "raw_data/annotations/ucf_temporal_annotation.txt",
    ucf_roots=[ROOT / "raw_data/kagglehub_cache_journal", ROOT / "raw_data/kagglehub_cache"],
)
OUT = ROOT / "source_benchmark" / "manifests"


def log(*a): print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-xd-normals", action="store_true",
                    help="Admit XD-train normal videos (non-overlapping titles). Makes XD a single-class source.")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    t0 = time.time()
    (OUT / "splits").mkdir(parents=True, exist_ok=True)
    removed = []

    def remove(c, code, n_seg=0, detail="", reason=None):
        removed.append(dict(dataset=c["dataset"], video_id=f"{c['dataset']}:{Path(str(c['path'])).stem}",
                            path=str(c["path"]), original_filename=str(c.get("original_filename", "")),
                            original_label=str(c.get("original_label", "")), v1_class=c.get("v1_class", ""),
                            v1_split=c.get("v1_split", ""), n_segments_affected=int(n_seg), reason_code=code,
                            reason=reason if reason is not None else S.REASONS.get(code, ""), detail=detail))

    # ------------------------------------------------------------------ 1. candidates
    log("[1/9] loading candidates")
    V1 = pd.read_csv(P["v1_videos"], dtype=str).fillna("")
    C1 = pd.read_csv(P["v1_clips"], dtype=str).fillna("")
    C1["vkey"] = C1.parent_video_id.str.lstrip("0"); C1["v1_cache_row"] = np.arange(len(C1))
    V1["vkey"] = V1.video_id.str.lstrip("0")
    clips_by_video = {k: g for k, g in C1.groupby("vkey")}
    variant_by_video = C1.drop_duplicates("vkey").set_index("vkey").variant.to_dict()
    dropped = json.load(open(P["v1_dropped"]))
    log("      v1 dropped_videos.json reasons:", dict(collections.Counter(d.get("reason", "") for d in dropped)))
    VD = pd.read_csv(P["viddetect"], dtype=str).fillna("")
    new = VD[((VD.source_dataset == "BusViolence") & (VD["class"] == "public_transport")) |
             ((VD.source_dataset == "VID") & VD["class"].isin(["classroom", "domestic", "other"]))]
    xd_test_titles = {S.xd_title(f) for f in os.listdir(P["xd_test_dir"])}
    cands = []
    for r in V1.to_dict("records"):
        cands.append(dict(origin="v1", dataset=r["source_dataset"], path=r["final_path"],
                          original_filename=r["original_filename"], original_label=r["original_label"],
                          v1_class=r["final_class"], v1_split=r["split"], vkey=r["vkey"], content_md5=r["content_md5"],
                          variant=variant_by_video.get(r["vkey"], "blurred" if r["source_dataset"] == "VID" else "rgb")))
    for r in new.to_dict("records"):
        cands.append(dict(origin="viddetect_new", dataset=r["source_dataset"], path=r["path"],
                          original_filename=r["original_filename"], original_label=r["original_label"],
                          v1_class="", v1_split="", vkey="", content_md5="", variant=r["variant"]))
    log(f"      {len(V1)} v1 videos + {len(new)} VID-Detect videos v1 could not place "
        f"({dict(collections.Counter(zip(new.source_dataset, new['class'])))})")

    # ------------------------------------------------------------------ 2. rules
    log("[2/9] applying inclusion rules")
    included, xd_overlap_titles = [], set()
    for c in cands:
        n_v1 = len(clips_by_video[c["vkey"]]) if c["vkey"] in clips_by_video else 0
        ds, lab = c["dataset"], c["original_label"]
        if ds == "XDtrain":
            title = S.xd_title(c["original_filename"]); overlap = title in xd_test_titles
            if overlap: xd_overlap_titles.add(title)
            if lab == "A":
                if overlap: remove(c, "xd_test_title_overlap", n_v1, f"title={title}"); continue
                if args.include_xd_normals: c["rule"] = S.rule_for(ds, lab); included.append(c); continue
            code, reason = S.EXCLUSIONS.get((ds, lab), ("no_rule", None))
            remove(c, code, n_v1, (f"title={title}; also_xd_test_title_overlap" if overlap else f"title={title}"), reason)
            continue
        rule = S.rule_for(ds, lab)
        if rule is None: remove(c, "no_rule", n_v1); continue
        c["rule"] = rule; included.append(c)
    pel = pd.read_csv(P["peliculas"], dtype=str).fillna("")
    for r in pel[pel.source_dataset == "Movies"].to_dict("records"):
        code, reason = S.EXCLUSIONS[("Movies", "*")]
        remove(dict(dataset="Movies", path=r.get("original_path") or r.get("final_path", ""),
                    original_filename=r.get("original_filename", ""), original_label=r.get("original_label", "")),
               code, 0, reason=reason)
    ucf_ann = {l.split()[0] for l in open(P["ucf_ann"]) if l.strip()}
    ucf = {}
    for root in P["ucf_roots"]:
        for dp, _, fs in os.walk(root):
            for f in fs:
                if re.match(r"^[A-Za-z]+\d{3}_x264\.(mp4|avi)$", f): ucf.setdefault(f, os.path.join(dp, f))
    for f, p in sorted(ucf.items()):
        cat = re.match(r"^([A-Za-z]+?)\d", f).group(1)
        code = ("external_benchmark" if f in ucf_ann else "ignore_category" if cat in S.UCF_IGNORE else
                "no_temporal_annotation" if cat in S.UCF_VIOLENT else "single_class_source_pending_annotation_ucf")
        remove(dict(dataset="UCF", path=p, original_filename=f, original_label=cat), code, 0)
    log(f"      included candidates: {len(included)} | XD-train titles overlapping XD-test: {len(xd_overlap_titles)}")

    # ------------------------------------------------------------------ 3. probing + dedup
    log(f"[3/9] probing {len(included)} files ({args.workers} threads)")
    with ThreadPoolExecutor(args.workers) as ex:
        probes = list(ex.map(lambda c: IO.probe_video(c["path"]), included))
    ok = []
    for c, (exists, readable, n, fps) in zip(included, probes):
        n_v1 = len(clips_by_video[c["vkey"]]) if c["vkey"] in clips_by_video else 0
        if not exists: remove(c, "missing_file", n_v1); continue
        if not readable: remove(c, "unreadable", n_v1); continue
        c["total_frames"], c["probe_fps"] = n, fps; ok.append(c)
    need_md5 = [c for c in ok if not c["content_md5"]]
    log(f"      computing MD5 for {len(need_md5)} files")
    with ThreadPoolExecutor(args.workers) as ex:
        for c, h in zip(need_md5, ex.map(lambda c: IO.file_md5(c["path"]), need_md5)): c["content_md5"] = h
    ok.sort(key=lambda c: (c["origin"] != "v1", c["dataset"], c["path"]))
    seen, kept = {}, []
    for c in ok:
        h = c["content_md5"]
        if h in seen:
            first = seen[h]
            conflict = first["rule"]["binary"] != c["rule"]["binary"]
            remove(c, "duplicate_content", len(clips_by_video.get(c["vkey"], [])),
                   f"same MD5 as {first['dataset']}:{Path(first['path']).name}" + ("; LABEL CONFLICT" if conflict else ""))
            continue
        seen[h] = c; kept.append(c)

    # ------------------------------------------------------------------ 4-5. windows + provenance
    log("[4/9] building windows")
    segs = []
    for c in kept:
        rule = c["rule"]; ds = c["dataset"]
        unit, unit_rule = S.source_unit_id(ds, c["original_label"], c["original_filename"], c["path"])
        windows = []
        if c["origin"] == "v1":
            g = clips_by_video.get(c["vkey"])
            if g is None:
                span = int(round(S.WINDOW_SECONDS * (c["probe_fps"] or S.FPS_FALLBACK)))
                remove(c, "shorter_than_window", 0, f"frames={c['total_frames']} needed={span} (v1 segmentation)"); continue
            for r in g.to_dict("records"):
                windows.append((int(r["start_frame"]), int(r["end_frame"]), float(r["fps"]), int(r["clip_index"]),
                                r["clip_id"], int(r["v1_cache_row"])))
        else:
            fps = c["probe_fps"] or S.FPS_FALLBACK
            span = int(round(S.WINDOW_SECONDS * fps))
            if c["total_frames"] < span:
                remove(c, "shorter_than_window", 0, f"frames={c['total_frames']} needed={span}"); continue
            for j in range(min(c["total_frames"] // span, S.MAX_WINDOWS_PER_VIDEO)):
                windows.append((j * span, j * span + span - 1, round(fps, 3), j, "", -1))
        for s, e, fps, ci, cid, crow in windows:
            segs.append(dict(video_id=f"{ds}:{Path(c['path']).stem}", source_unit_id=unit, unit_rule=unit_rule, dataset=ds,
                             path=c["path"], original_filename=c["original_filename"], original_label=c["original_label"],
                             v1_class=c["v1_class"], v1_split=c["v1_split"], v1_clip_id=cid, v1_cache_row=crow,
                             clip_index=ci, start_frame=s, end_frame=e, n_frames=e - s + 1, fps=fps,
                             total_frames=c["total_frames"], duration_s=round((e - s + 1) / fps, 3),
                             binary_label=rule["binary"], violence_type=rule["violence_type"],
                             type_provisional=rule["type_provisional"], negative_kind=rule["negative_kind"],
                             setting=rule["setting"], production=rule["production"],
                             privacy_blur=int(c["variant"] == "blurred"), label_source="dataset_label_trimmed_video",
                             label_confidence=rule["label_confidence"], content_md5=c["content_md5"]))
    df = pd.DataFrame(segs).sort_values(["dataset", "path", "start_frame"]).reset_index(drop=True)
    df.insert(0, "segment_id", [f"v2seg_{i:07d}" for i in range(len(df))])
    log(f"      {len(df)} windows from {df.video_id.nunique()} videos, {df.source_unit_id.nunique()} source units")

    # ------------------------------------------------------------------ 6. perceptual hashes
    log(f"[5/9] perceptual hashes for {len(df)} windows")
    pos = S.PHASH_FRAME_POSITIONS
    cache = IO.open_v1_cache_mmap(P["v1_cache"])
    crow, path, st, en, tot = (df.v1_cache_row.values, df.path.values, df.start_frame.values,
                               df.end_frame.values, df.total_frames.values)
    def hsh(i):
        if crow[i] >= 0:
            return IO.records_from_cache(cache, int(crow[i]), pos)
        return IO.records_from_video(path[i], st[i], en[i], tot[i], pos, mode=S.NEW_VIDEO_DECODE_MODE)
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(hsh, range(len(df))))
    REC = IO.stack_records(res); del res
    df["phash"] = ["-".join(f"{int(h):016x}" for h in row) for row in REC["H_crop"]]
    df["phash_full"] = ["-".join(f"{int(h):016x}" for h in row) for row in REC["H_full"]]
    df["phash_valid_frames"] = (REC["SD_crop"] >= S.PHASH_MIN_GRAY_STD).sum(1)
    np.savez(OUT / "phash_arrays.npz", segment_id=df.segment_id.values, **REC)
    log(f"      windows with <2 usable content frames: {int((df.phash_valid_frames < 2).sum())}")

    # ------------------------------------------------------------------ 7. near-duplicate components
    log("[6/9] near-duplicate search across source units (all pairs)")
    unit_codes, unit_index = pd.factorize(df.source_unit_id)
    n_cand, pairs = IO.near_duplicates_verified(REC, unit_codes, S.PHASH_CAND_MAX_HAMMING, S.PHASH_CAND_MIN_FRAMES,
                                                S.PHASH_VERIFY_MIN_R, S.PHASH_MIN_MATCHED_FRAMES, log=log)
    log(f"      stage-1 candidates (<= {S.PHASH_CAND_MAX_HAMMING} bits, >= {S.PHASH_CAND_MIN_FRAMES} frame): {n_cand} "
        f"| stage-2 verified: {len(pairs)}")
    # stage 3 (post-rebuild audit fix): fine-structure verification, see label_schema.PHASH_FINE_MIN_R
    def fine(i):
        if crow[i] >= 0:
            return IO.fine_from_cache(cache, int(crow[i]), pos)
        return IO.fine_from_video(path[i], st[i], en[i], tot[i], pos, mode=S.NEW_VIDEO_DECODE_MODE)
    need = sorted({k for pr in pairs for k in pr})
    with ThreadPoolExecutor(args.workers) as ex:
        FINE = dict(zip(need, ex.map(fine, need)))
    fscore = IO.fine_scores(pairs, FINE.__getitem__); del FINE
    n_coarse = len(pairs); keep_f = fscore >= S.PHASH_FINE_MIN_R
    pd.DataFrame([dict(segment_a=df.segment_id[i], segment_b=df.segment_id[j], unit_a=df.source_unit_id[i],
                       unit_b=df.source_unit_id[j], dataset_a=df.dataset[i], dataset_b=df.dataset[j], grad128=float(f))
                  for (i, j), f, k in zip(pairs, fscore, keep_f) if not k]
                 ).to_csv(OUT / "near_duplicate_edges_rejected_fine.csv", index=False)
    n_rej_cross = sum(1 for (i, j), k in zip(pairs, keep_f) if not k and df.dataset[i] != df.dataset[j])
    pairs = [pr for pr, k in zip(pairs, keep_f) if k]; fscore = fscore[keep_f]
    log(f"      stage-3 fine structure (grad128 >= {S.PHASH_FINE_MIN_R}): kept {len(pairs)} of {n_coarse} "
        f"| rejected {n_coarse - len(pairs)} ({n_rej_cross} cross-dataset)")
    parent = list(range(len(unit_index)))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in pairs:
        a, b = find(unit_codes[i]), find(unit_codes[j])
        if a != b: parent[max(a, b)] = min(a, b)
    roots = np.array([find(u) for u in range(len(unit_index))])
    comp_of_unit = {unit_index[u]: f"C{r:05d}" for u, r in enumerate(roots)}
    df["component_id"] = df.source_unit_id.map(comp_of_unit)
    edges = pd.DataFrame([dict(segment_a=df.segment_id[i], segment_b=df.segment_id[j], unit_a=df.source_unit_id[i],
                               unit_b=df.source_unit_id[j], dataset_a=df.dataset[i], dataset_b=df.dataset[j],
                               label_a=int(df.binary_label[i]), label_b=int(df.binary_label[j]), grad128=float(f))
                          for (i, j), f in zip(pairs, fscore)])
    edges.to_csv(OUT / "near_duplicate_edges.csv", index=False)
    csize = df.groupby("component_id").source_unit_id.nunique()
    log(f"      near-duplicate window pairs across units: {len(pairs)} | components: {df.component_id.nunique()} "
        f"| merged components (>1 unit): {int((csize > 1).sum())} | largest: {int(csize.max())} units")

    # ------------------------------------------------------------------ 7b. external-benchmark contamination (decision 1)
    rev = pd.read_csv(ROOT / S.CONTAMINATION_REVIEW)
    conf = rev[rev.visual_verdict == "same_footage"]
    bad_videos = set(conf.video_id)
    bad_comps = sorted(set(df.loc[df.video_id.isin(bad_videos), "component_id"]))
    for cid in bad_comps:
        g = df[df.component_id == cid]
        if len(g) > S.CONTAMINATION_MAX_COMPONENT_WINDOWS or set(g.dataset) - {"RWF2000"}:
            raise RuntimeError(f"contaminated group {cid} has {len(g)} windows from {sorted(set(g.dataset))}; "
                               "refusing to remove it without a manual decision")
    rm_mask = df.component_id.isin(bad_comps)
    ext_of = conf.groupby("video_id").external_video.apply(lambda v: ";".join(sorted(set(v))))
    for vid, g in df[rm_mask].groupby("video_id"):
        r = g.iloc[0]
        why = (f"shares footage with {ext_of[vid]}" if vid in ext_of.index
               else "same near-duplicate group as a confirmed contaminated video")
        remove(dict(dataset=r.dataset, path=r.path, original_filename=r.original_filename,
                    original_label=r.original_label, v1_class=r.v1_class, v1_split=r.v1_split),
               "external_benchmark_contamination", len(g), f"{why}; group {r.component_id}")
    keep = ~rm_mask.values
    df = df[keep].reset_index(drop=True)
    REC = {k: v[keep] for k, v in REC.items()}
    kept_ids = set(df.segment_id)
    edges = edges[edges.segment_a.isin(kept_ids) & edges.segment_b.isin(kept_ids)]
    edges.to_csv(OUT / "near_duplicate_edges.csv", index=False)
    np.savez(OUT / "phash_arrays.npz", segment_id=df.segment_id.values, **REC)
    excl = conf.drop_duplicates(["benchmark", "external_video"])[["benchmark", "external_video", "external_label"]]
    excl = excl.assign(reason="shares footage with source material (visually confirmed); excluded from external evaluation")
    excl.to_csv(OUT / "external_eval_exclusions.csv", index=False)
    n_contam_windows, n_contam_groups = int(rm_mask.sum()), len(bad_comps)
    log(f"      contamination: {len(bad_videos)} confirmed videos -> removed {n_contam_groups} groups "
        f"({n_contam_windows} windows); {excl.external_video.nunique()} external videos excluded from evaluation")

    # ------------------------------------------------------------------ 8. split
    log("[7/9] component-level split")
    df["stratum"] = df.dataset + "|" + df.binary_label.astype(str)
    T = df.stratum.value_counts().to_dict()
    cur = {sp: collections.Counter() for sp in S.SPLIT_FRACTIONS}
    assign = {}
    groups = {cid: g for cid, g in df.groupby("component_id")}
    for cid, g in groups.items():
        if g.dataset.isin(S.TRAIN_ONLY_DATASETS).any():                      # decision 3
            assign[cid] = ("train", "train_only_dataset"); cur["train"].update(g.stratum.value_counts().to_dict())
            continue
        if (g.unit_rule == "video_provenance_limited").all() and (g.v1_split != "").all() and g.v1_split.nunique() == 1:
            sp = g.v1_split.iloc[0]; assign[cid] = (sp, "v1_preserved"); cur[sp].update(g.stratum.value_counts().to_dict())
    rest = sorted((cid for cid in groups if cid not in assign), key=lambda c: (-len(groups[c]), c))
    for cid in rest:
        g = groups[cid]; cnt = g.stratum.value_counts().to_dict()
        best = max(S.SPLIT_FRACTIONS, key=lambda sp: sum(n * (S.SPLIT_FRACTIONS[sp] * T[s] - cur[sp][s]) / T[s]
                                                         for s, n in cnt.items()))
        policy = "group_assigned_v1_overridden" if (g.v1_split != "").any() else "group_assigned"
        assign[cid] = (best, policy); cur[best].update(cnt)
    df["split"] = df.component_id.map(lambda c: assign[c][0])
    df["split_policy"] = df.component_id.map(lambda c: assign[c][1])
    for col in ("video_id", "source_unit_id", "component_id"):
        n_span = int((df.groupby(col).split.nunique() > 1).sum())
        if n_span: raise RuntimeError(f"{n_span} values of {col} span splits -- split assignment is wrong")

    # ------------------------------------------------------------------ 9. weights
    log("[8/9] sample weights")
    df["sample_weight"] = 0.0; df["sample_weight_dataset_balanced"] = 0.0
    for sp, g in df.groupby("split"):
        k1 = g.component_id + "|" + g.binary_label.astype(str)
        n1 = k1.map(k1.value_counts()); U1 = g.assign(k=k1).drop_duplicates("k").groupby("binary_label").size()
        w = 1.0 / (n1.values * g.binary_label.map(U1).values); df.loc[g.index, "sample_weight"] = w / w.mean()
        k2 = k1 + "|" + g.dataset
        n2 = k2.map(k2.value_counts())
        Ud = g.assign(k=k2).drop_duplicates("k").groupby(["binary_label", "dataset"]).size()
        D = g.drop_duplicates(["binary_label", "dataset"]).groupby("binary_label").size()
        ud = np.array([Ud[(l, d)] for l, d in zip(g.binary_label, g.dataset)])
        w2 = 1.0 / (n2.values * ud * g.binary_label.map(D).values)
        df.loc[g.index, "sample_weight_dataset_balanced"] = w2 / w2.mean()

    # ------------------------------------------------------------------ outputs
    log("[9/9] writing outputs")
    cols = ["segment_id", "video_id", "source_unit_id", "unit_rule", "component_id", "dataset", "split", "split_policy",
            "start_frame", "end_frame", "n_frames", "fps", "total_frames", "duration_s", "clip_index",
            "binary_label", "violence_type", "type_provisional", "negative_kind", "setting", "production",
            "privacy_blur", "label_source", "label_confidence", "original_label", "original_filename",
            "sample_weight", "sample_weight_dataset_balanced", "phash", "phash_full", "phash_valid_frames", "content_md5",
            "path", "v1_class", "v1_split", "v1_clip_id", "v1_cache_row"]
    df[cols].to_csv(OUT / "source_windows.csv", index=False)
    units = df.groupby("source_unit_id").agg(unit_rule=("unit_rule", "first"), dataset=("dataset", "first"),
                                             component_id=("component_id", "first"), split=("split", "first"),
                                             n_videos=("video_id", "nunique"), n_segments=("segment_id", "size"),
                                             labels=("binary_label", lambda s: "+".join(map(str, sorted(set(s))))))
    units.to_csv(OUT / "source_units.csv")
    comps = df.groupby("component_id").agg(split=("split", "first"), split_policy=("split_policy", "first"),
                                           n_units=("source_unit_id", "nunique"), n_segments=("segment_id", "size"),
                                           datasets=("dataset", lambda s: "+".join(sorted(set(s)))),
                                           labels=("binary_label", lambda s: "+".join(map(str, sorted(set(s))))))
    comps.to_csv(OUT / "splits" / "group_split.csv")
    rm = pd.DataFrame(removed); rm.to_csv(OUT / "removed_rows.csv", index=False)
    info = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), runtime_min=round((time.time() - t0) / 60, 1),
                args=vars(args), inputs={k: dict(path=str(v), size=os.path.getsize(v), mtime=os.path.getmtime(v))
                                         for k, v in P.items() if isinstance(v, Path) and v.is_file()},
                constants=dict(window_s=S.WINDOW_SECONDS, max_windows=S.MAX_WINDOWS_PER_VIDEO, phash_positions=pos,
                               phash_max_hamming=S.PHASH_MAX_HAMMING, phash_min_frames=S.PHASH_MIN_MATCHED_FRAMES,
                               phash_verify_min_r=S.PHASH_VERIFY_MIN_R, new_video_decode=S.NEW_VIDEO_DECODE_MODE,
                               phash_cand_max_hamming=S.PHASH_CAND_MAX_HAMMING, phash_cand_min_frames=S.PHASH_CAND_MIN_FRAMES,
                               phash_fine_min_r=S.PHASH_FINE_MIN_R,
                               train_only_datasets=S.TRAIN_ONLY_DATASETS,
                               split_fractions=S.SPLIT_FRACTIONS),
                counts=dict(segments=len(df), videos=int(df.video_id.nunique()), source_units=int(df.source_unit_id.nunique()),
                            components=int(df.component_id.nunique()), near_duplicate_candidates=n_cand,
                            contamination_removed_groups=n_contam_groups, contamination_removed_windows=n_contam_windows,
                            external_videos_excluded=int(excl.external_video.nunique()),
                            near_duplicate_pairs=len(pairs), near_duplicate_pairs_coarse=n_coarse,
                            xd_train_titles_overlapping_xd_test=len(xd_overlap_titles), removed_rows=len(rm)))
    json.dump(info, open(OUT / "build_info.json", "w"), indent=2)
    absent = [f"{d} absent from {sp}" for d in sorted(set(df.dataset) - set(S.TRAIN_ONLY_DATASETS)) for sp in S.SPLIT_FRACTIONS
              if not ((df.dataset == d) & (df.split == sp)).any()]
    if absent: log("      WARNING: " + "; ".join(absent))
    log(f"\nDONE in {info['runtime_min']} min")
    log(pd.crosstab(df.dataset, [df.split, df.binary_label], margins=True).to_string())
    log("\nremoved rows by reason:\n" + rm.groupby(["dataset", "reason_code"]).size().to_string())


if __name__ == "__main__":
    main()
