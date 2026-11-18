#!/usr/bin/env python3
"""STEP 2 audit of the binary source manifest. Exits 1 if any HARD check fails.

Every hard check is recomputed from the manifest and the files on disk rather than trusted from the
builder: files are re-probed, MD5s and perceptual hashes are re-derived for a random sample directly
from the video files (not the cache), and split leakage is recomputed from scratch.
Outputs go to data_tables/audit/.
"""
import argparse, io as pyio, json, os, re, signal, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from video_data import label_schema as S
from video_data import video_io as IO

MAN = ROOT / "data_tables/manifests"
OUT = ROOT / "data_tables/audit"
XD_TEST_DIR = ROOT / "raw_data/xd_violence/data/video/test_videos"
EXTERNAL = {"XD-test": ROOT / "raw_data/frame_cache_journal_xd/xd_violence_clips.npz",
            "UCF-test": ROOT / "raw_data/frame_cache_journal_ucf/ucf_crime_violence_clips.npz"}
REVIEWED_MATCHES = ROOT / "data_tables/audit/external_benchmark_matches_reviewed.csv"   # never redirected
UCF_TEST_ANN = ROOT / "raw_data/annotations/ucf_temporal_annotation.txt"
REQUIRED = ["segment_id", "video_id", "source_unit_id", "dataset", "split", "start_frame", "end_frame", "fps",
            "binary_label", "violence_type", "negative_kind", "setting", "production", "privacy_blur",
            "label_source", "label_confidence", "original_label", "sample_weight", "phash"]


from video_data.frame_store import FrameStore, content_key


class Audit:
    def __init__(self): self.rows = []
    def add(self, name, level, passed, value, detail=""):
        self.rows.append(dict(check=name, level=level, passed=bool(passed), value=value, detail=detail))
        tag = "PASS" if passed else ("FAIL" if level == "HARD" else level)
        print(f"  [{tag:<4}] {name}: {value}{(' | ' + detail) if detail else ''}", flush=True)


def section(t): print(f"\n=== {t} ===", flush=True)


def stream_external_records(npz, positions=(3, 8, 13), with_fine=False):
    """Frame records of a DEFLATE-compressed clip cache, streamed clip by clip (no full load)."""
    with zipfile.ZipFile(npz) as z:
        meta = json.loads(np.load(pyio.BytesIO(z.read("meta.npy")), allow_pickle=True).item())
        with z.open("X.npy") as fh:
            major, _ = np.lib.format.read_magic(fh)
            reader = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
            shape, _, dtype = reader(fh)
            per = int(np.prod(shape[1:])) * np.dtype(dtype).itemsize
            recs, fines = [], []
            for k in range(shape[0]):
                buf = bytearray()
                while len(buf) < per:
                    chunk = fh.read(per - len(buf))
                    if not chunk: raise RuntimeError("truncated external cache")
                    buf.extend(chunk)
                clip = np.frombuffer(bytes(buf), dtype=dtype).reshape(shape[1:])
                recs.append([IO.frame_record(cv2.cvtColor(np.ascontiguousarray(clip[p]), cv2.COLOR_RGB2BGR)) for p in positions])
                if with_fine:
                    fr = [IO.fine_frame(cv2.cvtColor(np.ascontiguousarray(clip[p]), cv2.COLOR_RGB2GRAY)) for p in positions]
                    fines.append((np.stack([f[0] for f in fr]), np.stack([f[1] for f in fr])))
    if with_fine:
        return IO.stack_records(recs), meta, (np.stack([f[0] for f in fines]), np.stack([f[1] for f in fines]))
    return IO.stack_records(recs), meta


def manifest_fine_getter(df):
    """Stage-3 inputs per manifest row: frame store (v1 cache / supp) when present, else decoded from video."""
    store = FrameStore(stores=("supp",), use_v1=True)
    def get(i):
        key = content_key(df.path[i], df.start_frame[i], df.end_frame[i], decode="seek")
        if store.has(int(df.v1_cache_row[i]), key):
            return IO.fine_from_jpegs(store.jpegs(int(df.v1_cache_row[i]), key), S.PHASH_FRAME_POSITIONS)
        return IO.fine_from_video(df.path[i], df.start_frame[i], df.end_frame[i], df.total_frames[i], S.PHASH_FRAME_POSITIONS,
                                  mode=S.NEW_VIDEO_DECODE_MODE)
    return get


def external_matches(df, REC, fine_m, log=print):
    """Coarse-verified manifest windows matching XD-test / UCF-test cached clips, each with its stage-3 grad128."""
    rows = []
    for name, npz in EXTERNAL.items():
        t1 = time.time()
        EREC, meta, (EG, EB) = stream_external_records(npz, with_fine=True)
        _, ver_e = IO.near_duplicates_verified(EREC, None, S.PHASH_CAND_MAX_HAMMING, S.PHASH_CAND_MIN_FRAMES,
                                               S.PHASH_VERIFY_MIN_R, S.PHASH_MIN_MATCHED_FRAMES, B=REC)
        fs = IO.fine_scores(ver_e, lambda i: (EG[i], EB[i]), fine_m)
        for (i, j), f in zip(ver_e, fs):
            rows.append(dict(benchmark=name, external_video=meta[i]["video"], external_label=meta[i]["label"],
                             segment_id=df.segment_id[j], dataset=df.dataset[j], split=df.split[j], path=df.path[j],
                             grad128=float(f)))
        log(f"  {name}: {len(ver_e)} coarse-verified window matches ({time.time()-t1:.0f}s)")
    return pd.DataFrame(rows, columns=["benchmark", "external_video", "external_label", "segment_id", "dataset",
                                       "split", "path", "grad128"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehash-sample", type=int, default=300)
    ap.add_argument("--md5-sample", type=int, default=150)
    ap.add_argument("--skip-external", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--manifest-dir", default=None, help="audit this manifest directory (default: the frozen one)")
    ap.add_argument("--out-dir", default=None, help="write reports here (default: data_tables/audit)")
    args = ap.parse_args()
    global MAN, OUT
    frozen_man = MAN
    if args.manifest_dir: MAN = Path(args.manifest_dir)
    if args.out_dir: OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    A = Audit(); t0 = time.time()
    df = pd.read_csv(MAN / "source_windows.csv", dtype={"v1_split": str, "v1_class": str, "v1_clip_id": str,
                                                          "phash": str, "phash_full": str}).fillna(
        {"v1_split": "", "v1_class": "", "v1_clip_id": ""})
    rm = pd.read_csv(MAN / "removed_rows.csv").fillna("")
    summary = {}

    section("schema")
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    A.add("required columns present", "HARD", not missing_cols, "all present" if not missing_cols else missing_cols)
    null_counts = {c: int(df[c].isna().sum()) for c in REQUIRED if c in df.columns and df[c].isna().any()}
    A.add("no missing values in required columns", "HARD", not null_counts, null_counts or 0)

    section("counts")
    counts = dict(segments=len(df), videos=int(df.video_id.nunique()), source_units=int(df.source_unit_id.nunique()),
                  components=int(df.component_id.nunique()), violence=int((df.binary_label == 1).sum()),
                  non_violence=int((df.binary_label == 0).sum()))
    for k, v in counts.items(): A.add(k, "INFO", True, v)
    by_ds = df.groupby(["dataset", "binary_label"]).agg(segments=("segment_id", "size"), videos=("video_id", "nunique"),
                                                         units=("source_unit_id", "nunique")).reset_index()
    by_ds.to_csv(OUT / "counts_by_dataset.csv", index=False); print(by_ds.to_string(index=False))
    by_split = df.groupby(["split", "binary_label"]).agg(segments=("segment_id", "size"), videos=("video_id", "nunique"),
                                                          units=("source_unit_id", "nunique")).reset_index()
    by_split.to_csv(OUT / "counts_by_split.csv", index=False); print(by_split.to_string(index=False))
    pd.crosstab([df.dataset, df.binary_label], df.split).to_csv(OUT / "counts_dataset_label_split.csv")
    per_unit = df.groupby("source_unit_id").agg(dataset=("dataset", "first"), label=("binary_label", "first"),
                                                split=("split", "first"), segments=("segment_id", "size"),
                                                videos=("video_id", "nunique")).sort_values("segments", ascending=False)
    per_unit["share_of_label_segments"] = per_unit.segments / per_unit.label.map(df.binary_label.value_counts())
    per_unit.to_csv(OUT / "counts_by_source_unit.csv")
    top = per_unit.share_of_label_segments.max()
    A.add("largest single source unit share of its label", "WARN", top <= 0.05, f"{100*top:.2f}%",
          f"{per_unit.index[0]} ({int(per_unit.segments.iloc[0])} segments)")
    summary["counts"] = counts

    section("leakage: split membership")
    for col in ("video_id", "source_unit_id", "component_id"):
        n = int((df.groupby(col).split.nunique() > 1).sum())
        A.add(f"{col} values spanning splits", "HARD", n == 0, n)
    xd_test_titles = {S.xd_title(f) for f in os.listdir(XD_TEST_DIR)}
    xd_rows = df[df.dataset == "XDtrain"]
    n_overlap = int(xd_rows.original_filename.map(lambda f: S.xd_title(f) in xd_test_titles).sum())
    A.add("XD-test title overlap with manifest", "HARD", n_overlap == 0, n_overlap, f"{len(xd_rows)} XD rows in manifest")
    n_xd_test_paths = int(df.path.str.contains("test_videos").sum())
    A.add("XD-test files in manifest", "HARD", n_xd_test_paths == 0, n_xd_test_paths)
    annotated = df[df.label_source == "temporal_annotation"]
    # Enriched mode is decided by WHICH manifest is audited, never by its contents, so a doctored manifest
    # cannot opt out of the enrichment checks by carrying no annotated rows.
    enriched = args.manifest_dir is not None and Path(args.manifest_dir).resolve() != frozen_man.resolve()
    A.add("audit mode", "INFO", True, "enriched" if enriched else "frozen")
    if not enriched and not annotated.empty:
        A.add("human-annotation rows in the frozen manifest", "HARD", False, len(annotated))
    if not enriched:
        # frozen manifest: unchanged rule
        A.add("UCF rows in manifest", "HARD", int((df.dataset == "UCF").sum()) == 0, int((df.dataset == "UCF").sum()))
    else:
        # enriched manifest: UCF-train windows from human playback annotation are allowed, train-only.
        # The protection the old rule gave (no UCF-test footage) is kept and tightened.
        ucf_test = {l.split()[0].replace(".mp4", "") for l in open(UCF_TEST_ANN) if l.strip()}
        ext_v = set(pd.read_csv(MAN / "external_windows.csv").video)
        ucf = df[df.dataset == "UCF"]
        n_ucf_test = int(ucf.path.map(lambda q: Path(q).stem in ucf_test or Path(q).name in ext_v).sum())
        A.add("UCF-test videos in manifest", "HARD", n_ucf_test == 0, n_ucf_test)
        n_ucf_unann = int((ucf.label_source != "temporal_annotation").sum())
        A.add("UCF rows that are not human temporal annotations", "HARD", n_ucf_unann == 0, n_ucf_unann)
        n_ann_out = int((annotated.split != "train").sum())
        A.add("annotated rows outside train (D1)", "HARD", n_ann_out == 0, n_ann_out, f"{len(annotated)} annotated rows")
        frozen_csv = frozen_man / "source_windows.csv"
        with open(frozen_csv, "rb") as f1, open(MAN / "source_windows.csv", "rb") as f2:
            fb = f1.read(); prefix_ok = f2.read(len(fb)) == fb
        A.add("enriched manifest starts with the frozen manifest bytes", "HARD", prefix_ok, prefix_ok)
        fdf = pd.read_csv(frozen_csv, dtype=str)
        edf = pd.read_csv(MAN / "source_windows.csv", dtype=str)
        vt = lambda d: d[d.split.isin(["val", "test"])].reset_index(drop=True)
        same_vt = vt(fdf).equals(vt(edf))
        A.add("validation/test identical to the frozen manifest", "HARD", same_vt, f"{len(vt(fdf))} rows")
        tail = edf.iloc[len(fdf):]
        app_ok = bool(len(tail) == 0 or ((tail.label_source == "temporal_annotation") & (tail.split == "train")).all())
        A.add("appended rows are train-only human temporal annotations", "HARD", app_ok,
              f"{len(tail)} appended rows",
              "" if app_ok else f"offending: {tail[~((tail.label_source == 'temporal_annotation') & (tail.split == 'train'))].segment_id.tolist()[:5]}")
        n_ann_not_tail = len(annotated) - len(tail[tail.label_source == "temporal_annotation"])
        A.add("annotated rows only in the appended block", "HARD", n_ann_not_tail == 0, n_ann_not_tail)
        tf = tail.assign(fps=tail.fps.astype(float), n_frames=tail.n_frames.astype(int),
                         start_frame=tail.start_frame.astype(int), end_frame=tail.end_frame.astype(int))
        span = (S.WINDOW_SECONDS * tf.fps).round().astype(int)
        bad_len = int(((tf.n_frames != span) | (tf.end_frame - tf.start_frame + 1 != tf.n_frames)).sum())
        A.add("appended windows are exactly 1.5 s (45 frames at 30 fps, inclusive end)", "HARD", bad_len == 0, bad_len)
    outside = int((df.dataset.isin(S.TRAIN_ONLY_DATASETS) & (df.split != "train")).sum())
    A.add("train-only datasets outside train", "HARD", outside == 0, outside, f"train-only: {S.TRAIN_ONLY_DATASETS}")
    rev0 = pd.read_csv(REVIEWED_MATCHES)
    still = int(df.video_id.isin(set(rev0[rev0.visual_verdict == "same_footage"].video_id)).sum())
    A.add("confirmed external-contamination windows still in manifest", "HARD", still == 0, still)
    excl_ok = (MAN / "external_eval_exclusions.csv").exists()
    A.add("external evaluation exclusion list written", "HARD", excl_ok, excl_ok)

    section("leakage: perceptual hash across splits (recomputed)")
    arr = np.load(MAN / "phash_arrays.npz", allow_pickle=True)
    aligned = list(arr["segment_id"]) == list(df.segment_id)
    A.add("phash arrays aligned with manifest", "HARD", aligned, aligned)
    REC = {k: arr[k] for k in ("H_full", "H_crop", "SD_full", "SD_crop", "G", "R", "C")}
    stored = ["-".join(f"{int(h):016x}" for h in row) for row in REC["H_crop"]]
    A.add("manifest phash column equals stored arrays", "HARD", stored == list(df.phash), "checked")
    units = pd.factorize(df.source_unit_id)[0]; splits = pd.factorize(df.split)[0]
    n_cand, cross = IO.near_duplicates_verified(REC, units, S.PHASH_CAND_MAX_HAMMING, S.PHASH_CAND_MIN_FRAMES,
                                                S.PHASH_VERIFY_MIN_R, S.PHASH_MIN_MATCHED_FRAMES, only_cross=splits)
    A.add("cross-split candidates before verification", "INFO", True, n_cand)
    fine_m = manifest_fine_getter(df)
    cross_f = IO.fine_scores(cross, fine_m)
    pd.DataFrame([dict(segment_a=df.segment_id[i], segment_b=df.segment_id[j], split_a=df.split[i], split_b=df.split[j],
                       unit_a=df.source_unit_id[i], unit_b=df.source_unit_id[j], grad128=float(f))
                  for (i, j), f in zip(cross, cross_f)],
                 columns=["segment_a", "segment_b", "split_a", "split_b", "unit_a", "unit_b", "grad128"]
                 ).to_csv(OUT / "phash_cross_split_matches.csv", index=False)
    n_gated = int((cross_f >= S.PHASH_FINE_MIN_R).sum())
    A.add("cross-split coarse-verified pairs rejected by stage 3 (not leaks)", "INFO", True, len(cross) - n_gated)
    A.add("cross-split near-duplicate window pairs", "HARD", n_gated == 0, n_gated,
          f"candidates: whole-frame or content dHash <= {S.PHASH_CAND_MAX_HAMMING} bits on >= {S.PHASH_CAND_MIN_FRAMES} frame; "
          f"verified: >= {S.PHASH_MIN_MATCHED_FRAMES} frames each side with content-intersection r >= {S.PHASH_VERIFY_MIN_R} "
          f"and 128px gradient r >= {S.PHASH_FINE_MIN_R}")
    low = int((df.phash_valid_frames < 2).sum())
    A.add("windows with < 2 usable content frames", "WARN", low == 0, low, "cannot be matched by the content hash")

    section("re-derived perceptual hashes (sample, from video files)")
    rng = np.random.default_rng(7)
    samp = rng.choice(len(df), min(args.rehash_sample, len(df)), replace=False)
    def rehash(i):
        r = df.iloc[int(i)]
        mode = "seek" if r.v1_cache_row >= 0 else S.NEW_VIDEO_DECODE_MODE
        got = IO.records_from_video(r.path, r.start_frame, r.end_frame, r.total_frames, S.PHASH_FRAME_POSITIONS, mode=mode)
        return (r.phash == "-".join(f"{x[1]:016x}" for x in got)) and (r.phash_full == "-".join(f"{x[0]:016x}" for x in got))
    with ThreadPoolExecutor(args.workers) as ex: same = list(ex.map(rehash, samp))
    A.add("stored phash equals re-derived phash", "HARD", all(same), f"{sum(same)}/{len(same)}")

    section("files")
    vids = df.drop_duplicates("video_id")[["video_id", "path", "total_frames", "content_md5"]].reset_index(drop=True)
    with ThreadPoolExecutor(args.workers) as ex: pr = list(ex.map(lambda p: IO.probe_video(p), vids.path))
    vids["exists"] = [p[0] for p in pr]; vids["readable"] = [p[1] for p in pr]; vids["probe_frames"] = [p[2] for p in pr]
    A.add("missing files", "HARD", int((~vids.exists).sum()) == 0, int((~vids.exists).sum()))
    A.add("unreadable files", "HARD", int((vids.exists & ~vids.readable).sum()) == 0, int((vids.exists & ~vids.readable).sum()))
    dup_md5 = int(vids.content_md5.duplicated().sum())
    A.add("duplicate file contents (MD5) among videos", "HARD", dup_md5 == 0, dup_md5)
    ms = vids.sample(min(args.md5_sample, len(vids)), random_state=7)
    with ThreadPoolExecutor(args.workers) as ex: md5s = list(ex.map(IO.file_md5, ms.path))
    A.add("stored MD5 equals re-hashed MD5 (sample)", "HARD", list(ms.content_md5) == md5s,
          f"{sum(a == b for a, b in zip(ms.content_md5, md5s))}/{len(ms)}")

    section("frame ranges and durations")
    fr = df.merge(vids[["video_id", "probe_frames"]], on="video_id")
    bad_range = int(((fr.start_frame < 0) | (fr.end_frame <= fr.start_frame) | (fr.end_frame >= fr.probe_frames)).sum())
    A.add("invalid frame ranges (vs re-probed frame count)", "HARD", bad_range == 0, bad_range)
    tol = 1.0 / df.fps
    short = int((df.n_frames / df.fps < S.WINDOW_SECONDS - tol).sum())
    long_ = int((df.n_frames / df.fps > S.WINDOW_SECONDS + tol).sum())
    A.add("windows shorter than 1.5 s (beyond 1-frame tolerance)", "HARD", short == 0, short)
    A.add("windows longer than 1.5 s (beyond 1-frame tolerance)", "HARD", long_ == 0, long_)
    per_vid = int(df.groupby("video_id").size().max())
    A.add("max windows per video", "HARD", per_vid <= S.MAX_WINDOWS_PER_VIDEO, per_vid)

    section("labels")
    bad = {
        "binary_label": int((~df.binary_label.isin([0, 1])).sum()),
        "violence_type": int((~df.violence_type.isin(list(S.VIOLENCE_TYPES) + [S.TYPE_NONE])).sum()),
        "type/binary consistency": int(((df.binary_label == 1) & (df.violence_type == S.TYPE_NONE)).sum()
                                       + ((df.binary_label == 0) & (df.violence_type != S.TYPE_NONE)).sum()),
        "negative_kind": int((~df.negative_kind.isin(S.NEGATIVE_KINDS)).sum()),
        "negative_kind/binary consistency": int(((df.binary_label == 1) & (df.negative_kind != "none")).sum()
                                                + ((df.binary_label == 0) & (df.negative_kind == "none")).sum()),
        "setting": int((~df.setting.isin(S.SETTINGS)).sum()),
        "production": int((~df.production.isin(S.PRODUCTIONS)).sum()),
        "label_source": int((~df.label_source.isin(S.LABEL_SOURCES)).sum()),
        "label_confidence": int((~df.label_confidence.isin(S.LABEL_CONFIDENCE)).sum()),
        "split": int((~df.split.isin(list(S.SPLIT_FRACTIONS))).sum()),
        "UNKNOWN used as a label": int((df.violence_type == "UNKNOWN").sum()),
    }
    for k, v in bad.items(): A.add(f"invalid {k}", "HARD", v == 0, v)
    A.add("violence types are provisional (dataset-derived, not annotated)", "INFO", True,
          int(df.type_provisional.astype(str).str.lower().eq("true").sum()))

    section("class balance per dataset")
    single = [d for d, g in df.groupby("dataset") if g.binary_label.nunique() < 2]
    A.add("datasets contributing only one binary class", "HARD", not single, single or 0)
    ann_only = {d for d, g in df.groupby("dataset") if (g.label_source == "temporal_annotation").all()}
    absent = [f"{d}/{sp}" for d in sorted(set(df.dataset) - set(S.TRAIN_ONLY_DATASETS) - ann_only) for sp in S.SPLIT_FRACTIONS
              if not ((df.dataset == d) & (df.split == sp)).any()]
    A.add("datasets absent from a split (no in-distribution evaluation possible)", "WARN", not absent, absent or 0)
    ss = [f"{d}/{s}" for (d, s), g in df.groupby(["dataset", "split"]) if g.binary_label.nunique() < 2]
    A.add("dataset/split cells with only one binary class", "WARN", not ss, ss or 0)
    for sp, g in df.groupby("split"):
        share = g.groupby("dataset").sample_weight.sum() / g.sample_weight.sum()
        share_d = g.groupby("dataset").sample_weight_dataset_balanced.sum() / g.sample_weight_dataset_balanced.sum()
        summary.setdefault("weight_share", {})[sp] = {"sample_weight": share.round(4).to_dict(),
                                                      "dataset_balanced": share_d.round(4).to_dict()}
    print("  effective training mass by dataset (train split):")
    for k in ("sample_weight", "dataset_balanced"):
        print(f"    {k:<18}", {d: f"{100*v:.1f}%" for d, v in summary["weight_share"]["train"][k].items()})

    section("removed rows")
    by_reason = rm.groupby(["dataset", "reason_code"]).agg(videos=("video_id", "size"),
                                                           segments_affected=("n_segments_affected", "sum")).reset_index()
    by_reason.to_csv(OUT / "removed_by_reason.csv", index=False); print(by_reason.to_string(index=False))
    conflicts = int(rm.detail.astype(str).str.contains("LABEL CONFLICT").sum())
    A.add("duplicate files with conflicting labels", "WARN", conflicts == 0, conflicts)

    if not args.skip_external:
        section("external benchmarks: near-duplicates of XD-test / UCF-test clips inside the manifest")
        exdf = external_matches(df, REC, fine_m)
        for name in EXTERNAL:
            g = exdf[exdf.benchmark == name]
            A.add(f"{name}: manifest windows near-duplicating external clips (coarse)", "WARN", len(g) == 0, len(g),
                  f"{g.external_video.nunique()} external videos involved")
        rev = pd.read_csv(REVIEWED_MATCHES)
        excl_set = set(pd.read_csv(MAN / "external_eval_exclusions.csv").external_video)
        fp_set = set(rev[rev.visual_verdict == "false_positive"].external_video)
        best = exdf.groupby("external_video").grad128.max()
        prec = set(best[best >= S.EXTERNAL_PRECAUTION_MIN_R].index) - excl_set - fp_set
        exdf["review_status"] = ["excluded_from_evaluation" if v in excl_set else "reviewed_false_positive" if v in fp_set
                                 else "precautionary_excluded_unreviewed" if v in prec else "unreviewed_low_structure"
                                 for v in exdf.external_video]
        exdf.to_csv(OUT / "external_benchmark_matches.csv", index=False)
        extw = pd.read_csv(MAN / "external_windows.csv")
        if "excluded_precaution" not in extw: extw["excluded_precaution"] = 0
        flagged = set(extw[(extw.excluded_contamination == 1) | (extw.excluded_precaution == 1)].video)
        unflagged = sorted(v for v in (prec | excl_set) if v in set(extw.video) and v not in flagged)
        A.add("confirmed or precautionary external matches still in external evaluation", "HARD", not unflagged,
              len(unflagged), "; ".join(unflagged[:10]))
        A.add(f"unreviewed external videos with grad128 >= {S.EXTERNAL_PRECAUTION_MIN_R} (excluded from evaluation as a precaution)",
              "WARN", not prec, len(prec), "visual review would allow restoring them")
        low = exdf[exdf.review_status == "unreviewed_low_structure"]
        A.add(f"unreviewed external matches below grad128 {S.EXTERNAL_PRECAUTION_MIN_R} (kept in evaluation)", "INFO", True,
              int(low.external_video.nunique()), f"{len(low)} windows; max grad128 {low.grad128.max() if len(low) else float('nan'):.3f}")

    section("known unavailable sources (not fabricated)")
    unavailable = {
        "xd_train_temporal_annotations": "none on disk (all 500 XD annotations are XD-test videos)",
        "ucf_train_temporal_annotations": "none (annotation file covers UCF-test only)",
        "ucf_normal_videos": "not on disk (local mirror truncated)",
        "same_video_quiet_negatives": "not generated: require training-side temporal annotations",
    }
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(60)
        from huggingface_hub import list_repo_files
        fs = [f for f in list_repo_files("jherng/xd-violence", repo_type="dataset") if f.lower().endswith((".mp4", ".avi"))]
        signal.alarm(0)
        CODE = {"B1": "Fighting", "B2": "Shooting", "B4": "Riot", "B5": "Abuse", "B6": "CarAccident", "G": "Explosion", "A": "Normal"}
        tr = [f for f in fs if "test" not in f.lower()]
        cnt = {}
        for f in tr:
            m = re.search(r"label_([A-Z0-9-]+)\.", f)
            for t in (m.group(1).split("-") if m else []):
                if t in CODE: cnt[CODE[t]] = cnt.get(CODE[t], 0) + 1
        unavailable["xd_train_listing_hf"] = {"videos": len(tr), "by_label": cnt,
                                              "downloaded_locally": "Riot 381, Fighting(film) 274, Normal 700 only"}
    except Exception as e:
        signal.alarm(0); unavailable["xd_train_listing_hf"] = f"not verified ({type(e).__name__})"
    for k, v in unavailable.items(): print(f"  {k}: {v}")
    json.dump(unavailable, open(OUT / "unavailable_sources.json", "w"), indent=2)

    hard_fail = [r for r in A.rows if r["level"] == "HARD" and not r["passed"]]
    warns = [r for r in A.rows if r["level"] == "WARN" and not r["passed"]]
    summary.update(checks=A.rows, hard_failures=len(hard_fail), warnings=len(warns),
                   runtime_min=round((time.time() - t0) / 60, 1))
    json.dump(summary, open(OUT / "audit_summary.json", "w"), indent=2, default=str)
    with open(OUT / "audit_report.txt", "w") as f:
        for r in A.rows:
            f.write(f"[{'PASS' if r['passed'] else ('FAIL' if r['level']=='HARD' else r['level'])}] {r['check']}: {r['value']} {r['detail']}\n")
    section("VERDICT")
    print(f"  hard checks failed: {len(hard_fail)} | warnings: {len(warns)} | {summary['runtime_min']} min")
    for r in hard_fail: print(f"  FAIL {r['check']}: {r['value']}")
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
