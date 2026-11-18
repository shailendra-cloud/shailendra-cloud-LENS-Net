#!/usr/bin/env python3
"""Clean external evaluation windows (1.5 s) for XD-test and UCF-test, plus their frame store.

Replaces the old external protocol (5 s chunks sampled at a different rate than training, and
including footage shared with training data). Rules, from video_data/label_schema.py:
  * 1.5 s non-overlapping grid over each video
  * VIOLENCE if >= 50% of the window overlaps an annotated violent interval
  * NON-VIOLENCE (same_video_quiet) only if the window is outside every interval by the guard band
    (XD 2 s, UCF 1 s); anything else is IGNORE and never scored
  * quiet windows >= 3 s apart, at most 2 per selected violent window and <= 8 per video
  * XD Normal videos and UCF non-violent anomaly videos supply NON-VIOLENCE windows
  * Explosion and Arrest are IGNORE categories; XD CarAccident is a hard negative
  * visually confirmed contaminated external videos are flagged and excluded from evaluation
These windows are for evaluation only: never training, threshold selection or model selection.
"""
import argparse, json, os, re, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from video_data import label_schema as S
from video_data.frame_store import StoreWriter, content_key, N_FRAMES, SIZE, QUALITY

XD_DIR = ROOT / "raw_data/xd_violence/data/video/test_videos"
XD_ANN = ROOT / "raw_data/annotations/xd_violence_annotations.txt"
UCF_ANN = ROOT / "raw_data/annotations/ucf_temporal_annotation.txt"
UCF_BASE = ROOT / "raw_data/kagglehub_cache_journal/datasets/minhajuddinmeraj/anomalydetectiondatasetucf/versions/50"
OUT_CSV = ROOT / "data_tables/manifests/external_windows.csv"
XD_V = {"B1": "FIGHT_ASSAULT", "B2": "SHOOTING_WEAPON", "B4": "RIOT_CROWD", "B5": "ABUSE"}
UCF_V = {"Abuse": "ABUSE", "Assault": "FIGHT_ASSAULT", "Fighting": "FIGHT_ASSAULT", "Robbery": "OTHER_VIOLENCE",
         "Shooting": "SHOOTING_WEAPON"}
UCF_NV = ("Arson", "Burglary", "RoadAccidents", "Shoplifting", "Stealing", "Vandalism")
MAX_V, MAX_Q, MAX_NORMAL, MAX_HARD = 8, 8, 4, 4


def spread(items, k):
    if len(items) <= k: return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def spaced(items, k, gap):
    out = []
    for w in spread(items, len(items)):
        if len(out) >= k: break
        if all(abs(w[0] - o[0]) >= gap for o in out): out.append(w)
    if len(out) < k:                                  # fill from an even spread if spacing left room
        for w in spread(items, len(items)):
            if len(out) >= k: break
            if w not in out and all(abs(w[0] - o[0]) >= gap for o in out): out.append(w)
    return sorted(out)


def cap_total(rows, cap=None):
    """Canonical rule (Phase 5, Decision 3): at most 8 windows per source video IN TOTAL, counting every
    class together. Violent windows are kept first, then negatives spread evenly over the remaining room.
    Identical to the cap used by manifest generation, annotation derivation and validation."""
    cap = S.MAX_WINDOWS_TOTAL_PER_VIDEO if cap is None else cap
    if len(rows) <= cap: return rows
    viol = [r for r in rows if r["binary_label"] == 1][:cap]
    neg = [r for r in rows if r["binary_label"] == 0]
    return sorted(viol + spread(neg, cap - len(viol)), key=lambda r: r["start_frame"])


def plan_video(bench, name, path, category, intervals, mode, violence_type, fps, total):
    span = int(round(S.WINDOW_SECONDS * (fps or S.FPS_FALLBACK)))
    guard = int(round(S.GUARD_SECONDS["XD" if bench == "XD-test" else "UCF"] * (fps or S.FPS_FALLBACK)))
    gap = int(round(S.QUIET_MIN_GAP_SECONDS * (fps or S.FPS_FALLBACK)))
    grid = [(s, s + span - 1) for s in range(0, max(total - span + 1, 0), span)]
    def overlap(w): return sum(max(0, min(w[1] + 1, b) - max(w[0], a)) for a, b in intervals)
    def outside(w): return all(w[1] + 1 <= a - guard or w[0] >= b + guard for a, b in intervals)
    rows = []
    def add(w, lab, status, nk):
        rows.append(dict(benchmark=bench, video=name, path=str(path), category=category, fps=fps, total_frames=total,
                         start_frame=w[0], end_frame=w[1], binary_label=lab, label_status=status, negative_kind=nk,
                         violence_type=violence_type if lab == 1 else "NONE"))
    if mode == "violent":
        viol = [w for w in grid if overlap(w) >= 0.5 * span]
        quiet = [w for w in grid if outside(w)]
        sel_v = spread(viol, MAX_V)
        for w in sel_v: add(w, 1, "violent_interval", "none")
        for w in spaced(quiet, min(S.QUIET_PER_VIOLENT_WINDOW * len(sel_v), MAX_Q), gap): add(w, 0, "same_video_quiet", "same_video_quiet")
        n_ignored = len(grid) - len(viol) - len(quiet)
    elif mode == "normal":
        for w in spaced(grid, MAX_NORMAL, gap): add(w, 0, "normal_video", "everyday")
        n_ignored = 0
    elif mode == "hard_negative":
        inside = [w for w in grid if overlap(w) >= 0.5 * span]
        quiet = [w for w in grid if outside(w)]
        for w in spread(inside, MAX_HARD): add(w, 0, "non_violent_anomaly", "hard_negative_anomaly")
        for w in spaced(quiet, MAX_HARD, gap): add(w, 0, "same_video_quiet", "same_video_quiet")
        n_ignored = 0
    return cap_total(rows), len(grid)


def decode_video(job):
    """One sequential pass per video (seeking in long H.264 lands on keyframes)."""
    path, windows = job
    want = {}
    for wi, (s, e) in enumerate(windows):
        for f in np.linspace(s, e, N_FRAMES).round().astype(int):
            want.setdefault(int(f), []).append(wi)
    got = {wi: [] for wi in range(len(windows))}
    cap = cv2.VideoCapture(str(path)); i, last = 0, max(want) if want else -1
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY]
    while i <= last:
        ok, fr = cap.read()
        if not ok: break
        if i in want:
            _, e = cv2.imencode(".jpg", cv2.resize(fr, (SIZE, SIZE), interpolation=cv2.INTER_AREA), enc)
            for wi in want[i]: got[wi].append(e.tobytes())
        i += 1
    cap.release()
    out = []
    for wi in range(len(windows)):
        fr = got[wi]
        if len(fr) < N_FRAMES // 2: out.append(None); continue
        while len(fr) < N_FRAMES: fr.append(fr[-1])
        out.append(fr[:N_FRAMES])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--store", default="external")
    ap.add_argument("--plan-only", action="store_true",
                    help="re-plan the windows and rewrite the CSV, reusing the existing frame store "
                         "(valid when the new plan is a subset of the stored windows; verified below)")
    a = ap.parse_args(); t0 = time.time()
    rev = pd.read_csv(ROOT / S.CONTAMINATION_REVIEW)
    contaminated = set(rev[rev.visual_verdict == "same_footage"].external_video)
    plans, skipped = [], []
    # ---- XD-test
    ann = {}
    for line in open(XD_ANN):
        p = line.split()
        if p:
            nm = p[0][:-4] if p[0].endswith(".mp4") else p[0]
            nums = [int(x) for x in p[1:] if x.lstrip("-").isdigit()]
            ann[nm] = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2) if nums[i + 1] > nums[i] >= 0]
    for fn in sorted(os.listdir(XD_DIR)):
        nm = fn[:-4]; m = re.search(r"_label_([A-Z0-9\-]+)$", nm)
        toks = [t for t in (m.group(1).split("-") if m else []) if t not in ("0", "")]
        path = XD_DIR / fn
        cap = cv2.VideoCapture(str(path)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or S.FPS_FALLBACK; cap.release()
        if total <= 0: skipped.append((nm, "unreadable")); continue
        v = [t for t in toks if t in XD_V]
        if toks == ["A"]:
            plans.append(("XD-test", nm, path, "Normal", [], "normal", "NONE", fps, total))
        elif v:
            vt = XD_V[v[0]] if len(set(v)) == 1 and not (set(toks) - set(v)) else "OTHER_VIOLENCE"
            if not ann.get(nm): skipped.append((nm, "violent label without annotation")); continue
            plans.append(("XD-test", nm, path, "+".join(toks), ann[nm], "violent", vt, fps, total))
        elif "G" in toks:
            skipped.append((nm, "ignore_category_explosion"))
        elif "B6" in toks:
            plans.append(("XD-test", nm, path, "CarAccident", ann.get(nm, []), "hard_negative", "NONE", fps, total))
        else:
            skipped.append((nm, f"unrecognised labels {toks}"))
    # ---- UCF-test
    disk = {}
    for dp, _, fs in os.walk(UCF_BASE):
        for f in fs:
            if f.endswith(".mp4"): disk.setdefault(f, os.path.join(dp, f))
    for line in open(UCF_ANN):
        p = line.split()
        if len(p) < 4: continue
        nm, cat = p[0], p[1]
        nums = [int(x) for x in p[2:]]
        iv = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2) if nums[i] >= 0 and nums[i + 1] > nums[i]]
        if nm not in disk: skipped.append((nm, "not on disk")); continue
        if cat in S.UCF_IGNORE: skipped.append((nm, f"ignore_category_{cat}")); continue
        cap = cv2.VideoCapture(disk[nm]); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS) or S.FPS_FALLBACK; cap.release()
        if total <= 0: skipped.append((nm, "unreadable")); continue
        if cat in UCF_V: plans.append(("UCF-test", nm, disk[nm], cat, iv, "violent", UCF_V[cat], fps, total))
        elif cat in UCF_NV: plans.append(("UCF-test", nm, disk[nm], cat, iv, "hard_negative", "NONE", fps, total))
        else: skipped.append((nm, f"category {cat} not used"))
    rows, n_grid = [], 0
    for pl in plans:
        r, g = plan_video(*pl); rows += r; n_grid += g
    df = pd.DataFrame(rows)
    df["excluded_contamination"] = df.video.isin(contaminated).astype(int)
    df["frame_key"] = [content_key(p, s, e, decode="seq") for p, s, e in zip(df.path, df.start_frame, df.end_frame)]
    print(f"[external] {len(plans)} videos planned, {len(skipped)} skipped, {len(df)} windows "
          f"({int((df.binary_label==1).sum())} violent / {int((df.binary_label==0).sum())} non-violent)", flush=True)
    if a.plan_only:
        from video_data.frame_store import FrameStore
        have = set(FrameStore(stores=(a.store,), use_v1=False).index)
        missing = sorted(set(df.frame_key) - have)
        df["decode_failed"] = (~df.frame_key.isin(have)).astype(int)
        if "excluded_precaution" not in df: df["excluded_precaution"] = 0
        df.to_csv(OUT_CSV, index=False)
        print(f"[external] plan-only: {len(df)} windows written to {OUT_CSV}")
        print(f"[external] windows already in store '{a.store}': {len(df) - len(missing)} | missing: {len(missing)}")
        if missing: print("  first missing keys:", missing[:5])
        print(df.groupby(["benchmark", "label_status", "excluded_contamination"]).size().to_string())
        return
    jobs = [(p, list(zip(g.start_frame, g.end_frame))) for p, g in df.groupby("path", sort=False)]
    keys = [list(g.frame_key) for _, g in df.groupby("path", sort=False)]
    w, failed = StoreWriter(a.store), []
    with ProcessPoolExecutor(a.workers) as ex:
        for n, (kk, res) in enumerate(zip(keys, ex.map(decode_video, jobs, chunksize=1))):
            for k, jp in zip(kk, res):
                if jp is None: failed.append(k)
                else: w.add(k, jp)
            if n % 50 == 0: print(f"  video {n}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    w.close({"windows_csv": str(OUT_CSV)})
    df["decode_failed"] = df.frame_key.isin(set(failed)).astype(int)
    df.to_csv(OUT_CSV, index=False)
    pd.DataFrame(skipped, columns=["video", "reason"]).to_csv(ROOT / "data_tables/manifests/external_videos_skipped.csv", index=False)
    print(f"[external] store '{a.store}': {len(w.keys)} windows {w.pos/1e9:.2f} GB | decode failures {len(failed)} | {time.time()-t0:.0f}s")
    print(df.groupby(["benchmark", "label_status", "excluded_contamination"]).size().to_string())


if __name__ == "__main__":
    main()
