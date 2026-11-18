#!/usr/bin/env python3
"""Journal confirmation sets (PREREG_journal Sec. 3): VioPeru and AIRTLab. Same windowing as build_scfd_scvd.py:
1.5-s non-overlapping windows, <= 8 per video spread evenly, 32 frames at 256 px, JPEG q92. Every window takes its
video's label. `cluster` is the bootstrap unit (VioPeru: video; AIRTLab: label/clip number, shared by cam1 and cam2).
Frame store 'conf' under $LENSNET_ROOT/raw_data/journal/frames."""
import sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2, pandas as pd
ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT))
import video_data.frame_store as VF
VF.STORE_ROOT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/frames")
from video_data import label_schema as S
from video_data.build_external_windows import decode_video, spread

JD = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal")
vids = []
for sub, lab in (("train/Fight", 1), ("val/Fight", 1), ("train/NonFight", 0), ("val/NonFight", 0), ("false_positives_validation", 0)):
    vids += [("VioPeru", p, lab, sub, f"{sub}/{p.stem}") for p in sorted((JD / "viop" / sub).glob("*.avi"))]
AD = JD / "airt/violence-detection-dataset"
for cls, lab in (("violent", 1), ("non-violent", 0)):
    for cam in ("cam1", "cam2"):
        vids += [("AIRTLab", p, lab, f"{cls}/{cam}", f"{cls}/{p.stem}") for p in sorted((AD / cls / cam).glob("*.mp4"))]


def plan(v):
    bench, p, lab, cat, cluster = v
    cap = cv2.VideoCapture(str(p)); fps = cap.get(cv2.CAP_PROP_FPS) or S.FPS_FALLBACK; tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    span = int(round(S.WINDOW_SECONDS * (fps if fps > 1 else S.FPS_FALLBACK)))
    ws = [(s, s + span - 1) for s in range(0, max(tot - span + 1, 0), span)]
    if not ws and tot >= 8: ws = [(0, tot - 1)]
    rows = [dict(benchmark=bench, video=f"{cat}/{p.stem}", cluster=cluster, path=str(p), category=cat, fps=fps,
                 total_frames=tot, start_frame=s, end_frame=e, binary_label=lab) for s, e in ws]
    return rows if len(rows) <= 8 else sorted(spread(rows, 8), key=lambda r: r["start_frame"])


def main():
    t0 = time.time()
    with ProcessPoolExecutor(16) as ex: plans = list(ex.map(plan, vids))
    df = pd.DataFrame([r for pl in plans for r in pl])
    skipped = [str(v[1]) for v, pl in zip(vids, plans) if not pl]
    jobs = [(p, [(r.start_frame, r.end_frame) for r in g.itertuples()]) for p, g in df.groupby("path", sort=False)]
    w = VF.StoreWriter("conf"); failed = []
    with ProcessPoolExecutor(16) as ex:
        for (p, wins), frames in zip(jobs, ex.map(decode_video, jobs, chunksize=4)):
            for (s, e), fr in zip(wins, frames):
                if fr is None: failed.append((p, s)); continue
                w.add(VF.content_key(p, s, e, decode="seq"), fr)
    w.close(dict(note="journal confirmation sets (VioPeru, AIRTLab)"))
    fk = set(failed)
    df["decode_failed"] = [int((p, s) in fk) for p, s in zip(df.path, df.start_frame)]
    out = ROOT / "workspace/conf"; out.mkdir(exist_ok=True)
    df.to_csv(out / "conf_windows.csv", index=False)
    print(f"videos {len(vids)} (skipped {len(skipped)}), windows {len(df)}, decode failures {len(failed)}, {time.time()-t0:.0f}s")
    print(df.groupby(["benchmark", "binary_label"]).agg(videos=("video", "nunique"), clusters=("cluster", "nunique"), windows=("path", "size")))
    if skipped: print("skipped:", skipped[:10])


if __name__ == "__main__":
    main()
