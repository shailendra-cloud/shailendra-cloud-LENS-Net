#!/usr/bin/env python3
"""Fresh confirmatory sets (PREREG deviation 6): SCFD (Akti et al. 2019) and SCVD (full converted videos, Train+Test,
none used for training). 1.5-s non-overlapping windows, <= 8 per video spread evenly, 32 frames at 256 px, JPEG q92.
Every window takes its video's label. Frame store 'fresh' under $LENSNET_ROOT/raw_data/fresh/frames."""
import sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd
ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT))
import video_data.frame_store as VF
VF.STORE_ROOT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/fresh/frames")
from video_data import label_schema as S
from video_data.build_external_windows import decode_video, spread

FD = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/fresh")
SC = FD / "kh/datasets/toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd/versions/3/SCVD/SCVD_converted"
vids = [("SCFD", p, 1, "fight") for p in sorted((FD / "scfd/fight").glob("*.mp4"))] + \
       [("SCFD", p, 0, "noFight") for p in sorted((FD / "scfd/noFight").glob("*.mp4"))]
for part in ("Train", "Test"):
    for cat, lab in (("Violence", 1), ("Weaponized", 1), ("Normal", 0)):
        vids += [("SCVD", p, lab, cat) for p in sorted((SC / part / cat).glob("*.avi"))]


def plan(v):
    bench, p, lab, cat = v
    cap = cv2.VideoCapture(str(p)); fps = cap.get(cv2.CAP_PROP_FPS) or S.FPS_FALLBACK; tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    span = int(round(S.WINDOW_SECONDS * (fps if fps > 1 else S.FPS_FALLBACK)))
    ws = [(s, s + span - 1) for s in range(0, max(tot - span + 1, 0), span)]
    if not ws and tot >= 8: ws = [(0, tot - 1)]
    rows = [dict(benchmark=bench, video=p.stem if bench == "SCFD" else f"{cat}/{p.stem}", path=str(p), category=cat, fps=fps,
                 total_frames=tot, start_frame=s, end_frame=e, binary_label=lab) for s, e in ws]
    return rows if len(rows) <= 8 else sorted(spread(rows, 8), key=lambda r: r["start_frame"])


def main():
    t0 = time.time()
    with ProcessPoolExecutor(16) as ex: plans = list(ex.map(plan, vids))
    rows = [r for pl in plans for r in pl]
    df = pd.DataFrame(rows)
    skipped = [str(v[1]) for v, pl in zip(vids, plans) if not pl]
    jobs = [(p, [(r.start_frame, r.end_frame) for r in g.itertuples()]) for p, g in df.groupby("path", sort=False)]
    w = VF.StoreWriter("fresh"); failed = []
    with ProcessPoolExecutor(16) as ex:
        for (p, wins), frames in zip(jobs, ex.map(decode_video, jobs, chunksize=4)):
            for (s, e), fr in zip(wins, frames):
                if fr is None: failed.append((p, s)); continue
                w.add(VF.content_key(p, s, e, decode="seq"), fr)
    w.close(dict(note="fresh confirmatory sets"))
    fk = set(failed)
    df["decode_failed"] = [int((p, s) in fk) for p, s in zip(df.path, df.start_frame)]
    df.to_csv(ROOT / "workspace/fresh/fresh_windows.csv", index=False)
    print(f"videos {len(vids)} (skipped {len(skipped)}), windows {len(df)}, decode failures {len(failed)}, {time.time()-t0:.0f}s")
    print(df.groupby(["benchmark", "binary_label"]).agg(videos=("video", "nunique"), windows=("path", "size")))
    if skipped: print("skipped:", skipped[:10])


if __name__ == "__main__":
    main()
