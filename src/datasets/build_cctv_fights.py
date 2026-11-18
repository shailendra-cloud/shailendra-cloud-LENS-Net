#!/usr/bin/env python3
"""CCTV-Fights confirmation windows (PREREG_journal Sec. 7). 1.5-s non-overlapping grid; positive if >= 50% of the window
lies in a Fight segment, negative if it lies >= 1.0 s outside every segment, otherwise ignored. Per video <= 8 windows:
up to 4 of each class spread evenly, the other class filling places a class cannot use. 32 frames at 256 px, JPEG q92.
fps and frame counts come from groundtruth.json (OpenCV frame counts are unreliable for webm). cluster = video.
Frame store 'cfights' under $LENSNET_ROOT/raw_data/journal/frames."""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT))
import video_data.frame_store as VF
VF.STORE_ROOT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/frames")
from video_data import label_schema as S
from video_data.build_external_windows import decode_video, spread

CD = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/cctvfights")
GT = json.load(open(CD / "groundtruth.json"))["database"]
FILES = {p.stem: p for p in (CD / "videos").iterdir()}
GUARD_S, PER_CLASS, CAP = 1.0, 4, 8


def plan(name):
    g = GT[name]; fps = g["frame_rate"]; tot = g["nb_frames"]
    span = int(round(S.WINDOW_SECONDS * fps)); guard = GUARD_S * fps
    segs = [(a * fps, b * fps) for a, b in (x["segment"] for x in g["annotations"] if x["label"] == "Fight")]
    grid = [(s, s + span - 1) for s in range(0, max(tot - span + 1, 0), span)]
    def overlap(w): return sum(max(0.0, min(w[1] + 1, b) - max(w[0], a)) for a, b in segs)
    def outside(w): return all(w[1] + 1 <= a - guard or w[0] >= b + guard for a, b in segs)
    pos = [w for w in grid if overlap(w) >= 0.5 * span]; neg = [w for w in grid if outside(w)]
    npos = min(len(pos), CAP - min(len(neg), PER_CLASS)); nneg = min(len(neg), CAP - npos)
    rows = [dict(benchmark="CCTV-Fights", video=name, cluster=name, path=str(FILES[name]), category=g["source"],
                 subset=g["subset"], fps=fps, total_frames=tot, start_frame=w[0], end_frame=w[1], binary_label=lab)
            for lab, ws in ((1, spread(pos, npos)), (0, spread(neg, nneg))) for w in ws]
    return sorted(rows, key=lambda r: r["start_frame"]), len(grid), len(grid) - len(pos) - len(neg)


def main():
    t0 = time.time(); names = sorted(GT); assert set(names) == set(FILES)
    with ProcessPoolExecutor(16) as ex: plans = list(ex.map(plan, names))
    df = pd.DataFrame([r for pl, _, _ in plans for r in pl])
    print(f"grid windows {sum(p[1] for p in plans)}, ignored (ambiguous) {sum(p[2] for p in plans)}, selected {len(df)}", flush=True)
    jobs = [(p, [(r.start_frame, r.end_frame) for r in g.itertuples()]) for p, g in df.groupby("path", sort=False)]
    w = VF.StoreWriter("cfights"); failed = []
    with ProcessPoolExecutor(16) as ex:
        for (p, wins), frames in zip(jobs, ex.map(decode_video, jobs, chunksize=2)):
            for (s, e), fr in zip(wins, frames):
                if fr is None: failed.append((p, s)); continue
                w.add(VF.content_key(p, s, e, decode="seq"), fr)
    w.close(dict(note="CCTV-Fights confirmation (PREREG_journal Sec. 7)"))
    fk = set(failed)
    df["decode_failed"] = [int((p, s) in fk) for p, s in zip(df.path, df.start_frame)]
    out = ROOT / "workspace/cfights"; out.mkdir(exist_ok=True)
    df.to_csv(out / "cfights_windows.csv", index=False)
    print(f"videos {df.video.nunique()} of {len(names)}, windows {len(df)}, decode failures {len(failed)}, {time.time()-t0:.0f}s")
    print(df[df.decode_failed == 0].groupby(["category", "binary_label"]).agg(videos=("video", "nunique"), windows=("path", "size")))


if __name__ == "__main__":
    main()
