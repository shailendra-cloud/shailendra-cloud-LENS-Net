#!/usr/bin/env python3
"""E4 (PREREG_revision.md, Part E and its clarification): ambiguous windows (neither >= 50% inside a fight nor >= 1 s
outside every fight) of the CCTV-Fights CCTV videos and the UBI-Fights videos that are in the evaluation sets, from the
same 1.5-s grids as build_cctv_fights.py and confirm_footage_gate.plan_ubi, up to 4 per video spread evenly, label 0 (added as negatives).
Frame store 'amb' under raw_data/journal/frames; windows in workspace/amb/amb_windows.csv."""
import sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, cv2
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import build_cctv_fights as B, confirm_footage_gate as C2
import video_data.frame_store as VF
from video_data.build_external_windows import decode_video, spread
VF.STORE_ROOT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/frames")
OUT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace/amb"); CAP = 4


def amb_cctv(name):
    g = B.GT[name]; fps = g["frame_rate"]; tot = g["nb_frames"]
    span = int(round(B.S.WINDOW_SECONDS * fps)); guard = B.GUARD_S * fps
    segs = [(a * fps, b * fps) for a, b in (x["segment"] for x in g["annotations"] if x["label"] == "Fight")]
    grid = [(s, s + span - 1) for s in range(0, max(tot - span + 1, 0), span)]
    def overlap(w): return sum(max(0.0, min(w[1] + 1, b) - max(w[0], a)) for a, b in segs)
    def outside(w): return all(w[1] + 1 <= a - guard or w[0] >= b + guard for a, b in segs)
    amb = [w for w in grid if not overlap(w) >= 0.5 * span and not outside(w)]
    return [dict(benchmark="CCTV-Fights", video=name, cluster=name, path=str(B.FILES[name]), category=g["source"], fps=fps, total_frames=tot,
                 start_frame=w[0], end_frame=w[1], binary_label=0) for w in spread(amb, min(len(amb), CAP))]


def amb_ubi(a):
    p, cat = a
    y = pd.read_csv(C2.UBA / f"{p.stem}.csv", header=None)[0].values.astype(int)
    cap = cv2.VideoCapture(str(p)); fps = cap.get(cv2.CAP_PROP_FPS) or C2.S.FPS_FALLBACK; cap.release()
    tot = len(y); span = int(round(C2.S.WINDOW_SECONDS * fps)); guard = int(round(C2.GUARD_S * fps))
    grid = [(s, s + span - 1) for s in range(0, max(tot - span + 1, 0), span)]
    near = np.convolve(y, np.ones(2 * guard + 1), mode="same") > 0
    amb = [w for w in grid if not y[w[0]:w[1] + 1].mean() >= 0.5 and near[w[0]:w[1] + 1].any()]
    return [dict(benchmark="UBI-Fights", video=p.stem, cluster=f"UBI/{p.stem}", path=str(p), category=cat, fps=fps, total_frames=tot,
                 start_frame=w[0], end_frame=w[1], binary_label=0) for w in spread(amb, min(len(amb), CAP))]


def main():
    t0 = time.time()
    cf = pd.read_csv(B.ROOT / "workspace/cfights/cfights_windows.csv"); cf = cf[(cf.decode_failed == 0) & (cf.category == "CCTV")]
    c2 = pd.read_csv(C2.C2 / "conf2_windows.csv"); c2 = c2[(c2.decode_failed == 0) & (c2.benchmark == "UBI-Fights")]
    ubi = c2.drop_duplicates("video")[["path", "category"]]
    with ProcessPoolExecutor(16) as ex:
        rows = [r for pl in ex.map(amb_cctv, sorted(cf.video.unique())) for r in pl]
        rows += [r for pl in ex.map(amb_ubi, [(Path(p), c) for p, c in ubi.itertuples(index=False)]) for r in pl]
    df = pd.DataFrame(rows)
    jobs = [(p, [(r.start_frame, r.end_frame) for r in g.itertuples()]) for p, g in df.groupby("path", sort=False)]
    w = VF.StoreWriter("amb"); failed = []
    with ProcessPoolExecutor(16) as ex:
        for (p, wins), frames in zip(jobs, ex.map(decode_video, jobs, chunksize=2)):
            for (s, e), fr in zip(wins, frames):
                if fr is None: failed.append((p, s)); continue
                w.add(VF.content_key(p, s, e, decode="seq"), fr)
    w.close(dict(note="E4 ambiguous windows (PREREG_revision Part E)"))
    fk = set(failed); df["decode_failed"] = [int((p, s) in fk) for p, s in zip(df.path, df.start_frame)]
    OUT.mkdir(parents=True, exist_ok=True); df.to_csv(OUT / "amb_windows.csv", index=False)
    print(f"windows {len(df)}, decode failures {len(failed)}, {time.time()-t0:.0f}s"); print(df.groupby("benchmark").agg(videos=("video", "nunique"), windows=("path", "size")))


if __name__ == "__main__":
    main()
