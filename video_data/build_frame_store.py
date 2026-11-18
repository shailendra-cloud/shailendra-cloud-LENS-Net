#!/usr/bin/env python3
"""Build the supplementary frame store for manifest windows that are not in the v1 cache.

Frames are produced by cnn_bigru_model.cache_one_clip, which STEP 1 verified is byte-identical to the
v1 cache builder, so supplementary windows are preprocessed exactly like cached ones.
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from video_data import cnn_bigru_model as mm
from video_data.frame_store import StoreWriter, content_key, N_FRAMES, SIZE, QUALITY


def _one(t):
    path, s, e = t
    return mm.cache_one_clip((path, N_FRAMES, SIZE, QUALITY, int(s), int(e)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data_tables/manifests/source_windows.csv"))
    ap.add_argument("--name", default="supp")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    df = pd.read_csv(a.manifest)
    need = df[df.v1_cache_row < 0].drop_duplicates(["path", "start_frame", "end_frame"])
    tasks = list(zip(need.path, need.start_frame, need.end_frame))
    print(f"[store:{a.name}] {len(tasks)} windows to decode ({a.workers} workers)", flush=True)
    w, failed, t0 = StoreWriter(a.name), [], time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (t, jp) in enumerate(zip(tasks, ex.map(_one, tasks, chunksize=4))):
            if jp is None or len(jp) != N_FRAMES:
                failed.append(dict(path=t[0], start=int(t[1]), end=int(t[2]))); continue
            w.add(content_key(*t, decode="seek"), jp)
            if k % 500 == 0: print(f"  {k}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    w.close({"source_manifest": a.manifest, "failed": len(failed)})
    json.dump(failed, open(w.dir / "failed.json", "w"), indent=2)
    print(f"[store:{a.name}] done: {len(w.keys)} windows, {w.pos/1e9:.2f} GB, {len(failed)} failed, {time.time()-t0:.0f}s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
