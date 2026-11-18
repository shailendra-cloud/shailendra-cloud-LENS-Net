#!/usr/bin/env python3
"""Label-free, content-keyed, memory-mapped frame store.

A window's frames are identified only by content: sha1(path | start | end | 32 | 256 | 92 | decode).
Labels, splits and weights are never stored with the frames; they are joined from manifests at load
time, so relabelling or re-splitting never invalidates cached frames. Windows already in the v1 cache
are read from it directly (memory-mapped), never duplicated.

Layout: source_benchmark/frames/<store>/blob.bin (uint8, memory-mapped), offsets.npy (N, 33 int64),
keys.npy (N,), info.json.
"""
import hashlib, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STORE_ROOT = ROOT / "source_benchmark" / "frames"
V1_CACHE = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[1]) / "raw_data/frame_cache/clip_T32_S256_N31279_v1.npz")
N_FRAMES, SIZE, QUALITY = 32, 256, 92


def content_key(path, start, end, decode="seek"):
    return hashlib.sha1(f"{path}|{int(start)}|{int(end)}|{N_FRAMES}|{SIZE}|{QUALITY}|{decode}".encode()).hexdigest()


class StoreWriter:
    """Append-only writer; the store only becomes visible (renamed into place) when closed."""
    def __init__(self, name):
        self.dir = STORE_ROOT / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.dir / "blob.bin.part", "wb")
        self.pos, self.keys, self.offs = 0, [], []

    def add(self, key, jpegs):
        if len(jpegs) != N_FRAMES:
            raise ValueError(f"expected {N_FRAMES} frames, got {len(jpegs)}")
        o = [self.pos]
        for b in jpegs:
            self.fh.write(b); self.pos += len(b); o.append(self.pos)
        self.keys.append(key); self.offs.append(o)

    def close(self, extra=None):
        self.fh.close()
        np.save(self.dir / "offsets.npy", np.asarray(self.offs, np.int64))
        np.save(self.dir / "keys.npy", np.asarray(self.keys))
        os.replace(self.dir / "blob.bin.part", self.dir / "blob.bin")
        json.dump({"windows": len(self.keys), "bytes": self.pos, **(extra or {})}, open(self.dir / "info.json", "w"), indent=2)


class FrameStore:
    """Resolve a window to its 32 JPEG frames from the v1 cache (by row) or a content-keyed store."""
    def __init__(self, stores=(), use_v1=True):
        self.v1 = None
        if use_v1:
            sys.path.insert(0, str(ROOT))
            from video_data.video_io import open_v1_cache_mmap
            self.v1 = open_v1_cache_mmap(V1_CACHE)
        self.index, self.blobs, self.offsets = {}, {}, {}
        for name in stores:
            d = STORE_ROOT / name
            if not (d / "blob.bin").exists():
                raise FileNotFoundError(f"frame store '{name}' is not built ({d})")
            self.blobs[name] = np.memmap(d / "blob.bin", dtype=np.uint8, mode="r")
            self.offsets[name] = np.load(d / "offsets.npy")
            for i, k in enumerate(np.load(d / "keys.npy")):
                self.index[str(k)] = (name, i)

    def has(self, v1_row, key):
        return (self.v1 is not None and v1_row >= 0) or key in self.index

    def jpegs(self, v1_row, key):
        if self.v1 is not None and v1_row >= 0:
            o, b = self.v1["offsets"][v1_row], self.v1["blob"]
        else:
            name, i = self.index[key]
            o, b = self.offsets[name][i], self.blobs[name]
        return [b[o[t]:o[t + 1]] for t in range(len(o) - 1)]
