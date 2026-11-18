#!/usr/bin/env python3
"""Fetch only the non-video files (frame-level annotations, split lists) of the official UBI-Fights zip by HTTP range
requests (https://socia-lab.di.ubi.pt/EventDetection/UBI_FIGHTS.zip); the videos come from the Kaggle mirror."""
import io, zipfile, collections, time
from pathlib import Path
import requests, urllib3
urllib3.disable_warnings()
U = "https://socia-lab.di.ubi.pt/EventDetection/UBI_FIGHTS.zip"
OUT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/conf2/ubi_annotation")
S = requests.Session()
def get(**kw):
    for i in range(8):
        try: return S.get(U, verify=False, timeout=180, **kw)
        except Exception as e: print("retry", i, repr(e)[:80], flush=True); time.sleep(10)
    raise SystemExit("server unreachable")
h = S.head(U, verify=False, timeout=180); N = int(h.headers["Content-Length"]); print("size", N, "ranges", h.headers.get("Accept-Ranges"), flush=True)
class R(io.RawIOBase):
    p = 0
    def seekable(s): return True
    def readable(s): return True
    def seek(s, o, w=0): s.p = o if w == 0 else (s.p + o if w == 1 else N + o); return s.p
    def tell(s): return s.p
    def readinto(s, b):
        k = min(len(b), N - s.p)
        if k <= 0: return 0
        c = get(headers={"Range": f"bytes={s.p}-{s.p + k - 1}"}).content; b[:len(c)] = c; s.p += len(c); return len(c)
z = zipfile.ZipFile(io.BufferedReader(R(), buffer_size=4 << 20))
names = z.namelist(); print("entries", len(names), collections.Counter(Path(n).suffix for n in names).most_common(6), flush=True)
small = [n for n in names if not n.endswith("/") and Path(n).suffix.lower() not in (".mp4", ".avi")]
print("non-video files", len(small), small[:5], flush=True)
for i, n in enumerate(small):
    d = OUT / n; d.parent.mkdir(parents=True, exist_ok=True); d.write_bytes(z.read(n))
    if i % 100 == 0: print("fetched", i, flush=True)
print("ANNOTATIONS DONE", len(small), flush=True)
