#!/usr/bin/env python3
"""Fine-structure score (grad128) for every verified near-duplicate window pair, then simulate grouping.

grad128 = Pearson r of Sobel gradient magnitude on 128x128 grey over the intersection of both frames'
content boxes, summarised with the coarse rule (weaker side's 2nd-best matched frame). Checked
against a calibration set of verified pairs (not included) before scoring the edge list.
Read-only apart from data_tables/audit/near_duplicate_edges_fine.csv.
"""
import sys, time
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from video_data import label_schema as S
from video_data.frame_store import FrameStore, content_key

AUD, MAN = ROOT / "data_tables/audit", ROOT / "data_tables/manifests"
SIDE, MIN_FRAC = 128, 0.15


def prep(gray):
    g = cv2.resize(gray, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    rows = np.percentile(g, 90, axis=1) > S.PHASH_CROP_P90
    cols = np.percentile(g, 90, axis=0) > S.PHASH_CROP_P90
    box = (0, 0, 0, 0)
    if rows.any() and cols.any():
        ri, ci = np.where(rows)[0], np.where(cols)[0]
        box = (ri[0], ri[-1] + 1, ci[0], ci[-1] + 1)
    f = cv2.GaussianBlur(g.astype(np.float32), (3, 3), 0)
    return cv2.magnitude(cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1)).astype(np.float16), box


def box_masks(boxes):
    """boxes (N,F,4) -> row (N,F,SIDE) and col (N,F,SIDE) bool masks."""
    ar = np.arange(SIDE)
    r = (ar >= boxes[..., 0:1]) & (ar < boxes[..., 1:2])
    c = (ar >= boxes[..., 2:3]) & (ar < boxes[..., 3:4])
    return r, c


def grad_scores(GA, BA, GB, BB, batch=256):
    """GA/GB (P,F,S,S) float16, BA/BB (P,F,4). Returns (P,) score."""
    P, F = GA.shape[:2]; out = np.empty(P, np.float32)
    for s in range(0, P, batch):
        e = min(P, s + batch)
        a = GA[s:e].astype(np.float32).reshape(e - s, F, 1, -1); b = GB[s:e].astype(np.float32).reshape(e - s, 1, F, -1)
        ra, ca = box_masks(BA[s:e]); rb, cb = box_masks(BB[s:e])
        rows = ra[:, :, None, :] & rb[:, None, :, :]; cols = ca[:, :, None, :] & cb[:, None, :, :]
        m = (rows[..., :, None] & cols[..., None, :]).reshape(e - s, F, F, -1).astype(np.float32)
        n = m.sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ma = (a * m).sum(-1) / n; mb = (b * m).sum(-1) / n
            va = (a * a * m).sum(-1) / n - ma ** 2; vb = (b * b * m).sum(-1) / n - mb ** 2
            r = ((a * b * m).sum(-1) / n - ma * mb) / np.sqrt(va * vb)
        ok = (n >= MIN_FRAC * SIDE * SIDE) & (va > 1.0) & (vb > 1.0)
        r = np.where(ok, np.nan_to_num(r, nan=-1.0), -1.0)
        out[s:e] = np.minimum(np.sort(r.max(2), 1)[:, -2], np.sort(r.max(1), 1)[:, -2])
    return out


def main():
    t0 = time.time()
    seg = pd.read_csv(MAN / "source_windows.csv")
    e = pd.read_csv(AUD / "near_duplicate_edges_scored.csv")
    cal = pd.read_csv(AUD / "fine_probe_calibration.csv")
    need = pd.unique(np.concatenate([e.segment_a, e.segment_b, cal.segment_a, cal.segment_b]))
    pos = {s: i for i, s in enumerate(need)}
    s_idx = seg.set_index("segment_id").loc[need]
    store = FrameStore(stores=("supp",), use_v1=True)
    G = np.empty((len(need), 3, SIDE, SIDE), np.float16); B = np.empty((len(need), 3, 4), np.int32)
    for i, r in enumerate(s_idx.itertuples()):
        jp = store.jpegs(int(r.v1_cache_row), content_key(r.path, r.start_frame, r.end_frame, decode="seek"))
        for k, p in enumerate(S.PHASH_FRAME_POSITIONS):
            G[i, k], B[i, k] = prep(cv2.imdecode(np.asarray(jp[p]), cv2.IMREAD_GRAYSCALE))
        if i % 5000 == 0: print(f"  decoded {i}/{len(need)} ({time.time() - t0:.0f}s)", flush=True)

    ia = np.array([pos[s] for s in cal.segment_a]); ib = np.array([pos[s] for s in cal.segment_b])
    chk = grad_scores(G[ia], B[ia], G[ib], B[ib])
    dev = np.abs(chk - cal.grad128.values).max()
    print(f"calibration consistency with probe: max |diff| {dev:.4f}", flush=True)
    assert dev < 0.02, "vectorised grad128 disagrees with the probe"

    ia = np.array([pos[s] for s in e.segment_a]); ib = np.array([pos[s] for s in e.segment_b])
    e["grad128"] = np.concatenate([grad_scores(G[ia[s:s + 20000]], B[ia[s:s + 20000]], G[ib[s:s + 20000]], B[ib[s:s + 20000]])
                                   for s in range(0, len(e), 20000)])
    e.to_csv(AUD / "near_duplicate_edges_fine.csv", index=False)
    print(f"scored {len(e)} edges ({time.time() - t0:.0f}s)", flush=True)

    cross = e.dataset_a != e.dataset_b
    print("\n== grad128 distribution of verified edges ==")
    for name, mask in (("within-dataset", ~cross), ("cross-dataset", cross)):
        print(f"  {name:<15} n={mask.sum():>6} quantiles(10,25,50,75,90) {np.quantile(e.grad128[mask], [.1, .25, .5, .75, .9]).round(3).tolist()}")
    for d in ("BusViolence", "HockeyFights", "RLVS", "RWF2000", "VID"):
        m = (e.dataset_a == d) & (e.dataset_b == d)
        print(f"  {d:<15} n={m.sum():>6} quantiles {np.quantile(e.grad128[m], [.1, .25, .5, .75, .9]).round(3).tolist()}")

    units = seg.source_unit_id.unique(); nwin = seg.groupby("source_unit_id").size()
    dsw = seg.groupby("dataset").size()
    print("\n== grouping under coarse>=0.67 AND grad128>=tau ==")
    for tau in (-1.0, 0.30, 0.35, 0.40, 0.45, 0.50):
        k = e[e.grad128 >= tau]
        Gr = nx.Graph(); Gr.add_nodes_from(units); Gr.add_edges_from(zip(k.unit_a, k.unit_b))
        comps = list(nx.connected_components(Gr))
        xl = k[k.dataset_a != k.dataset_b].drop_duplicates(["unit_a", "unit_b"])
        line = f"tau={tau:+.2f}: kept {len(k)}/{len(e)} edges, {len(comps)} groups, cross-dataset unit links {len(xl)}"
        print(line)
        for d in ("HockeyFights", "BusViolence", "RLVS", "RWF2000"):
            locked = biggest = 0
            for c in comps:
                du = [u for u in c if u.startswith(d + ":")]
                if not du: continue
                w = int(nwin[du].sum())
                if any(u.startswith("VID:") for u in c): locked += w
                else: biggest = max(biggest, w)
            print(f"     {d:<13} locked train-only {100 * locked / dsw[d]:5.1f}% | largest splittable group {100 * biggest / dsw[d]:5.1f}%")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
