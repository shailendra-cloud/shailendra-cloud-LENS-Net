#!/usr/bin/env python3
"""I/O helpers shared by the source-manifest builder and auditor (read-only access to v1 artefacts).

Near-duplicate detection is two-stage:
  1. candidates : 64-bit dHash of the CONTENT region of each frame (letterbox / pillarbox bars cropped)
  2. verify     : Pearson correlation of 32x32 content-cropped grey thumbnails
A pair matches only if >= PHASH_MIN_MATCHED_FRAMES frames on EACH side are matched in both stages.
Cropping is essential: vertical phone videos with black side bars otherwise hash as
"black | bright strip | black" regardless of content and match each other spuriously.
"""
import hashlib, os, zipfile
import cv2
import numpy as np

from video_data.label_schema import (PHASH_MIN_GRAY_STD, PHASH_CROP_P90, PHASH_CROP_MIN_FRAC, PHASH_THUMB)


def open_v1_cache_mmap(path):
    """Memory-map the arrays inside the uncompressed v1 .npz cache without loading them into RAM."""
    out = {}
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
    with open(path, "rb") as fh:
        for info in infos:
            if info.compress_type != zipfile.ZIP_STORED:
                raise RuntimeError(f"{info.filename} is compressed; cannot memory-map")
            fh.seek(info.header_offset); lh = fh.read(30)
            assert lh[:4] == b"PK\x03\x04"
            fh.seek(info.header_offset + 30 + int.from_bytes(lh[26:28], "little") + int.from_bytes(lh[28:30], "little"))
            major, _ = np.lib.format.read_magic(fh)
            reader = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(fh)
            arr = np.memmap(path, dtype=dtype, mode="r", offset=fh.tell(), shape=shape, order="F" if fortran else "C")
            out[info.filename[:-4]] = arr if info.filename == "blob.npy" else np.array(arr)
    if int(out["offsets"][-1, -1]) != out["blob"].shape[0]:
        raise RuntimeError("offsets do not end at blob size")
    return out


def content_crop(gray):
    """Crop letterbox / pillarbox bars. Returns the content region, or None if too little remains."""
    cols = np.where(np.percentile(gray, 90, axis=0) > PHASH_CROP_P90)[0]
    rows = np.where(np.percentile(gray, 90, axis=1) > PHASH_CROP_P90)[0]
    if len(cols) == 0 or len(rows) == 0:
        return None
    c = gray[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return c if c.size >= PHASH_CROP_MIN_FRAC * gray.size else None


def frame_signature(bgr):
    """(dhash64 of content, grey std of content, z-normalised float16 thumbnail of content)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    c = content_crop(g)
    if c is None:
        return 0, 0.0, np.zeros((PHASH_THUMB, PHASH_THUMB), np.float16)
    s = cv2.resize(c, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).astype(np.uint8).flatten()
    h = int.from_bytes(np.packbits(bits).tobytes(), "big")
    t = cv2.resize(c, (PHASH_THUMB, PHASH_THUMB), interpolation=cv2.INTER_AREA).astype(np.float32)
    t = (t - t.mean()) / (t.std() + 1e-6)
    return h, float(c.std()), t.astype(np.float16)


def hash_is_valid(std):
    return std >= PHASH_MIN_GRAY_STD


def probe_video(path):
    """(exists, readable, total_frames, fps)."""
    if not os.path.isfile(path):
        return False, False, 0, 0.0
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if ok else 0.0
    cap.release()
    return True, ok and n > 0, n, fps


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def window_frame_indices(start_f, end_f, total, n_frames=32):
    """Frame indices the v1 sampler reads for a window (MAST_NET._cache_one_video)."""
    lo = max(0, int(start_f)); hi = min(int(end_f), max(total - 1, 0))
    if hi < lo: lo, hi = 0, max(total - 1, 0)
    return np.linspace(lo, hi, n_frames).round().astype(int)


def signatures_from_video(path, start_f, end_f, total, positions, cache_size=256, jpeg_quality=92):
    """Signatures of selected sampler positions, via the cache's resize + JPEG round-trip."""
    idx = window_frame_indices(start_f, end_f, total)
    cap = cv2.VideoCapture(str(path)); out = []
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    for p in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx[p])); ok, fr = cap.read()
        if not ok:
            out.append((0, 0.0, np.zeros((PHASH_THUMB, PHASH_THUMB), np.float16))); continue
        fr = cv2.resize(fr, (cache_size, cache_size), interpolation=cv2.INTER_AREA)
        _, e = cv2.imencode(".jpg", fr, enc)
        out.append(frame_signature(cv2.imdecode(e, cv2.IMREAD_COLOR)))
    cap.release()
    return out


def signatures_from_cache(cache, row, positions):
    out = []
    for p in positions:
        b = cache["blob"][cache["offsets"][row, p]:cache["offsets"][row, p + 1]]
        out.append(frame_signature(cv2.imdecode(np.asarray(b), cv2.IMREAD_COLOR)))
    return out


def _candidates(HA, VA, HB, VB, max_hamming, min_frames, chunk, keep_fn):
    Hj = HB[None, :, None, :]; Vj = VB[None, :, None, :]
    out = []
    for a in range(0, len(HA), chunk):
        b = min(len(HA), a + chunk)
        ok = (np.bitwise_count(HA[a:b, None, :, None] ^ Hj) <= max_hamming) & VA[a:b, None, :, None] & Vj
        hit = np.minimum(ok.any(axis=3).sum(axis=2), ok.any(axis=2).sum(axis=2)) >= min_frames
        hit &= keep_fn(a, b)
        ri, rj = np.nonzero(hit)
        out.extend(zip((ri + a).tolist(), rj.tolist()))
    return out


def near_duplicate_pairs(H, V, groups, max_hamming, min_frames, chunk=48, only_cross=None):
    """Stage-1 candidates within one set: pairs i<j in different groups (and, optionally, different
    `only_cross` labels) with >= min_frames dHash-matched frames on each side."""
    N = len(H)
    def keep(a, b):
        m = (np.arange(N)[None, :] > np.arange(a, b)[:, None]) & (groups[a:b, None] != groups[None, :])
        if only_cross is not None:
            m &= only_cross[a:b, None] != only_cross[None, :]
        return m
    return _candidates(H, V, H, V, max_hamming, min_frames, chunk, keep)


def near_duplicate_pairs_between(HA, VA, HB, VB, max_hamming, min_frames, chunk=32):
    """Stage-1 candidates between two different sets (e.g. external benchmark vs manifest)."""
    return _candidates(HA, VA, HB, VB, max_hamming, min_frames, chunk, lambda a, b: True)


def verify_pairs(pairs, THA, VA, THB=None, VB=None, min_r=0.9, min_frames=2, batch=4000):
    """Stage-2: keep pairs whose content thumbnails correlate >= min_r on >= min_frames frames per side."""
    THB = THA if THB is None else THB; VB = VA if VB is None else VB
    keep = []
    for s in range(0, len(pairs), batch):
        chunk = pairs[s:s + batch]
        if not chunk: break
        ia = np.array([p[0] for p in chunk]); ib = np.array([p[1] for p in chunk])
        A = THA[ia].reshape(len(chunk), THA.shape[1], -1).astype(np.float32)
        B = THB[ib].reshape(len(chunk), THB.shape[1], -1).astype(np.float32)
        r = np.einsum("pfk,pgk->pfg", A, B) / A.shape[2]
        ok = (r >= min_r) & VA[ia][:, :, None] & VB[ib][:, None, :]
        good = np.minimum(ok.any(axis=2).sum(axis=1), ok.any(axis=1).sum(axis=1)) >= min_frames
        keep.extend(p for p, g in zip(chunk, good) if g)
    return keep


# ============================================================================================
#  Revised matching (calibration): union candidates + content-intersection verification
# ============================================================================================
G_SIDE = 64


def content_box(gray):
    """Row/column masks of picture content (letterbox / pillarbox bars excluded)."""
    rows = np.percentile(gray, 90, axis=1) > PHASH_CROP_P90
    cols = np.percentile(gray, 90, axis=0) > PHASH_CROP_P90
    if rows.any() and cols.any():
        r = np.zeros_like(rows); c = np.zeros_like(cols)
        ri, ci = np.where(rows)[0], np.where(cols)[0]
        r[ri[0]:ri[-1] + 1] = True; c[ci[0]:ci[-1] + 1] = True
        return r, c
    return np.zeros_like(rows), np.zeros_like(cols)


def frame_record(bgr):
    """(dhash whole frame, dhash content, grey std whole frame, grey std content, 64x64 grey, row mask, col mask)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    def dh(img):
        s_ = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
        return int.from_bytes(np.packbits((s_[:, 1:] > s_[:, :-1]).astype(np.uint8).flatten()).tobytes(), "big")
    g64 = cv2.resize(g, (G_SIDE, G_SIDE), interpolation=cv2.INTER_AREA)
    r, c = content_box(g64)
    crop = content_crop(g)
    return (dh(g), dh(crop) if crop is not None else 0, float(g.std()),
            float(crop.std()) if crop is not None else 0.0, g64, r, c)


def _empty_record():
    z = np.zeros(G_SIDE, bool)
    return (0, 0, 0.0, 0.0, np.zeros((G_SIDE, G_SIDE), np.uint8), z, z)


def records_from_cache(cache, row, positions):
    out = []
    for p in positions:
        b = cache["blob"][cache["offsets"][row, p]:cache["offsets"][row, p + 1]]
        out.append(frame_record(cv2.imdecode(np.asarray(b), cv2.IMREAD_COLOR)))
    return out


def records_from_video(path, start_f, end_f, total, positions, cache_size=256, jpeg_quality=92, mode="seek"):
    """Records for sampler positions via the cache's resize + JPEG round-trip.

    mode="seek" reproduces the v1 sampler exactly (use for windows whose frames live in the v1 cache);
    mode="sequential" decodes forward without seeking, which is deterministic for codecs whose seeks
    are not frame-accurate.
    """
    idx = window_frame_indices(start_f, end_f, total)
    want = [int(idx[p]) for p in positions]
    cap = cv2.VideoCapture(str(path)); frames = {}
    if mode == "seek":
        for k in want:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, fr = cap.read()
            frames[k] = fr if ok else None
    else:
        k = 0; need = set(want)
        while k <= max(want):
            ok, fr = cap.read()
            if not ok: break
            if k in need: frames[k] = fr
            k += 1
    cap.release()
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]; out = []
    for k in want:
        fr = frames.get(k)
        if fr is None:
            out.append(_empty_record()); continue
        fr = cv2.resize(fr, (cache_size, cache_size), interpolation=cv2.INTER_AREA)
        _, e = cv2.imencode(".jpg", fr, enc)
        out.append(frame_record(cv2.imdecode(e, cv2.IMREAD_COLOR)))
    return out


def stack_records(records):
    """List (per window) of per-frame records -> dict of arrays."""
    return dict(H_full=np.array([[x[0] for x in r] for r in records], np.uint64),
                H_crop=np.array([[x[1] for x in r] for r in records], np.uint64),
                SD_full=np.array([[x[2] for x in r] for r in records], np.float32),
                SD_crop=np.array([[x[3] for x in r] for r in records], np.float32),
                G=np.array([[x[4] for x in r] for r in records], np.uint8),
                R=np.array([[x[5] for x in r] for r in records], bool),
                C=np.array([[x[6] for x in r] for r in records], bool))


def intersection_r(GA, RA, CA, GB, RB, CB, batch=1000, min_frac=None):
    """Per candidate pair, (P,F,F) Pearson r between every frame of A and every frame of B, computed only
    over the intersection of the two frames' content boxes. Inputs are already indexed per pair:
    GA (P,F,64,64) uint8, RA/CA (P,F,64) bool."""
    min_frac = PHASH_CROP_MIN_FRAC if min_frac is None else min_frac
    P, F = GA.shape[:2]; out = np.full((P, F, F), -1.0, np.float32)
    for s in range(0, P, batch):
        e = min(P, s + batch)
        a = GA[s:e].reshape(e - s, F, -1).astype(np.float32); b = GB[s:e].reshape(e - s, F, -1).astype(np.float32)
        rows = RA[s:e][:, :, None, :] & RB[s:e][:, None, :, :]
        cols = CA[s:e][:, :, None, :] & CB[s:e][:, None, :, :]
        m = (rows[..., :, None] & cols[..., None, :]).reshape(e - s, F, F, -1).astype(np.float32)
        n = m.sum(-1)
        sa = np.einsum("pfk,pfgk->pfg", a, m); sb = np.einsum("pgk,pfgk->pfg", b, m)
        saa = np.einsum("pfk,pfgk->pfg", a * a, m); sbb = np.einsum("pgk,pfgk->pfg", b * b, m)
        sab = np.einsum("pfk,pgk,pfgk->pfg", a, b, m)
        with np.errstate(divide="ignore", invalid="ignore"):
            ma, mb = sa / n, sb / n
            va, vb = saa / n - ma ** 2, sbb / n - mb ** 2
            r = (sab / n - ma * mb) / np.sqrt(va * vb)
        ok = (n >= min_frac * GA.shape[2] * GA.shape[3]) & (va > 1.0) & (vb > 1.0)
        out[s:e] = np.where(ok, np.nan_to_num(r, nan=-1.0), -1.0)
    return out


def pair_score(rmat, VA=None, VB=None):
    """Decision statistic: the weaker side's 2nd-best matched-frame correlation (rule needs 2 frames/side)."""
    r = rmat.copy()
    if VA is not None: r[~VA[:, :, None].repeat(r.shape[2], 2)] = -1
    if VB is not None: r[~VB[:, None, :].repeat(r.shape[1], 1)] = -1
    best_a = np.sort(r.max(axis=2), axis=1)[:, -2]
    best_b = np.sort(r.max(axis=1), axis=1)[:, -2]
    return np.minimum(best_a, best_b)



def candidate_pairs(A, groups, max_hamming, min_frames, only_cross=None, B=None):
    """Stage 1: union of whole-frame and content-cropped dHash candidates.

    Whole-frame hashes catch dark / overlaid CCTV whose content box varies; cropped hashes catch
    letterboxed material. Within one set (B is None) pairs are i<j in different groups; between sets
    (B given) every (i in A, j in B) pair is considered.
    """
    from video_data.label_schema import PHASH_MIN_GRAY_STD as T
    out = set()
    for hk, sk in (("H_full", "SD_full"), ("H_crop", "SD_crop")):
        va = A[sk] >= T
        if B is None:
            out.update(near_duplicate_pairs(A[hk], va, groups, max_hamming, min_frames, only_cross=only_cross))
        else:
            out.update(near_duplicate_pairs_between(A[hk], va, B[hk], B[sk] >= T, max_hamming, min_frames))
    return sorted(out)


def verify_intersection(pairs, A, min_r, min_frames, B=None, batch=1000):
    """Stage 2: keep pairs with >= min_frames frames per side correlating >= min_r over shared content."""
    B = A if B is None else B
    keep = []
    for s_ in range(0, len(pairs), batch):
        chunk = pairs[s_:s_ + batch]
        ia = np.array([p[0] for p in chunk]); ib = np.array([p[1] for p in chunk])
        r = intersection_r(A["G"][ia], A["R"][ia], A["C"][ia], B["G"][ib], B["R"][ib], B["C"][ib])
        ok = r >= min_r
        good = np.minimum(ok.any(axis=2).sum(axis=1), ok.any(axis=1).sum(axis=1)) >= min_frames
        keep.extend(p for p, g in zip(chunk, good) if g)
    return keep



def near_duplicates_verified(A, groups, cand_hamming, cand_min_frames, min_r, min_frames,
                             only_cross=None, B=None, chunk=32, verify_batch=1000, log=None):
    """Two-stage search with verification applied chunk by chunk, so memory stays bounded however
    permissive the candidate rule is. Candidates: whole-frame OR content dHash within cand_hamming on
    >= cand_min_frames frames per side. Verification: verify_intersection (>= min_frames per side,
    content-intersection r >= min_r). Returns (n_candidates, verified_pairs)."""
    from video_data.label_schema import PHASH_MIN_GRAY_STD as T
    Bs = A if B is None else B
    NA, NB = len(A["H_full"]), len(Bs["H_full"])
    VfA, VcA = A["SD_full"] >= T, A["SD_crop"] >= T
    HfB, HcB = Bs["H_full"][None, :, None, :], Bs["H_crop"][None, :, None, :]
    VfB, VcB = (Bs["SD_full"] >= T)[None, :, None, :], (Bs["SD_crop"] >= T)[None, :, None, :]
    n_cand, keep = 0, []
    for c, a in enumerate(range(0, NA, chunk)):
        b = min(NA, a + chunk)
        okf = (np.bitwise_count(A["H_full"][a:b, None, :, None] ^ HfB) <= cand_hamming) & VfA[a:b, None, :, None] & VfB
        okc = (np.bitwise_count(A["H_crop"][a:b, None, :, None] ^ HcB) <= cand_hamming) & VcA[a:b, None, :, None] & VcB
        hit = ((np.minimum(okf.any(3).sum(2), okf.any(2).sum(2)) >= cand_min_frames) |
               (np.minimum(okc.any(3).sum(2), okc.any(2).sum(2)) >= cand_min_frames))
        if B is None:
            hit &= np.arange(NB)[None, :] > np.arange(a, b)[:, None]
            hit &= groups[a:b, None] != groups[None, :]
            if only_cross is not None:
                hit &= only_cross[a:b, None] != only_cross[None, :]
        ri, rj = np.nonzero(hit)
        if len(ri):
            pairs = list(zip((ri + a).tolist(), rj.tolist())); n_cand += len(pairs)
            keep.extend(verify_intersection(pairs, A, min_r, min_frames, B=B, batch=verify_batch))
        if log is not None and c % 150 == 0:
            log(f"      ... {b}/{NA} rows | candidates {n_cand} | verified {len(keep)}")
    return n_cand, keep


# ---------------------------------------------------------------- stage 3: fine-structure verification
def fine_frame(gray):
    """128x128 Sobel gradient magnitude (float16) and content box (r0, r1, c0, c1) of a grey frame."""
    from video_data.label_schema import PHASH_FINE_SIDE as SIDE
    g = cv2.resize(gray, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    rows = np.percentile(g, 90, axis=1) > PHASH_CROP_P90
    cols = np.percentile(g, 90, axis=0) > PHASH_CROP_P90
    box = (0, 0, 0, 0)
    if rows.any() and cols.any():
        ri, ci = np.where(rows)[0], np.where(cols)[0]
        box = (ri[0], ri[-1] + 1, ci[0], ci[-1] + 1)
    f = cv2.GaussianBlur(g.astype(np.float32), (3, 3), 0)
    grad = cv2.magnitude(cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1))
    return grad.astype(np.float16), np.asarray(box, np.int32)


def _box_masks(boxes, side):
    ar = np.arange(side)
    return ((ar >= boxes[..., 0:1]) & (ar < boxes[..., 1:2]), (ar >= boxes[..., 2:3]) & (ar < boxes[..., 3:4]))


def fine_pair_scores(GA, BA, GB, BB, batch=256):
    """GA/GB (P,F,S,S) gradients, BA/BB (P,F,4) boxes, indexed per pair -> (P,) weaker side's 2nd-best frame r."""
    P, F, side = GA.shape[0], GA.shape[1], GA.shape[2]
    out = np.empty(P, np.float32)
    for s in range(0, P, batch):
        e = min(P, s + batch)
        a = GA[s:e].astype(np.float32).reshape(e - s, F, 1, -1); b = GB[s:e].astype(np.float32).reshape(e - s, 1, F, -1)
        ra, ca = _box_masks(BA[s:e], side); rb, cb = _box_masks(BB[s:e], side)
        rows = ra[:, :, None, :] & rb[:, None, :, :]; cols = ca[:, :, None, :] & cb[:, None, :, :]
        m = (rows[..., :, None] & cols[..., None, :]).reshape(e - s, F, F, -1).astype(np.float32)
        n = m.sum(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ma = (a * m).sum(-1) / n; mb = (b * m).sum(-1) / n
            va = (a * a * m).sum(-1) / n - ma ** 2; vb = (b * b * m).sum(-1) / n - mb ** 2
            r = ((a * b * m).sum(-1) / n - ma * mb) / np.sqrt(va * vb)
        ok = (n >= PHASH_CROP_MIN_FRAC * side * side) & (va > 1.0) & (vb > 1.0)
        r = np.where(ok, np.nan_to_num(r, nan=-1.0), -1.0)
        out[s:e] = np.minimum(np.sort(r.max(2), 1)[:, -2], np.sort(r.max(1), 1)[:, -2])
    return out


def fine_scores(pairs, fine_a, fine_b=None, chunk=20000):
    """Stage-3 scores for (i, j) pairs. fine_a(i) / fine_b(j) return (grad (F,S,S), box (F,4)) for a window;
    each window is computed once. fine_b defaults to fine_a (pairs within one set)."""
    fine_b = fine_a if fine_b is None else fine_b
    if not pairs: return np.zeros(0, np.float32)
    ia = np.array([p[0] for p in pairs]); ib = np.array([p[1] for p in pairs])
    def table(idx, fn):
        u = np.unique(idx); pos = {k: n for n, k in enumerate(u)}
        first = fn(int(u[0]))
        G = np.empty((len(u),) + first[0].shape, np.float16); B = np.empty((len(u),) + first[1].shape, np.int32)
        G[0], B[0] = first
        for n, k in enumerate(u[1:], 1): G[n], B[n] = fn(int(k))
        return G, B, np.array([pos[k] for k in idx])
    if fine_b is fine_a:
        G, B, p = table(np.concatenate([ia, ib]), fine_a); pa, pb = p[:len(ia)], p[len(ia):]; GB_, BB_ = G, B
    else:
        G, B, pa = table(ia, fine_a); GB_, BB_, pb = table(ib, fine_b)
    return np.concatenate([fine_pair_scores(G[pa[s:s + chunk]], B[pa[s:s + chunk]], GB_[pb[s:s + chunk]], BB_[pb[s:s + chunk]])
                           for s in range(0, len(ia), chunk)])


def fine_from_cache(cache, row, positions):
    """Stage-3 inputs for a v1-cache window (same frames as records_from_cache)."""
    fr = [fine_frame(cv2.imdecode(np.asarray(cache["blob"][cache["offsets"][row, p]:cache["offsets"][row, p + 1]]),
                                  cv2.IMREAD_GRAYSCALE)) for p in positions]
    return np.stack([f[0] for f in fr]), np.stack([f[1] for f in fr])


def fine_from_jpegs(jpegs, positions):
    fr = [fine_frame(cv2.imdecode(np.asarray(jpegs[p]), cv2.IMREAD_GRAYSCALE)) for p in positions]
    return np.stack([f[0] for f in fr]), np.stack([f[1] for f in fr])


def fine_from_video(path, start_f, end_f, total, positions, cache_size=256, jpeg_quality=92, mode="seek"):
    """Stage-3 inputs for a window decoded from video: same frames and JPEG round-trip as records_from_video."""
    from video_data.label_schema import PHASH_FINE_SIDE as SIDE
    idx = window_frame_indices(start_f, end_f, total)
    want = [int(idx[p]) for p in positions]
    cap = cv2.VideoCapture(str(path)); frames = {}
    if mode == "seek":
        for k in want:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, fr = cap.read()
            frames[k] = fr if ok else None
    else:
        k = 0; need = set(want)
        while k <= max(want):
            ok, fr = cap.read()
            if not ok: break
            if k in need: frames[k] = fr
            k += 1
    cap.release()
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]; G, B = [], []
    for k in want:
        fr = frames.get(k)
        if fr is None:
            G.append(np.zeros((SIDE, SIDE), np.float16)); B.append(np.zeros(4, np.int32)); continue
        _, e = cv2.imencode(".jpg", cv2.resize(fr, (cache_size, cache_size), interpolation=cv2.INTER_AREA), enc)
        g, b = fine_frame(cv2.imdecode(e, cv2.IMREAD_GRAYSCALE)); G.append(g); B.append(b)
    return np.stack(G), np.stack(B)
