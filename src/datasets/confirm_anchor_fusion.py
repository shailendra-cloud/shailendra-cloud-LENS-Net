#!/usr/bin/env python3
"""LENS-Net-A confirmation (PREREG_journal Sec. 10): XD-Violence training subset (edited footage) and AIE (indoor CCTV).
  plan        windows: consecutive 1.5-s windows from frame 0 + one end-aligned window, at most 64 per video (spread)
  build K     frame store 'c3_K' for chunk K (32 frames at 256 px, JPEG q92)
  clean K     remove the frame store of chunk K (features and predictions are kept)
  dedup       near-duplicate exclusion (mean CLIP CLS cosine >= 0.95 to any training window)
  score       LENS-family models from features + VadCLIP-L14 (from cls_frames)
  analyze     video-level AUC (score = max window probability; secondary mean of top 3) and claims K1-K4
Every model is scored once; labels are read only by `plan` (video labels) and `analyze`."""
import json, re, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd
ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT)); sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import video_data.frame_store as VF
VF.STORE_ROOT = ROOT / "raw_data/journal/frames"
from video_data import label_schema as S
JD = ROOT / "raw_data/journal/conf3"; C3 = ROOT / "workspace/conf3"; P = C3 / "preds"
CHUNK, MAXW = 6000, 64


def aie_videos():
    """AIE clips; the label is the file-name suffix: _assault -> 1, _non_assault -> 0, _alert -> excluded (Sec. 10).
    The file names carry no source-footage id, so each clip is its own cluster (pre-registered fallback)."""
    base = JD / "kaggle/datasets/arnaldovitor/aie-dataset/versions/2/aie"; out = []
    for p in sorted(base.glob("*.mp4")):
        m = re.fullmatch(r"(\d+)_(assault|non_assault|alert)", p.stem); assert m, p.name
        if m.group(2) == "alert": continue
        out.append(dict(benchmark="AIE", video=p.name, cluster=f"AIE/{p.stem}", path=str(p), category=m.group(2), binary_label=int(m.group(2) == "assault")))
    return out


def xd_videos():
    sel = pd.read_csv(C3 / "xd_train_selection.csv")
    return [dict(benchmark="XD-train", video=r.name, cluster="XD/" + r.name.split("__")[0], path=str(JD / "xd" / r.path), category=r.labels,
                 binary_label=int(r.binary_label)) for r in sel.itertuples()]


def plan_one(v):
    from video_data.build_external_windows import spread
    cap = cv2.VideoCapture(v["path"]); fps = cap.get(cv2.CAP_PROP_FPS) or S.FPS_FALLBACK; tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    span = int(round(S.WINDOW_SECONDS * (fps if fps > 1 else S.FPS_FALLBACK)))
    ws = [(s, s + span - 1) for s in range(0, max(tot - span + 1, 0), span)]
    if ws and ws[-1][1] + 1 < tot: ws.append((tot - span, tot - 1))
    if not ws and tot >= 8: ws = [(0, tot - 1)]
    if len(ws) > MAXW: ws = spread(ws, MAXW)
    return [dict(v, fps=fps, total_frames=tot, start_frame=s, end_frame=e) for s, e in ws]


def plan():
    vids = xd_videos() + aie_videos(); t0 = time.time()
    with ProcessPoolExecutor(16) as ex: rows = [r for pl in ex.map(plan_one, vids) for r in pl]
    df = pd.DataFrame(rows); df["chunk"] = np.arange(len(df)) // CHUNK
    C3.mkdir(parents=True, exist_ok=True); df.drop(columns=["binary_label"]).to_csv(C3 / "windows.csv", index=False)
    pd.DataFrame(vids).to_csv(C3 / "videos.csv", index=False)                       # video labels, read only by analyze
    print(f"videos {pd.DataFrame(vids).groupby('benchmark').size().to_dict()}, windows {len(df)}, chunks {df.chunk.max() + 1}, "
          f"videos without window {len(vids) - df.video.nunique()}, {time.time()-t0:.0f}s", flush=True)


def build(k):
    from video_data.build_external_windows import decode_video
    df = pd.read_csv(C3 / "windows.csv"); df = df[df.chunk == k].reset_index(drop=True)
    jobs = [(p, [(r.start_frame, r.end_frame) for r in g.itertuples()]) for p, g in df.groupby("path", sort=False)]
    w = VF.StoreWriter(f"c3_{k}"); failed = []
    with ProcessPoolExecutor(16) as ex:
        for (p, wins), frames in zip(jobs, ex.map(decode_video, jobs, chunksize=1)):
            for (s, e), fr in zip(wins, frames):
                if fr is None: failed.append((p, s)); continue
                w.add(VF.content_key(p, s, e, decode="seq"), fr)
    w.close(dict(note=f"LENS-Net-A confirmation chunk {k} (PREREG_journal Sec. 10)"))
    fk = set(failed); df["decode_failed"] = [int((p, s) in fk) for p, s in zip(df.path, df.start_frame)]
    (C3 / "chunks").mkdir(exist_ok=True); df.to_csv(C3 / f"chunks/c3_{k}.csv", index=False); print(f"chunk {k}: {len(df)} windows, decode failures {len(failed)}", flush=True)


def clean(k):
    d = VF.STORE_ROOT / f"c3_{k}"
    assert d.parent == ROOT / "raw_data/journal/frames" and re.fullmatch(r"c3_\d+", d.name)
    if d.exists(): shutil.rmtree(d)


def chunks():
    n = int(pd.read_csv(C3 / "windows.csv", usecols=["chunk"]).chunk.max()) + 1
    return n, [pd.read_csv(C3 / f"chunks/c3_{k}.csv") for k in range(n)]


def windows_ok():
    n, cs = chunks(); w = pd.concat([c[c.decode_failed == 0] for c in cs], ignore_index=True); return n, w


def dedup():
    import torch, torch.nn.functional as F, lensnet_setup as L, lensnet as V
    man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "split"])
    sp = pd.read_csv(ROOT / "workspace/splits/std_rwf.csv", usecols=["segment_id", "split"])
    train = set(man[man.split == "train"].segment_id) | set(sp[sp.split == "train"].segment_id)
    b = V.Compact("bench"); tb = F.normalize(b.cls[np.where(man.segment_id.isin(train).values)[0]], dim=-1)
    n, w = windows_ok(); mx = []
    for k in range(n):
        tc = F.normalize(V.Compact(f"c3_{k}").cls, dim=-1)
        mx.append(torch.cat([(tc @ tb[i:i + 4096].t()).max(1).values.unsqueeze(1) for i in range(0, len(tb), 4096)], 1).max(1).values.numpy())
    mx = np.concatenate(mx); assert len(mx) == len(w)
    w.assign(max_cos_to_train=mx, excluded_near_duplicate=(mx >= 0.95).astype(int)).to_csv(C3 / "near_duplicate_exclusion.csv", index=False)
    print("excluded windows:", pd.Series(mx >= 0.95).groupby(w.benchmark.values).sum().to_dict(), "of", w.groupby("benchmark").size().to_dict(), flush=True)


def score():
    import torch, lensnet_setup as L, lensnet as V, vadclip_run as VR
    assert L.BACKBONE == "L14" and L.FRAMES == 16 and L.PSET == "orig"
    n, _ = chunks(); P.mkdir(parents=True, exist_ok=True)
    jobs = [(f"v2_{sp}_{tag}_seed{s}", cfg, f"v2_{sp}_{tag}_seed{s}") for sp in ("audited", "std_std_rwf") for s in (42, 43, 44)
            for cfg, tag in (("LENS-M", "LENS-M_vrex"), ("LP", "LP"), ("V2b", "V2b"), ("MP", "MP"), ("LENS-FG", "LENS-FG_vrex"))]
    feats = [V.Compact(f"c3_{k}") for k in range(n)]
    for name, cfg, ck in jobs:
        if not (L.OUT / ck / "best.pt").exists(): continue
        m = V.V2(cfg, L.text_anchors()); m.load_state_dict(torch.load(L.OUT / ck / "best.pt", weights_only=True)); m.eval()
        pd.DataFrame({"p_violent": np.concatenate([V.predict(m, f, np.arange(len(f.index))) for f in feats])}).to_csv(P / f"{name}.csv", index=False)
    zs = V.V2("LP", L.text_anchors()); zs.eval()
    pd.DataFrame({"p_violent": np.concatenate([V.predict(zs, f, np.arange(len(f.index))) for f in feats])}).to_csv(P / "ZS.csv", index=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); orig_load = VR.VM.clip.load
    VR.VM.clip.load = lambda nm, device: orig_load(VR.CLIPNAME["L14"], device, download_root=str(ROOT / "raw_data/clip_weights"))
    X = np.concatenate([np.load(VR.FEATS["L14"] / f"c3_{k}/cls_frames.npy") for k in range(n)])
    for s in (42, 43, 44):
        d = ROOT / f"workspace/runs_vadclip/VadCLIP-L14_audited_lr0.0001_seed{s}"
        m = VR.VM.CLIPVAD(num_class=2, embed_dim=768, visual_length=16, visual_width=768, visual_head=1, visual_layers=1,
                          attn_window=4, prompt_prefix=10, prompt_postfix=10, device=dev).to(dev)
        r = m.load_state_dict(torch.load(d / "best.pt", weights_only=True), strict=False)
        assert not r.unexpected_keys and all(x.startswith("clipmodel.") for x in r.missing_keys)
        pd.DataFrame({"p_violent": VR.score(m, X, dev)}).to_csv(P / f"{d.name}.csv", index=False)
    print("scored", len(jobs), "LENS-family runs + ZS + VadCLIP", flush=True)


def analyze():
    from sklearn.metrics import roc_auc_score
    _, w = windows_ok(); x = pd.read_csv(C3 / "near_duplicate_exclusion.csv")
    assert (x.path.values == w.path.values).all() and (x.start_frame.values == w.start_frame.values).all()
    keep = x.excluded_near_duplicate.values == 0
    vid = pd.read_csv(C3 / "videos.csv").set_index("video")
    B = ["XD-train", "AIE"]; auc = lambda y, p: 100 * roc_auc_score(y, p)
    def load(name): return pd.read_csv(P / f"{name}.csv").p_violent.values
    def video_scores(p, how):
        d = pd.DataFrame({"video": w.video.values[keep], "p": p[keep]})
        g = d.groupby("video").p
        s = g.max() if how == "max" else g.apply(lambda v: np.sort(v.values)[-3:].mean())
        return s
    groups = {}
    for f in sorted(P.glob("*.csv")): groups.setdefault(re.sub(r"_seed4[234]$", "", f.stem), []).append(f.stem)
    groups["LENS-Net-A"] = [f"A{s}" for s in (42, 43, 44)]; groups["LENS-Net-A (P3)"] = [f"AT{s}" for s in (42, 43, 44)]
    cache = {}
    def preds(n):
        if n not in cache:
            if n.startswith("AT"): cache[n] = (load(f"v2_std_std_rwf_LP_seed{n[2:]}") + load(f"v2_std_std_rwf_LENS-M_vrex_seed{n[2:]}")) / 2
            elif n.startswith("A"): cache[n] = (load(f"v2_audited_LP_seed{n[1:]}") + load(f"v2_audited_LENS-M_vrex_seed{n[1:]}")) / 2
            else: cache[n] = load(n)
        return cache[n]
    res = {"n_videos": {}, "n_windows": {b: int(((w.benchmark == b).values & keep).sum()) for b in B},
           "n_excluded_windows": {b: int(((w.benchmark == b).values & ~keep).sum()) for b in B}, "models": {}}
    for how in ("max", "top3"):
        avg = {}
        for g, members in groups.items():
            vs = [video_scores(preds(m), how) for m in members]; avg[g] = sum(vs) / len(vs); row = {"seeds": len(members)}
            for b in B:
                ids = [v for v in vs[0].index if vid.loc[v, "benchmark"] == b]; y = vid.loc[ids, "binary_label"].values
                vals = [auc(y, s.loc[ids].values) for s in vs]
                row[b] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
                res["n_videos"][b] = len(ids)
            res["models"].setdefault(how, {})[g] = row
        def boot(a, c, b):
            ids = [v for v in avg[a].index if vid.loc[v, "benchmark"] == b]; y = vid.loc[ids, "binary_label"].values
            pa, pc = avg[a].loc[ids].values, avg[c].loc[ids].values; cl = vid.loc[ids, "cluster"].values
            idx = pd.Series(np.arange(len(ids))).groupby(cl).apply(np.array).values; rng = np.random.default_rng(0); ds = []
            for _ in range(2000):
                i = np.concatenate(idx[rng.integers(0, len(idx), len(idx))])
                if len(np.unique(y[i])) == 2: ds.append(auc(y[i], pa[i]) - auc(y[i], pc[i]))
            lo, hi = np.percentile(ds, [2.5, 97.5])
            return dict(diff=float(auc(y, pa) - auc(y, pc)), ci=[float(lo), float(hi)], better=bool(lo > 0), worse=bool(hi < 0), non_inferior=bool(lo > -1.0), n_clusters=int(len(idx)))
        M, Pr = "v2_audited_LENS-M_vrex", "v2_audited_LP"
        cl = {"K1 A - LENS-Net on XD-train (better)": boot("LENS-Net-A", M, "XD-train"), "K2 A - LENS-Net on AIE (non-inferior)": boot("LENS-Net-A", M, "AIE"),
              "K3 A - probe on XD-train (non-inferior)": boot("LENS-Net-A", Pr, "XD-train"), "K4 A - probe on AIE (better)": boot("LENS-Net-A", Pr, "AIE")}
        ok = [cl[k]["better"] if "(better)" in k else cl[k]["non_inferior"] for k in cl]
        res[f"claims_{how}"] = dict(claims=cl, holds=dict(zip(cl, ok)), confirmed=bool(all(ok)))
        res[f"vs_others_{how}"] = {f"A - {c}": {b: boot("LENS-Net-A", c, b) for b in B} for c in ("v2_audited_V2b", "v2_audited_MP", "ZS", "v2_audited_LENS-FG_vrex", "VadCLIP-L14_audited_lr0.0001") if c in avg}
    json.dump(res, open(ROOT / "workspace/logs/conf3_results.json", "w"), indent=2)
    fmt = lambda d: f"{d['diff']:+.2f} [{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}]"
    print("videos", res["n_videos"], "windows", res["n_windows"], "excluded windows", res["n_excluded_windows"])
    for how in ("max", "top3"):
        print(f"=== video score = {how}")
        for g, v in sorted(res["models"][how].items()): print(f"  {g:38s} " + "  ".join(f"{b} {v[b]['mean']:.2f}±{v[b]['sd']:.2f}" for b in B))
        c = res[f"claims_{how}"]; print(f"  --- claims ({'PRIMARY' if how == 'max' else 'secondary'}): CONFIRMED = {c['confirmed']}")
        for k, d in c["claims"].items(): print(f"     {k:44s} {fmt(d)}  holds={c['holds'][k]}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in ("build", "clean"): {"build": build, "clean": clean}[cmd](int(sys.argv[2]))
    else: {"plan": plan, "dedup": dedup, "score": score, "analyze": analyze}[cmd]()
