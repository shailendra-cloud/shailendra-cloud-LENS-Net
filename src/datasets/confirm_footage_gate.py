#!/usr/bin/env python3
"""LENS-FG confirmation on untouched data (PREREG_journal Sec. 9): Movies (edited footage) and UBI-Fights (surveillance).
  build       windows + frame store 'conf2'  (Movies: 1.5-s grid, <= 8 per video, video label;
              UBI-Fights: frame-level annotation; positive if >= 50% of the window is fight, negative if >= 1.0 s
              from every fight frame, <= 4 per class per video, the other class filling free places)
  dedup       near-duplicate exclusion against the training windows (mean CLIP CLS cosine >= 0.95)
  score-lensnet    LENS models from features (CPU): LENS-FG, LENS-FG-post, LENS-Net, w/o motion, LENS-AG, zero-shot
  score-gpu   fine-tuned P1 baselines, CNN-BiGRU, VadCLIP-L14 (GPU)
  analyze     AUC per set and the pre-registered claims C1-C4 -> logs/conf2_results.json
Each model is scored once; labels are read only by `build` and `analyze`."""
import json, re, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd
ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT)); sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import video_data.frame_store as VF
VF.STORE_ROOT = ROOT / "raw_data/journal/frames"
from video_data import label_schema as S
JD = ROOT / "raw_data/journal/conf2"; C2 = ROOT / "workspace/conf2"; P = C2 / "preds"
MOV = JD / "kaggle/datasets/naveenk903/movies-fight-detection-dataset/versions/1/Peliculas"
UBI = JD / "kaggle/datasets/smmohiuddinkhanshiam/ubi-fights/versions/1/UBI_FIGHTS"; UBA = JD / "ubi_annotation/UBI_FIGHTS/annotation"
GUARD_S, PER_CLASS, CAP = 1.0, 4, 8


def plan_movie(a):
    p, lab, cat = a
    from video_data.build_external_windows import spread
    cap = cv2.VideoCapture(str(p)); fps = cap.get(cv2.CAP_PROP_FPS) or S.FPS_FALLBACK; tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    span = int(round(S.WINDOW_SECONDS * (fps if fps > 1 else S.FPS_FALLBACK)))
    ws = [(s, s + span - 1) for s in range(0, max(tot - span + 1, 0), span)]
    if not ws and tot >= 8: ws = [(0, tot - 1)]
    rows = [dict(benchmark="Movies", video=f"{cat}/{p.stem}", cluster=f"Movies/{cat}/{p.stem}", path=str(p), category=cat, fps=fps,
                 total_frames=tot, start_frame=s, end_frame=e, binary_label=lab) for s, e in ws]
    return rows if len(rows) <= 8 else sorted(spread(rows, 8), key=lambda r: r["start_frame"])


def plan_ubi(a):
    p, cat = a
    from video_data.build_external_windows import spread
    y = pd.read_csv(UBA / f"{p.stem}.csv", header=None)[0].values.astype(int)
    cap = cv2.VideoCapture(str(p)); fps = cap.get(cv2.CAP_PROP_FPS) or S.FPS_FALLBACK; cap.release()
    tot = len(y); span = int(round(S.WINDOW_SECONDS * fps)); guard = int(round(GUARD_S * fps))
    grid = [(s, s + span - 1) for s in range(0, max(tot - span + 1, 0), span)]
    near = np.convolve(y, np.ones(2 * guard + 1), mode="same") > 0                      # within 1 s of a fight frame
    pos = [w for w in grid if y[w[0]:w[1] + 1].mean() >= 0.5]
    neg = [w for w in grid if not near[w[0]:w[1] + 1].any()]
    npos = min(len(pos), CAP - min(len(neg), PER_CLASS)); nneg = min(len(neg), CAP - npos)
    rows = [dict(benchmark="UBI-Fights", video=p.stem, cluster=f"UBI/{p.stem}", path=str(p), category=cat, fps=fps, total_frames=tot,
                 start_frame=w[0], end_frame=w[1], binary_label=lab) for lab, ws in ((1, spread(pos, npos)), (0, spread(neg, nneg))) for w in ws]
    return sorted(rows, key=lambda r: r["start_frame"])


def build():
    from video_data.build_external_windows import decode_video
    t0 = time.time()
    mv = [(p, 1, "fights") for p in sorted((MOV / "fights").iterdir())] + [(p, 0, "noFights") for p in sorted((MOV / "noFights").iterdir())]
    ub = [(p, c) for c in ("fight", "normal") for p in sorted((UBI / c).glob("*.mp4"))]
    assert all((UBA / f"{p.stem}.csv").exists() for p, _ in ub)
    with ProcessPoolExecutor(16) as ex: rows = [r for pl in ex.map(plan_movie, mv) for r in pl] + [r for pl in ex.map(plan_ubi, ub) for r in pl]
    df = pd.DataFrame(rows)
    jobs = [(p, [(r.start_frame, r.end_frame) for r in g.itertuples()]) for p, g in df.groupby("path", sort=False)]
    w = VF.StoreWriter("conf2"); failed = []
    with ProcessPoolExecutor(16) as ex:
        for (p, wins), frames in zip(jobs, ex.map(decode_video, jobs, chunksize=2)):
            for (s, e), fr in zip(wins, frames):
                if fr is None: failed.append((p, s)); continue
                w.add(VF.content_key(p, s, e, decode="seq"), fr)
    w.close(dict(note="LENS-FG confirmation: Movies + UBI-Fights (PREREG_journal Sec. 9)"))
    fk = set(failed); df["decode_failed"] = [int((p, s) in fk) for p, s in zip(df.path, df.start_frame)]
    C2.mkdir(parents=True, exist_ok=True); df.to_csv(C2 / "conf2_windows.csv", index=False)
    print(f"videos {len(mv)} + {len(ub)}, windows {len(df)}, decode failures {len(failed)}, {time.time()-t0:.0f}s")
    print(df[df.decode_failed == 0].groupby(["benchmark", "binary_label"]).agg(videos=("video", "nunique"), windows=("path", "size")))


def windows():
    w = pd.read_csv(C2 / "conf2_windows.csv"); return w[w.decode_failed == 0].reset_index(drop=True)


def dedup():
    import torch, torch.nn.functional as F, lensnet_setup as L, lensnet as V
    man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "split"])
    sp = pd.read_csv(ROOT / "workspace/splits/std_rwf.csv", usecols=["segment_id", "split"])
    train = set(man[man.split == "train"].segment_id) | set(sp[sp.split == "train"].segment_id)
    b = V.Compact("bench"); tb = F.normalize(b.cls[np.where(man.segment_id.isin(train).values)[0]], dim=-1)
    c = V.Compact("conf2"); tc = F.normalize(c.cls, dim=-1); w = windows()
    assert (c.index.path.values == w.path.values).all() and (c.index.start_frame.values == w.start_frame.values).all()
    mx = torch.cat([(tc @ tb[i:i + 4096].t()).max(1).values.unsqueeze(1) for i in range(0, len(tb), 4096)], 1).max(1).values.numpy()
    w.drop(columns=["binary_label"]).assign(max_cos_to_train=mx, excluded_near_duplicate=(mx >= 0.95).astype(int)).to_csv(C2 / "near_duplicate_exclusion.csv", index=False)
    print("excluded:", pd.Series(mx >= 0.95).groupby(w.benchmark.values).sum().to_dict(), "of", w.groupby("benchmark").size().to_dict())


def score_lensnet():
    import torch, lensnet_setup as L, lensnet as V
    assert L.BACKBONE == "L14" and L.FRAMES == 16 and L.PSET == "orig"
    P.mkdir(parents=True, exist_ok=True); cf = V.Compact("conf2"); rows = np.arange(len(cf.index)); windows()
    jobs = []
    for sp in ("audited", "std_std_rwf"):
        for s in (42, 43, 44):
            for c, tag in (("LENS-FG", "LENS-FG_vrex"), ("LENS-M", "LENS-M_vrex"), ("V2b", "V2b"), ("LENS-AG", "LENS-AG_vrex")):
                jobs.append((f"v2_{sp}_{tag}_seed{s}", c, f"v2_{sp}_{tag}_seed{s}"))
            jobs.append((f"v2_{sp}_LENS-FGpost_vrex_seed{s}", "LENS-FG", f"v2_{sp}_LENS-M_vrex_seed{s}"))     # gate at inference only
    for name, cfg, ck in jobs:
        m = V.V2(cfg, L.text_anchors()); m.load_state_dict(torch.load(L.OUT / ck / "best.pt", weights_only=True)); m.eval()
        pd.DataFrame({"p_violent": V.predict(m, cf, rows)}).to_csv(P / f"{name}.csv", index=False)
    zs = V.V2("LP", L.text_anchors()); zs.eval(); pd.DataFrame({"p_violent": V.predict(zs, cf, rows)}).to_csv(P / "ZS.csv", index=False)
    fg = V.V2("LENS-FG", L.text_anchors()); pd.DataFrame({"p_raw": fg.p_raw(cf.cls).numpy()}).to_csv(C2 / "p_raw.csv", index=False)
    print("LENS-family models scored:", len(jobs) + 1, flush=True)


def score_gpu():
    import torch
    from torch.utils.data import DataLoader
    from video_data import window_data as D
    from video_data.evaluate_cnn_bigru import load
    import train_video_baseline as TB, vadclip_run as VR
    dev = torch.device("cuda"); w = windows().assign(binary_label=0); P.mkdir(parents=True, exist_ok=True)
    store = VF.FrameStore(stores=("conf2",), use_v1=False)
    def run(model, name, fn=None):
        if (P / f"{name}.csv").exists(): return
        dl = DataLoader(D.WindowDataset(w, store, False, decode="seq"), batch_size=8, num_workers=8, pin_memory=True)
        pd.DataFrame({"p_violent": D.predict(model, dl, dev, tta=True, p_violent_fn=fn)}).to_csv(P / f"{name}.csv", index=False); print("scored", name, flush=True)
    for s in (42, 43, 44):
        m, fn, _ = load("cnn_bigru", str(ROOT / f"source_benchmark/runs/binary_hybrid_seed{s}/best.pt"), dev); run(m, f"MAST-Net_seed{s}", fn)
    for base in ("A1_videomae", "A1_swin3d_t", "A1_r2plus1d_18", "A1_xclip_lr5e-06"):
        for s in (42, 43, 44):
            d = ROOT / f"workspace/runs_baselines/{base}_seed{s}"; kind = json.load(open(d / "summary.json"))["model"]
            m = TB.Net(kind).to(dev); m.load_state_dict(torch.load(d / "best.pt", map_location=dev), strict=(kind != "xclip")); m.eval()
            run(m, d.name); del m; torch.cuda.empty_cache()
    orig_load = VR.VM.clip.load
    VR.VM.clip.load = lambda n, device: orig_load(VR.CLIPNAME["L14"], device, download_root=str(ROOT / "raw_data/clip_weights"))
    X = np.load(VR.FEATS["L14"] / "conf2/cls_frames.npy", mmap_mode="r")
    for s in (42, 43, 44):
        d = ROOT / f"workspace/runs_vadclip/VadCLIP-L14_audited_lr0.0001_seed{s}"
        m = VR.VM.CLIPVAD(num_class=2, embed_dim=768, visual_length=16, visual_width=768, visual_head=1, visual_layers=1,
                          attn_window=4, prompt_prefix=10, prompt_postfix=10, device=dev).to(dev)
        r = m.load_state_dict(torch.load(d / "best.pt", weights_only=True), strict=False)
        assert not r.unexpected_keys and all(x.startswith("clipmodel.") for x in r.missing_keys)
        pd.DataFrame({"p_violent": VR.score(m, np.asarray(X), dev)}).to_csv(P / f"{d.name}.csv", index=False); print("scored", d.name, flush=True)


def score_probes():
    """CLIP probe (LP) and motion probe (MP) checkpoints, scored after the PREREG_journal Sec. 9 analysis for the consolidated table (descriptive)."""
    import torch, lensnet_setup as L, lensnet as V
    cf = V.Compact("conf2"); rows = np.arange(len(cf.index))
    for c in ("LP", "MP"):
        for s in (42, 43, 44):
            n = f"v2_audited_{c}_seed{s}"; m = V.V2(c, L.text_anchors()); m.load_state_dict(torch.load(L.OUT / n / "best.pt", weights_only=True)); m.eval()
            pd.DataFrame({"p_violent": V.predict(m, cf, rows)}).to_csv(P / f"{n}.csv", index=False)
    print("probes scored", flush=True)


def analyze():
    from sklearn.metrics import roc_auc_score
    w = windows(); x = pd.read_csv(C2 / "near_duplicate_exclusion.csv"); assert (x.path.values == w.path.values).all()
    keep = x.excluded_near_duplicate.values == 0; B = ["Movies", "UBI-Fights"]; auc = lambda y, p: 100 * roc_auc_score(y, p)
    pr = pd.read_csv(C2 / "p_raw.csv").p_raw.values
    groups = {}
    for f in sorted(P.glob("*.csv")): groups.setdefault(re.sub(r"_seed4[234]$", "", f.stem), []).append(f)
    res = {"n_windows": {b: int(((w.benchmark == b) & keep).sum()) for b in B}, "n_videos": {b: int(w[(w.benchmark == b) & keep].video.nunique()) for b in B},
           "n_excluded_near_duplicate": {b: int(((w.benchmark == b) & ~keep).sum()) for b in B},
           "p_raw_mean": {**{b: float(pr[(w.benchmark == b).values].mean()) for b in B},
                          **{f"Movies/{c}": float(pr[((w.benchmark == "Movies") & (w.category == c)).values].mean()) for c in ("fights", "noFights")}}, "models": {}}
    avg = {}
    for g, fs in groups.items():
        ps = [pd.read_csv(f).p_violent.values for f in fs]; assert all(len(p) == len(w) for p in ps); avg[g] = np.mean(ps, 0); row = {"seeds": len(fs)}
        for b in B:
            m = keep & (w.benchmark == b).values; v = [auc(w.binary_label.values[m], p[m]) for p in ps]
            row[b] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
        for c in ("fights", "noFights"):        # Movies per class: mean score (the two classes come from different kinds of footage)
            m = keep & ((w.benchmark == "Movies") & (w.category == c)).values; row[f"Movies_mean_score_{c}"] = float(avg[g][m].mean())
        res["models"][g] = row
    def boot(a, b, bench):
        m = keep & (w.benchmark == bench).values; y, pa, pb, c = w.binary_label.values[m], avg[a][m], avg[b][m], w.cluster.values[m]
        cl = pd.Series(np.arange(len(y))).groupby(c).apply(np.array).values; rng = np.random.default_rng(0); ds = []
        for _ in range(2000):
            i = np.concatenate(cl[rng.integers(0, len(cl), len(cl))])
            if len(np.unique(y[i])) == 2: ds.append(auc(y[i], pa[i]) - auc(y[i], pb[i]))
        lo, hi = np.percentile(ds, [2.5, 97.5])
        return dict(diff=float(auc(y, pa) - auc(y, pb)), ci=[float(lo), float(hi)], better=bool(lo > 0), worse=bool(hi < 0), non_inferior=bool(lo > -1.0))
    def claims(fg, m, nm):
        c = {"C1 FG - LENS-Net on Movies (better)": boot(fg, m, "Movies"), "C2 FG - LENS-Net on UBI-Fights (non-inferior)": boot(fg, m, "UBI-Fights"),
             "C3 FG - w/o motion on UBI-Fights (better)": boot(fg, nm, "UBI-Fights"), "C4 FG - w/o motion on Movies (non-inferior)": boot(fg, nm, "Movies")}
        ok = [c[k]["better"] if "(better)" in k else c[k]["non_inferior"] for k in c]
        return dict(claims=c, holds=dict(zip(c, ok)), confirmed=bool(all(ok)))
    M, N = "v2_audited_LENS-M_vrex", "v2_audited_V2b"
    res["primary_LENS-FG_P1"] = claims("v2_audited_LENS-FG_vrex", M, N)
    res["secondary_LENS-FG-post_P1"] = claims("v2_audited_LENS-FGpost_vrex", M, N)
    res["secondary_LENS-FG_P3"] = claims("v2_std_std_rwf_LENS-FG_vrex", "v2_std_std_rwf_LENS-M_vrex", "v2_std_std_rwf_V2b")
    res["lensnet_vs_P1"] = {f"LENS-Net - {c}": {b: boot(M, c, b) for b in B}            # added for the consolidated paired table
                            for c in ("A1_videomae", "A1_swin3d_t", "A1_r2plus1d_18", "A1_xclip_lr5e-06", "VadCLIP-L14_audited_lr0.0001", "MAST-Net",
                                      "v2_audited_V2b", "v2_audited_LP", "v2_audited_MP", "ZS", "v2_audited_LENS-FG_vrex") if c in avg}
    res["vs_baselines_P1"] = {f"LENS-FG - {c}": {b: boot("v2_audited_LENS-FG_vrex", c, b) for b in B}
                              for c in ("A1_videomae", "A1_swin3d_t", "A1_r2plus1d_18", "A1_xclip_lr5e-06", "VadCLIP-L14_audited_lr0.0001", "MAST-Net", "v2_audited_LENS-AG_vrex") if c in avg}
    json.dump(res, open(ROOT / "workspace/logs/conf2_results.json", "w"), indent=2)
    fmt = lambda d: f"{d['diff']:+.2f} [{d['ci'][0]:+.2f},{d['ci'][1]:+.2f}]"
    print("windows", res["n_windows"], "videos", res["n_videos"], "excluded", res["n_excluded_near_duplicate"], "p_raw", {k: round(v, 3) for k, v in res["p_raw_mean"].items()})
    for g, v in sorted(res["models"].items()): print(f"{g:40s} " + "  ".join(f"{b} {v[b]['mean']:.2f}±{v[b]['sd']:.2f}" for b in B))
    for key in ("primary_LENS-FG_P1", "secondary_LENS-FG-post_P1", "secondary_LENS-FG_P3"):
        print(f"--- {key}: CONFIRMED = {res[key]['confirmed']}")
        for k, d in res[key]["claims"].items(): print(f"   {k:48s} {fmt(d)}  holds={res[key]['holds'][k]}")
    print("--- LENS-FG vs baselines (P1)")
    for k, v in res["vs_baselines_P1"].items(): print(f"   {k:44s} " + "  ".join(f"{b} {fmt(d)}{' better' if d['better'] else ''}{' WORSE' if d['worse'] else ''}" for b, d in v.items()))


if __name__ == "__main__":
    {"build": build, "dedup": dedup, "score-lensnet": score_lensnet, "score-gpu": score_gpu, "score-probes": score_probes, "analyze": analyze}[sys.argv[1]]()
