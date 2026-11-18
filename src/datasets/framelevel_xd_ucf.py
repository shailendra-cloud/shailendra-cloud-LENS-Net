#!/usr/bin/env python3
"""Standard frame-level protocol on the full XD-Violence and UCF-Crime test sets (PREREG_journal Sec. 8).
  plan            windows for every test video (1.5-s grid from frame 0 + one end-aligned window) -> framelevel/windows.csv
  build K         frame store 'flK' for chunk K (32 frames at 256 px, JPEG q92) -> framelevel/chunks/flK.csv
  score-lensnet K      LENS runs of the current LENS_BACKBONE/LENS_FRAMES (CPU)
  score-gpu K     fine-tuned P1 baselines, CNN-BiGRU and VadCLIP-L14 (GPU)
  clean K         delete the frame store of chunk K (features and predictions are kept)
Predictions: framelevel/preds/flK/<tag><run>.csv, one row per window of chunks/flK.csv with decode_failed == 0."""
import json, os, re, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd
ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT)); sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import video_data.frame_store as VF
VF.STORE_ROOT = ROOT / "raw_data/journal/frames"
from video_data import label_schema as S
FL = ROOT / "workspace/framelevel"; CH = FL / "chunks"
VP = ROOT / "raw_data"
XD_DIR = VP / "xd_violence/data/video/test_videos"
UCF_BASE = VP / "kagglehub_cache_journal/datasets/minhajuddinmeraj/anomalydetectiondatasetucf/versions/50"
UCF_NORMAL = VP / "kagglehub_cache_journal/datasets/thelazygenius/ucfcrime-only-normal-videos"
XD_ANN, UCF_ANN = VP / "annotations/xd_violence_annotations.txt", VP / "annotations/ucf_temporal_annotation.txt"
CHUNK = 6000
P1_BASELINES = ("A1_videomae", "A1_swin3d_t", "A1_r2plus1d_18", "A1_xclip_lr5e-06")


def ground_truth():
    """Official temporal annotations as frame intervals [a, b) per test video; category per video."""
    gt = {}
    for line in open(XD_ANN):
        p = line.split()
        if p:
            nums = [int(x) for x in p[1:]]
            gt[("XD", p[0][:-4] if p[0].endswith(".mp4") else p[0])] = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
    cat = {}
    for line in open(UCF_ANN):
        p = line.split()
        if len(p) < 4: continue
        nums = [int(x) for x in p[2:]]
        gt[("UCF", p[0][:-4])] = [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2) if nums[i] >= 0 and nums[i + 1] > nums[i]]
        cat[p[0][:-4]] = p[1]
    return gt, cat


def videos():
    gt, cat = ground_truth()
    out = [("XD", f[:-4], XD_DIR / f, "Normal" if f[:-4].endswith("label_A") else "Violent") for f in sorted(os.listdir(XD_DIR)) if f.endswith(".mp4")]
    disk = {}
    for base in (UCF_BASE, UCF_NORMAL):
        for dp, _, fs in os.walk(base):
            for f in fs:
                if f.endswith(".mp4"): disk.setdefault(f[:-4], Path(dp) / f)
    missing = [n for (b, n) in gt if b == "UCF" and n not in disk]
    out += [("UCF", n, disk[n], cat[n]) for (b, n) in sorted(gt) if b == "UCF" and n in disk]
    return out, missing


def plan_one(v):
    bench, name, path, category = v
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or S.FPS_FALLBACK; tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    span = int(round(S.WINDOW_SECONDS * fps))
    ws = [(s, s + span - 1, s) for s in range(0, tot - span + 1, span)]          # (start, end, first frame it owns)
    covered = ws[-1][1] + 1 if ws else 0
    if covered < tot: ws.append((max(tot - span, 0), tot - 1, covered))           # end-aligned window owns only new frames
    return [dict(benchmark=bench, video=name, cluster=f"{bench}/{name}", path=str(path), category=category, fps=fps,
                 total_frames=tot, start_frame=s, end_frame=e, own_from=o) for s, e, o in ws]


def plan():
    vs, missing = videos(); t0 = time.time()
    with ProcessPoolExecutor(16) as ex: rows = [r for pl in ex.map(plan_one, vs) for r in pl]
    df = pd.DataFrame(rows)
    rev = pd.read_csv(ROOT / S.CONTAMINATION_REVIEW)
    cont = {v[:-4] if v.endswith(".mp4") else v for v in rev[rev.visual_verdict == "same_footage"].external_video}   # UCF names carry .mp4
    df["contaminated"] = df.video.isin(cont).astype(int)
    df["chunk"] = np.arange(len(df)) // CHUNK        # windows of one video may span two chunks; scoring is per window
    FL.mkdir(parents=True, exist_ok=True); df.to_csv(FL / "windows.csv", index=False)
    json.dump({"missing_ucf": missing, "n_videos": df.groupby("benchmark").video.nunique().to_dict(),
               "contaminated_videos": sorted(set(df.video[df.contaminated == 1]))}, open(FL / "plan.json", "w"), indent=2)
    print(f"videos {df.groupby('benchmark').video.nunique().to_dict()}, windows {len(df)}, chunks {df.chunk.max() + 1}, "
          f"missing UCF {len(missing)}, contaminated videos {df[df.contaminated == 1].video.nunique()}, {time.time()-t0:.0f}s", flush=True)


def build(k):
    from video_data.build_external_windows import decode_video
    df = pd.read_csv(FL / "windows.csv"); df = df[df.chunk == k].reset_index(drop=True)
    jobs = [(p, [(r.start_frame, r.end_frame) for r in g.itertuples()]) for p, g in df.groupby("path", sort=False)]
    w = VF.StoreWriter(f"fl{k}"); failed = []
    with ProcessPoolExecutor(16) as ex:
        for (p, wins), frames in zip(jobs, ex.map(decode_video, jobs, chunksize=1)):
            for (s, e), fr in zip(wins, frames):
                if fr is None: failed.append((p, s)); continue
                w.add(VF.content_key(p, s, e, decode="seq"), fr)
    w.close(dict(note=f"frame-level XD/UCF chunk {k} (PREREG_journal Sec. 8)"))
    fk = set(failed); df["decode_failed"] = [int((p, s) in fk) for p, s in zip(df.path, df.start_frame)]
    CH.mkdir(parents=True, exist_ok=True); df.to_csv(CH / f"fl{k}.csv", index=False)
    print(f"chunk {k}: windows {len(df)}, decode failures {len(failed)}", flush=True)


def chunk_windows(k):
    w = pd.read_csv(CH / f"fl{k}.csv"); return w[w.decode_failed == 0].reset_index(drop=True)


def score_lensnet(k):
    import torch, lensnet_setup as L, lensnet as V
    assert L.PSET == "orig"
    tag = ("" if L.BACKBONE == "L14" else "b16_") + ("" if L.FRAMES == 16 else f"f{L.FRAMES}_")
    P = FL / "preds" / f"fl{k}"; P.mkdir(parents=True, exist_ok=True)
    cf = V.Compact(f"fl{k}"); rows = np.arange(len(cf.index)); w = chunk_windows(k)
    assert (cf.index.path.values == w.path.values).all() and (cf.index.start_frame.values == w.start_frame.values).all()
    want = ["LENS-M_vrex", "V2b"] if tag == "" else ["LENS-M_vrex"]
    names = [d.name for d in sorted(L.OUT.glob("v2_audited_*_seed4*")) if (d / "summary.json").exists() and any(d.name == f"v2_audited_{c}_{d.name[-6:]}" for c in want)]
    for n in names:
        c = json.load(open(L.OUT / n / "summary.json"))["config"]
        m = V.V2(c, L.text_anchors()); m.load_state_dict(torch.load(L.OUT / n / "best.pt", weights_only=True)); m.eval()
        pd.DataFrame({"p_violent": V.predict(m, cf, rows)}).to_csv(P / f"{tag}{n}.csv", index=False)
    if tag == "":
        zs = V.V2("LP", L.text_anchors()); zs.eval()
        pd.DataFrame({"p_violent": V.predict(zs, cf, rows)}).to_csv(P / "ZS.csv", index=False)
    print(f"chunk {k} [{tag or 'L14'}] LENS-family runs scored: {len(names)}", flush=True)


def score_gpu(k):
    import torch
    from torch.utils.data import DataLoader
    from video_data import window_data as D
    from video_data.evaluate_cnn_bigru import load
    import train_video_baseline as TB, vadclip_run as VR
    dev = torch.device("cuda"); w = chunk_windows(k).assign(binary_label=0)
    P = FL / "preds" / f"fl{k}"; P.mkdir(parents=True, exist_ok=True)
    store = VF.FrameStore(stores=(f"fl{k}",), use_v1=False)
    def run(model, name, fn=None):
        if (P / f"{name}.csv").exists(): return
        dl = DataLoader(D.WindowDataset(w, store, False, decode="seq"), batch_size=8, num_workers=8, pin_memory=True)
        pd.DataFrame({"p_violent": D.predict(model, dl, dev, tta=True, p_violent_fn=fn)}).to_csv(P / f"{name}.csv", index=False)
    for s in (42, 43, 44):
        m, fn, _ = load("cnn_bigru", str(ROOT / f"source_benchmark/runs/binary_hybrid_seed{s}/best.pt"), dev); run(m, f"MAST-Net_seed{s}", fn)
    for base in P1_BASELINES:
        for s in (42, 43, 44):
            d = ROOT / f"workspace/runs_baselines/{base}_seed{s}"
            kind = json.load(open(d / "summary.json"))["model"]
            m = TB.Net(kind).to(dev); m.load_state_dict(torch.load(d / "best.pt", map_location=dev), strict=(kind != "xclip")); m.eval()
            run(m, d.name); del m; torch.cuda.empty_cache()
    orig_load = VR.VM.clip.load
    VR.VM.clip.load = lambda n, device: orig_load(VR.CLIPNAME["L14"], device, download_root=str(ROOT / "raw_data/clip_weights"))
    X = np.load(VR.FEATS["L14"] / f"fl{k}/cls_frames.npy", mmap_mode="r")
    for s in (42, 43, 44):
        d = ROOT / f"workspace/runs_vadclip/VadCLIP-L14_audited_lr0.0001_seed{s}"
        if (P / f"{d.name}.csv").exists(): continue
        m = VR.VM.CLIPVAD(num_class=2, embed_dim=768, visual_length=16, visual_width=768, visual_head=1, visual_layers=1,
                          attn_window=4, prompt_prefix=10, prompt_postfix=10, device=dev).to(dev)
        r = m.load_state_dict(torch.load(d / "best.pt", weights_only=True), strict=False)
        assert not r.unexpected_keys and all(x.startswith("clipmodel.") for x in r.missing_keys)
        pd.DataFrame({"p_violent": VR.score(m, np.asarray(X), dev)}).to_csv(P / f"{d.name}.csv", index=False); del m; torch.cuda.empty_cache()
    print(f"chunk {k}: GPU models scored", flush=True)


def clean(k):
    d = VF.STORE_ROOT / f"fl{k}"
    assert d.parent == ROOT / "raw_data/journal/frames" and re.fullmatch(r"fl\d+", d.name)
    if d.exists(): shutil.rmtree(d); print(f"chunk {k}: frame store removed", flush=True)


def trim(k):
    """After chunk K is fully scored: keep L14 compact.npz, B16 compact_f8.npz and L14 vmae.npy (enough to re-score every
    LENS model used here); remove intermediates no model of this protocol reads again (disk is limited)."""
    if not (FL / "preds" / f"fl{k}" / "DONE").exists(): return
    F = ROOT / "workspace/features"
    for rel in (f"l14/fl{k}/cls_frames.npy", f"l14/fl{k}/compact_f8.npz", f"b16/fl{k}/cls_frames.npy", f"b16/fl{k}/compact.npz", f"b16/fl{k}/vmae.npy"):
        f = F / rel
        if f.exists(): f.unlink()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "plan": plan()
    else: {"build": build, "score-lensnet": score_lensnet, "score-gpu": score_gpu, "clean": clean, "trim": trim}[cmd](int(sys.argv[2]))
