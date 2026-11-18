#!/usr/bin/env python3
"""Score the fixed checkpoints on CCTV-Fights (PREREG_journal Sec. 7). Each model is scored once; labels are not read.
Writes cfights/preds/<tag><run>.csv (one row per window in features/l14/cfights/index.csv order; column p_violent).
Usage: LENS_BACKBONE=.. LENS_FRAMES=.. score_cctv_fights.py lensnet   (CPU; LENS runs of that backbone/frame setting)
       score_cctv_fights.py gpu                                    (GPU; fine-tuned video baselines, CNN-BiGRU, VadCLIP)"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import lensnet_setup as L, lensnet as V
ROOT = L.ROOT; CF = ROOT / "workspace/cfights"; P = CF / "preds"; P.mkdir(parents=True, exist_ok=True)
from video_data import window_data as D
import video_data.frame_store as VF
TAG = ("" if L.BACKBONE == "L14" else "b16_") + ("" if L.FRAMES == 16 else f"f{L.FRAMES}_")


def windows():
    idx = pd.read_csv(ROOT / "workspace/features/l14/cfights/index.csv")
    w = pd.read_csv(CF / "cfights_windows.csv"); w = w[w.decode_failed == 0].reset_index(drop=True)
    assert (idx.path.values == w.path.values).all() and (idx.start_frame.values == w.start_frame.values).all()
    return w


def lensnet_runs():
    assert L.PSET == "orig"
    cf = V.Compact("cfights"); rows = np.arange(len(cf.index)); windows()
    names = [d.name for d in sorted(L.OUT.glob("v2_audited_*_seed4*")) + sorted(L.OUT.glob("v2_std_std_rwf_*_seed4*"))
             if (d / "summary.json").exists() and not d.name.endswith("_pilot")]
    for n in names:
        c = json.load(open(L.OUT / n / "summary.json"))["config"]
        m = V.V2(c, L.text_anchors()); m.load_state_dict(torch.load(L.OUT / n / "best.pt", weights_only=True)); m.eval()
        pd.DataFrame({"p_violent": V.predict(m, cf, rows)}).to_csv(P / f"{TAG}{n}.csv", index=False)
    if L.BACKBONE == "L14" and L.FRAMES == 16:
        zs = V.V2("LP", L.text_anchors()); zs.eval()
        pd.DataFrame({"p_violent": V.predict(zs, cf, rows)}).to_csv(P / f"{TAG}ZS.csv", index=False)
    print(f"[{TAG or 'L14'}] LENS-family runs scored: {len(names)}", flush=True)


def gpu():
    dev = torch.device("cuda"); w = windows().assign(binary_label=0)          # labels are not used for scoring
    VF.STORE_ROOT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/frames")
    store = VF.FrameStore(stores=("cfights",), use_v1=False)
    def run(model, name, fn=None):
        if (P / f"{name}.csv").exists(): return
        dl = DataLoader(D.WindowDataset(w, store, False, decode="seq"), batch_size=8, num_workers=8, pin_memory=True)
        pd.DataFrame({"p_violent": D.predict(model, dl, dev, tta=True, p_violent_fn=fn)}).to_csv(P / f"{name}.csv", index=False)
        print("scored", name, flush=True)
    sys.path.insert(0, str(ROOT))
    from video_data.evaluate_cnn_bigru import load
    for s in (42, 43, 44):
        m, fn, _ = load("cnn_bigru", str(ROOT / f"source_benchmark/runs/binary_hybrid_seed{s}/best.pt"), dev); run(m, f"MAST-Net_seed{s}", fn)
    import train_video_baseline as TB
    for d in sorted((ROOT / "workspace/runs_baselines").glob("A*_seed*")):
        if not (d / "summary.json").exists(): continue
        kind = json.load(open(d / "summary.json"))["model"]
        m = TB.Net(kind).to(dev); m.load_state_dict(torch.load(d / "best.pt", map_location=dev), strict=(kind != "xclip")); m.eval()
        run(m, d.name); del m; torch.cuda.empty_cache()
    import vadclip_run as VR                                                   # official VadCLIP, lr 1e-4 runs
    orig_load = VR.VM.clip.load
    for d in sorted((ROOT / "workspace/runs_vadclip").glob("VadCLIP-*_lr0.0001_seed*")):
        if not (d / "best.pt").exists() or (P / f"{d.name}.csv").exists(): continue
        bb = d.name.split("_")[0].split("-")[1]
        VR.VM.clip.load = lambda n, device, bb=bb: orig_load(VR.CLIPNAME[bb], device, download_root=str(ROOT / "raw_data/clip_weights"))
        m = VR.VM.CLIPVAD(num_class=2, embed_dim=VR.EMB[bb], visual_length=16, visual_width=VR.EMB[bb], visual_head=1, visual_layers=1,
                          attn_window=4, prompt_prefix=10, prompt_postfix=10, device=dev).to(dev)
        r = m.load_state_dict(torch.load(d / "best.pt", weights_only=True), strict=False)
        assert not r.unexpected_keys and all(k.startswith("clipmodel.") for k in r.missing_keys)   # only frozen CLIP absent
        X = np.load(VR.FEATS[bb] / "cfights/cls_frames.npy", mmap_mode="r")
        pd.DataFrame({"p_violent": VR.score(m, np.asarray(X), dev)}).to_csv(P / f"{d.name}.csv", index=False)
        print("scored", d.name, flush=True); del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    {"lensnet": lensnet_runs, "gpu": gpu}[sys.argv[1]]()
