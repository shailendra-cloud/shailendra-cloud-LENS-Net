#!/usr/bin/env python3
"""E4 (PREREG_revision.md, Part E): score every Table-V model on the ambiguous windows (amb/amb_windows.csv) with the
scoring code of the original window sets (conf2.score_v2 / conf2.score_gpu). Writes amb/preds/<run>.csv; skips existing.
Usage: score_ambiguous_windows.py cpu | gpu"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import lensnet_setup as L, lensnet as V
ROOT = L.ROOT; A = ROOT / "workspace/amb"; P = A / "preds"; P.mkdir(parents=True, exist_ok=True)


def windows():
    w = pd.read_csv(A / "amb_windows.csv"); return w[w.decode_failed == 0].reset_index(drop=True)


def cpu():
    cf = V.Compact("amb"); rows = np.arange(len(cf.index)); assert len(rows) == len(windows())
    for tag, cfg in (("LENS-M_vrex", "LENS-M"), ("LP", "LP"), ("V2b", "V2b"), ("MP", "MP")):
        for s in (42, 43, 44):
            n = f"v2_audited_{tag}_seed{s}"
            m = V.V2(cfg, L.text_anchors()); m.load_state_dict(torch.load(L.OUT / n / "best.pt", weights_only=True)); m.eval()
            pd.DataFrame({"p_violent": V.predict(m, cf, rows)}).to_csv(P / f"{n}.csv", index=False)
    zs = V.V2("LP", L.text_anchors()); zs.eval(); pd.DataFrame({"p_violent": V.predict(zs, cf, rows)}).to_csv(P / "ZS.csv", index=False)
    print("cpu models scored", flush=True)


def gpu():
    from torch.utils.data import DataLoader
    import video_data.frame_store as VF
    from video_data import window_data as D
    from video_data.evaluate_cnn_bigru import load
    import train_video_baseline as TB, vadclip_run as VR, train_finetuned_on_split as RTS
    VF.STORE_ROOT = ROOT / "raw_data/journal/frames"
    dev = torch.device("cuda"); w = windows(); store = VF.FrameStore(stores=("amb",), use_v1=False)
    def run(model, name, fn=None):
        if (P / f"{name}.csv").exists(): return
        dl = DataLoader(D.WindowDataset(w, store, False, decode="seq"), batch_size=8, num_workers=8, pin_memory=True)
        pd.DataFrame({"p_violent": D.predict(model, dl, dev, tta=True, p_violent_fn=fn)}).to_csv(P / f"{name}.csv", index=False); print("scored", name, flush=True)
    for s in (42, 43, 44):
        m, fn, _ = load("v2_binary", str(ROOT / f"source_benchmark/runs/binary_hybrid_seed{s}/best.pt"), dev); run(m, f"MAST-Net_seed{s}", fn)
    dirs = [ROOT / f"workspace/runs_baselines/{b}_seed{s}" for b in ("A1_videomae", "A1_swin3d_t", "A1_r2plus1d_18", "A1_xclip_lr5e-06",
            "A1_videomae_lr2e-05", "A1_r2plus1d_18_lr2e-05") for s in (42, 43, 44)]
    dirs += [ROOT / f"workspace/runs_review/A1_videomae_lora_lr0.0001_seed{s}" for s in (42, 43, 44)]
    for d in dirs:
        sm = json.load(open(d / "summary.json")); kind = sm["model"]
        m = TB.Net(kind).to(dev)
        if sm.get("lora"): RTS.add_lora(m); m.to(dev)
        m.load_state_dict(torch.load(d / "best.pt", map_location=dev), strict=(kind != "xclip")); m.eval(); run(m, d.name); del m; torch.cuda.empty_cache()
    orig_load = VR.VM.clip.load
    VR.VM.clip.load = lambda n, device: orig_load(VR.CLIPNAME["L14"], device, download_root=str(ROOT / "raw_data/clip_weights"))
    X = np.load(VR.FEATS["L14"] / "amb/cls_frames.npy", mmap_mode="r")
    for s in (42, 43, 44):
        d = ROOT / f"workspace/runs_vadclip/VadCLIP-L14_audited_lr0.0001_seed{s}"
        if (P / f"{d.name}.csv").exists(): continue
        m = VR.VM.CLIPVAD(num_class=2, embed_dim=768, visual_length=16, visual_width=768, visual_head=1, visual_layers=1,
                          attn_window=4, prompt_prefix=10, prompt_postfix=10, device=dev).to(dev)
        r = m.load_state_dict(torch.load(d / "best.pt", weights_only=True), strict=False)
        assert not r.unexpected_keys and all(x.startswith("clipmodel.") for x in r.missing_keys)
        pd.DataFrame({"p_violent": VR.score(m, np.asarray(X), dev)}).to_csv(P / f"{d.name}.csv", index=False); print("scored", d.name, flush=True)


if __name__ == "__main__":
    {"cpu": cpu, "gpu": gpu}[sys.argv[1]]()
