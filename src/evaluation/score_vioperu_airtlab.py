#!/usr/bin/env python3
"""Score models on the journal confirmation sets (PREREG_journal Sec. 3). Run ONCE, after all E1/E2 runs are final.
Writes conf/preds/<tag><run>.csv (one row per conf window, order of workspace/features/l14/conf/index.csv; column p_violent).
tag = "" for ViT-L/14 16 frames (the default setting), else "b16_", "f8_", "b16_f8_" from LENS_BACKBONE / LENS_FRAMES.
Usage: score_vioperu_airtlab.py lensnet          (CPU; run once per LENS_BACKBONE/LENS_FRAMES setting)
       score_vioperu_airtlab.py gpu-baselines (GPU; fine-tuned video baselines A1/A2 and the CNN-BiGRU detector)"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import lensnet_setup as L, lensnet as V
ROOT = L.ROOT; CF = ROOT / "workspace/conf"; P = CF / "preds"; P.mkdir(parents=True, exist_ok=True)
from video_data import window_data as D
import video_data.frame_store as VF
TAG = ("" if L.BACKBONE == "L14" else "b16_") + ("" if L.FRAMES == 16 else f"f{L.FRAMES}_") + ("" if L.PSET == "orig" else f"p{L.PSET}_")


def conf_df():
    idx = pd.read_csv((__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace/features/l14/conf/index.csv"))
    w = pd.read_csv(CF / "conf_windows.csv"); w = w[w.decode_failed == 0].reset_index(drop=True)
    assert (idx.path.values == w.path.values).all() and (idx.start_frame.values == w.start_frame.values).all()
    return w


def lensnet_runs():
    cf = V.Compact("conf"); rows = np.arange(len(cf.index)); conf_df()
    names = [d.name for d in sorted(L.OUT.glob("v2_audited_*_seed4*")) + sorted(L.OUT.glob("v2_std_std_rwf_*_seed4*"))
             if (d / "summary.json").exists() and not d.name.endswith("_pilot")]
    for n in names:
        c = json.load(open(L.OUT / n / "summary.json"))["config"]
        m = V.V2(c, L.text_anchors()); m.load_state_dict(torch.load(L.OUT / n / "best.pt", weights_only=True)); m.eval()
        pd.DataFrame({"p_violent": V.predict(m, cf, rows)}).to_csv(P / f"{TAG}{n}.csv", index=False)
    if L.BACKBONE == "L14" and L.FRAMES == 16:              # zero-shot per prompt set (ZS.csv = original prompts)
        zs = V.V2("LP", L.text_anchors()); zs.eval()
        pd.DataFrame({"p_violent": V.predict(zs, cf, rows)}).to_csv(P / f"{TAG}ZS.csv", index=False)
    print(f"[{TAG or 'L14'}] LENS-family runs scored: {len(names)}", flush=True)


def gpu():
    dev = torch.device("cuda"); w = conf_df().assign(binary_label=0)      # labels are not used for scoring
    VF.STORE_ROOT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/frames"); store = VF.FrameStore(stores=("conf",), use_v1=False)
    def run(model, name, fn=None):
        dl = DataLoader(D.WindowDataset(w, store, False, decode="seq"), batch_size=8, num_workers=8, pin_memory=True)
        pd.DataFrame({"p_violent": D.predict(model, dl, dev, tta=True, p_violent_fn=fn)}).to_csv(P / f"{name}.csv", index=False)
        print("scored", name, flush=True)
    sys.path.insert(0, str(ROOT))
    from video_data.evaluate_cnn_bigru import load
    for s in (42, 43, 44):
        m, fn, _ = load("cnn_bigru", str(ROOT / f"source_benchmark/runs/binary_hybrid_seed{s}/best.pt"), dev); run(m, f"MAST-Net_seed{s}", fn)
    import train_video_baseline as TB
    for d in sorted((ROOT / "workspace/runs_baselines").glob("A*_seed*")):
        if not (d / "summary.json").exists() or (P / f"{d.name}.csv").exists(): continue
        kind = json.load(open(d / "summary.json"))["model"]
        m = TB.Net(kind).to(dev); m.load_state_dict(torch.load(d / "best.pt", map_location=dev), strict=(kind != "xclip")); m.eval(); run(m, d.name)


if __name__ == "__main__":
    {"lensnet": lensnet_runs, "gpu-baselines": gpu}[sys.argv[1]]()
