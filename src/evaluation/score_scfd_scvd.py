#!/usr/bin/env python3
"""Score every model on the fresh confirmatory sets (PREREG deviation 6). Writes fresh/preds/<run>.csv with one row per
fresh window (same order as $LENSNET_ROOT/workspace/features/l14/fresh/index.csv) and column p_violent. Also writes
fresh/near_duplicate_exclusion.csv (mean CLIP CLS cosine >= 0.95 to any audited-train or std_rwf-train window).
Usage: score_scfd_scvd.py dedup | lensnet | gpu"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import lensnet_setup as L, lensnet as V
ROOT = L.ROOT; FR = ROOT / "workspace/fresh"; P = FR / "preds"; P.mkdir(parents=True, exist_ok=True)
from video_data import window_data as D
import video_data.frame_store as VF


def fresh_df():
    idx = pd.read_csv(L.FEAT / "fresh" / "index.csv")
    w = pd.read_csv(FR / "fresh_windows.csv"); w = w[w.decode_failed == 0].reset_index(drop=True)
    assert (idx.path.values == w.path.values).all() and (idx.start_frame.values == w.start_frame.values).all()
    return w


def dedup():
    man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "split"])
    sp = pd.read_csv(ROOT / "workspace/splits/std_rwf.csv", usecols=["segment_id", "split"])
    train = set(man[man.split == "train"].segment_id) | set(sp[sp.split == "train"].segment_id)
    b = V.Compact("bench"); rows = np.where(man.segment_id.isin(train).values)[0]
    tb = F.normalize(b.cls[rows], dim=-1); tf = F.normalize(V.Compact("fresh").cls, dim=-1)
    mx = torch.cat([(tf @ tb[i:i + 4096].t()).max(1).values.unsqueeze(1) for i in range(0, len(tb), 4096)], 1).max(1).values
    w = fresh_df().assign(max_cos_to_train=mx.numpy(), excluded_near_duplicate=(mx.numpy() >= 0.95).astype(int))
    w.to_csv(FR / "near_duplicate_exclusion.csv", index=False)
    print("excluded:", w.groupby("benchmark").excluded_near_duplicate.sum().to_dict(), "max cos quantiles", np.quantile(mx.numpy(), [0.5, 0.99, 1.0]).round(3))


PT = ("" if L.BACKBONE == "L14" else "b16_") + ("" if L.FRAMES == 16 else f"f{L.FRAMES}_") + ("" if L.PSET == "orig" else f"p{L.PSET}_")  # variant prefix (none for the default ViT-L/14, 16 frames, original prompts)


def lensnet_runs():
    fr = V.Compact("fresh"); rows = np.arange(len(fr.index))
    names = [d.name for d in sorted(L.OUT.glob("v2_audited_*_seed4*")) + sorted(L.OUT.glob("v2_std_std_rwf_*_seed4*"))]
    for n in names:
        s = json.load(open(L.OUT / n / "summary.json")); c = s["config"]
        m = V.V2(c, L.text_anchors()); m.load_state_dict(torch.load(L.OUT / n / "best.pt")); m.eval()
        pd.DataFrame({"p_violent": V.predict(m, fr, rows)}).to_csv(P / f"{PT}{n}.csv", index=False)
    zs = V.V2("LP", L.text_anchors()); zs.eval()          # zero-shot: w = text anchors, no training
    pd.DataFrame({"p_violent": V.predict(zs, fr, rows)}).to_csv(P / f"{PT}ZS.csv", index=False)
    print("LENS-family runs scored:", len(names), "+ zero-shot")


def gpu():
    dev = torch.device("cuda"); w = fresh_df()
    VF.STORE_ROOT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/fresh/frames"); store = VF.FrameStore(stores=("fresh",), use_v1=False)
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
    if SKIP_FT: return
    import clip_tail as C, lensnet_ft as FT
    import extract_clip_tokens as E
    for d in sorted((ROOT / "workspace/runs_ft").glob("LENS-FT_seed*")):
        if not (d / "summary.json").exists(): continue
        ck = torch.load(d / "best.pt", map_location=dev); vh = C.load_clip(dev, "fp16")
        tail = C.Tail(C.load_clip(dev, "fp32")).to(dev); tail.load_state_dict(ck["tail"]); tail.eval()
        head = V.V2("V2b", L.text_anchors()).to(dev); head.load_state_dict(ck["head"]); head.eval()
        f, st, dec = E.load_set("fresh"); dl = DataLoader(E.Windows(f, st, dec), batch_size=4, num_workers=8)
        pe = np.zeros(len(f))
        with torch.no_grad():
            for x, ii in dl:
                B = x.shape[0]
                with torch.autocast("cuda", dtype=torch.float16):
                    t = tail(C.head(vh, x.to(dev).half().flatten(0, 1)).float()).view(B, 16, 17, 768)
                pe[ii.numpy()] = torch.softmax(head(*FT.to_inputs(t)).float(), 1)[:, 1].cpu().numpy()
        pd.DataFrame({"p_violent": pe}).to_csv(P / f"{d.name}.csv", index=False); print("scored", d.name, flush=True)


SKIP_FT = False
if __name__ == "__main__":
    if sys.argv[1] == "gpu-baselines": SKIP_FT = True; gpu()
    else: {"dedup": dedup, "lensnet": lensnet_runs, "gpu": gpu}[sys.argv[1]]()
