#!/usr/bin/env python3
"""Robustness under frame corruptions (PREREG_journal Sec. 5; descriptive). One wrapper around the frame store corrupts
the stored 256-px frames identically for every model (PNG re-encode, except JPEG). For each (set, condition) it writes
robust/preds/<set>/<cond>/<model>_seed<s>.csv (column p_violent, row order of the set's window table).
Usage: robustness.py <set: test|viop> <cond> [<cond> ...]   (cond: clean, blur1..3, dark1..3, noise1..3, jpeg1..3)"""
import hashlib, json, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import lensnet_setup as L, lensnet as V, extract_clip_tokens as E, train_video_baseline as TB
assert L.BACKBONE == "L14" and L.FRAMES == 16
ROOT = L.ROOT; sys.path.insert(0, str(ROOT))
from video_data import window_data as D
import video_data.frame_store as VF
OUT = ROOT / "workspace/robust"
SEV = {"blur": (1, 2, 3), "dark": (0.5, 0.3, 0.15), "noise": (8, 16, 32), "jpeg": (30, 15, 5)}


def corrupt(img, kind, s, seed):
    if kind == "blur": return cv2.GaussianBlur(img, (0, 0), sigmaX=s)
    if kind == "dark": return np.clip(img.astype(np.float32) * s, 0, 255).astype(np.uint8)
    if kind == "noise":
        r = np.random.default_rng(seed); return np.clip(img.astype(np.float32) + r.normal(0, s, img.shape), 0, 255).astype(np.uint8)
    raise ValueError(kind)


class CorruptStore:
    def __init__(self, store, cond):
        self.s, self.cond = store, cond
        self.kind, self.sev = (None, None) if cond == "clean" else (cond[:-1], SEV[cond[:-1]][int(cond[-1]) - 1])
    def has(self, r, k): return self.s.has(r, k)
    def jpegs(self, r, k):
        jp = self.s.jpegs(r, k)
        if self.kind is None: return jp
        out = []
        for t, b in enumerate(jp):
            img = cv2.imdecode(np.frombuffer(b, np.uint8) if isinstance(b, (bytes, bytearray)) else np.asarray(b), cv2.IMREAD_COLOR)
            if self.kind == "jpeg":
                out.append(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.sev])[1]); continue
            seed = int(hashlib.md5(f"{k}|{r}|{t}".encode()).hexdigest()[:8], 16)
            out.append(cv2.imencode(".png", corrupt(img, self.kind, self.sev, seed))[1])
        return out


def load(setname):
    if setname == "test":
        man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv")
        te = man[man.split == "test"].reset_index(drop=True)
        return te, VF.FrameStore(stores=("supp",), use_v1=True), "seek", "component_id"
    VF.STORE_ROOT = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/frames")
    w = pd.read_csv(ROOT / "workspace/conf/conf_windows.csv")
    w = w[(w.decode_failed == 0) & (w.benchmark == "VioPeru")].reset_index(drop=True)
    return w, VF.FrameStore(stores=("conf",), use_v1=False), "seq", "video"


def main():
    setname, conds = sys.argv[1], sys.argv[2:]
    df, base, dec, _ = load(setname); dev = torch.device("cuda")
    import open_clip
    from transformers import VideoMAEForVideoClassification
    clip = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai", device=dev, precision="fp16")[0].visual.eval()
    assert type(clip.transformer.resblocks[0].mlp.gelu).__name__ == "QuickGELU"; clip.output_tokens = True
    vm = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").to(dev).half().eval()
    PE = torch.tensor(np.load(L.FEAT / "text_prompts.npy")).to(dev)            # (10,768), as in make_compact
    anchors = L.text_anchors()
    heads = {}
    for cfg, nm in (("LENS-M", "LENS-M_vrex"), ("V2b", "V2b"), ("MP", "MP")):
        for s in (42, 43, 44):
            m = V.V2(cfg, anchors); m.load_state_dict(torch.load(L.OUT / f"v2_audited_{nm}_seed{s}" / "best.pt", weights_only=True)); heads[f"{cfg}_seed{s}"] = m.eval()
    for cond in conds:
        od = OUT / "preds" / setname / cond; od.mkdir(parents=True, exist_ok=True)
        store = CorruptStore(base, cond)
        # ---- LENS features from corrupted frames (same transforms as extract_clip_tokens / make_compact / extract_vmae)
        N = len(df); cls = torch.zeros(N, 768); S = torch.zeros(N, 16, 17, 10); Mv = torch.zeros(N, 768)
        with torch.no_grad():
            for x, idx in DataLoader(E.Windows(df, store, dec), batch_size=8, num_workers=8):
                B = x.shape[0]; x = x.to(dev).half().flatten(0, 1)
                c, pt = clip(x); g = pt.view(-1, 16, 16, 1024).permute(0, 3, 1, 2)
                g = F.adaptive_avg_pool2d(g.float(), 4).flatten(2).transpose(1, 2).half() @ clip.proj
                z = F.normalize(torch.cat([c.unsqueeze(1), g], 1).view(B, 16, 17, 768).half().float(), dim=-1)
                cls[idx] = z[:, :, 0].mean(1).cpu(); S[idx] = (z @ PE.t()).half().float().cpu()
            for x, _, i in DataLoader(D.WindowDataset(df.assign(binary_label=df.binary_label), store, False, decode=dec), batch_size=16, num_workers=8):
                h = vm.videomae(pixel_values=x.to(dev).half())[0]; Mv[i] = vm.fc_norm(h.mean(1)).float().cpu()
            for k, m in heads.items():
                pd.DataFrame({"p_violent": torch.softmax(m(cls, S, Mv), 1)[:, 1].numpy()}).to_csv(od / f"{k}.csv", index=False)
        # ---- end-to-end baselines on the same corrupted frames
        for kind in ("videomae", "swin3d_t", "r2plus1d_18"):
            for s in (42, 43, 44):
                net = TB.Net(kind).to(dev); net.load_state_dict(torch.load(ROOT / f"workspace/runs_baselines/A1_{kind}_seed{s}/best.pt", map_location=dev)); net.eval()
                pd.DataFrame({"p_violent": TB.predict(net, df, store, dev, decode=dec)}).to_csv(od / f"{kind}_seed{s}.csv", index=False)
                del net; torch.cuda.empty_cache()
        from video_data.evaluate_cnn_bigru import load as mload
        for s in (42, 43, 44):
            m, fn, _ = mload("cnn_bigru", str(ROOT / f"source_benchmark/runs/binary_hybrid_seed{s}/best.pt"), dev)
            dl = DataLoader(D.WindowDataset(df, store, False, decode=dec), batch_size=8, num_workers=8, pin_memory=True)
            pd.DataFrame({"p_violent": D.predict(m, dl, dev, tta=True, p_violent_fn=fn)}).to_csv(od / f"CNN-BiGRU_seed{s}.csv", index=False)
        print(setname, cond, "done", flush=True)


if __name__ == "__main__":
    main()
