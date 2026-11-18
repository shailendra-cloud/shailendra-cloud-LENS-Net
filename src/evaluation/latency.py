#!/usr/bin/env python3
"""PREREG_journal 6C latency: batch 1 window, fp16, one GPU, 10 warm-up then the median of 100 forward passes (CUDA-synchronised),
model forward only (frames already on the GPU). Writes logs/latency.json."""
import json, sys, time, subprocess
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, open_clip
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import lensnet_setup as L, lensnet as V, train_video_baseline as TB
ROOT = L.ROOT; sys.path.insert(0, str(ROOT)); from video_data import cnn_bigru_model as mm
dev = torch.device("cuda"); torch.backends.cudnn.benchmark = True
def timeit(fn, n=100, w=10):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(w): fn()
        torch.cuda.synchronize(); ts = []
        for _ in range(n):
            t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(1000 * (time.perf_counter() - t))
    return float(np.median(ts))
def util(): return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader", "-i", "0"], capture_output=True, text=True).stdout.strip()
out = {"gpu": torch.cuda.get_device_name(0), "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"], capture_output=True, text=True).stdout.strip(),
       "gpu_state_before": util()}
x16 = torch.randn(1, 16, 3, 224, 224, device=dev).half()
from transformers import VideoMAEForVideoClassification
vm = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").to(dev).half().eval()
def clip_vis(name):
    v = open_clip.create_model_and_transforms(name, pretrained="openai", device=dev, precision="fp16")[0].visual.eval(); v.output_tokens = True; return v
for key, name, nf, grid, width in (("LENS-Net (L14, 16f)", "ViT-L-14-quickgelu", 16, 16, 1024), ("L1 (L14, 8f)", "ViT-L-14-quickgelu", 8, 16, 1024),
                                   ("L2 (B16, 16f)", "ViT-B-16-quickgelu", 16, 14, 768), ("L3 (B16, 8f)", "ViT-B-16-quickgelu", 8, 14, 768)):
    v = clip_vis(name); D_ = v.proj.shape[1]; E = F.normalize(torch.randn(10, D_, device=dev), dim=-1)
    head = V.V2("LENS-M", F.normalize(torch.randn(2, D_), dim=-1).numpy()).to(dev).eval()
    def f(v=v, nf=nf, grid=grid, width=width, E=E, head=head):
        c, pt = v(x16[0, :nf])                                                   # CLIP on nf frames
        g = F.adaptive_avg_pool2d(pt.view(-1, grid, grid, width).permute(0, 3, 1, 2).float(), 4).flatten(2).transpose(1, 2).half() @ v.proj
        z = F.normalize(torch.cat([c.unsqueeze(1), g], 1).float(), dim=-1)      # (nf,17,D)
        S = (z @ E.t()).unsqueeze(0); u = z[:, 0].mean(0, keepdim=True)         # PPS input, clip embedding
        m = vm.fc_norm(vm.videomae(pixel_values=x16)[0].mean(1)).float()        # motion stream (16 frames)
        return head(u, S, m)
    out[key] = timeit(f); del v, head; torch.cuda.empty_cache()
for kind, key in (("videomae", "VideoMAE-B FT"), ("swin3d_t", "Video Swin-T FT"), ("r2plus1d_18", "R(2+1)D-18 FT"), ("xclip", "X-CLIP FT")):
    n = TB.Net(kind).to(dev).half().eval(); out[key] = timeit(lambda n=n: n(x16)); del n; torch.cuda.empty_cache()
ck = torch.load(ROOT / "source_benchmark/runs/binary_hybrid_seed42/best.pt", map_location="cpu", weights_only=False); a = ck["arch"]
mn = mm.MASTNet(backbone_kind=a["backbone_kind"], pretrained=False, num_classes=2, use_motion=a["use_motion"], motion_pretrained=a["motion_pretrained"],
                use_interaction=a["use_interaction"], whiten=a["whiten"]).to(dev).half().eval()
out["CNN-BiGRU"] = timeit(lambda: mn(x16)); del mn
vcl = clip_vis("ViT-L-14-quickgelu")                       # VadCLIP (L14): per-frame CLIP image embeddings + temporal/GCN head (text features cacheable)
import vadclip_run as VR
VR.VM.clip.load = (lambda o: (lambda n, device: o("ViT-L/14", device, download_root=str(ROOT / "raw_data/clip_weights"))))(VR.VM.clip.load)
vc = VR.VM.CLIPVAD(num_class=2, embed_dim=768, visual_length=16, visual_width=768, visual_head=1, visual_layers=1, attn_window=4, prompt_prefix=10, prompt_postfix=10, device=dev).to(dev).eval()
tf = vc.encode_textprompt(VR.PROMPT_TEXT)
def fv():
    e = vcl(x16[0])[0].unsqueeze(0).float(); vf = vc.encode_video(e, None, None)
    return vc.classifier(vf + vc.mlp2(vf))
out["VadCLIP (L14)"] = timeit(fv)
out["gpu_state_after"] = util()
json.dump(out, open(ROOT / "workspace/logs/latency.json", "w"), indent=2)
for k, v in out.items(): print(f"{k:22s} {v:8.1f} ms" if isinstance(v, float) else f"{k:22s} {v}")
