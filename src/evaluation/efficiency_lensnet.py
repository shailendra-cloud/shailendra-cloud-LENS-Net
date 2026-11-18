#!/usr/bin/env python3
"""Inference cost per 1.5-s window (16 frames) and training cost. FLOPs from torch.utils.flop_counter (exact, CPU, fp32);
parameter counts split into total and trained. Latency is measured separately (GPU) only if a GPU is idle enough."""
import json, sys, time
from pathlib import Path
import torch
from torch.utils.flop_counter import FlopCounterMode
ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]); sys.path.insert(0, str(ROOT)); sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import clip_tail as C, lensnet_setup as L, lensnet as V, train_video_baseline as TB
from video_data import cnn_bigru_model as mm
from transformers import VideoMAEForVideoClassification
torch.set_num_threads(16)
def gflops(fn):
    with FlopCounterMode(display=False) as fc: fn()
    return fc.get_total_flops() / 1e9
def npar(m): return sum(p.numel() for p in m.parameters())
out = {}
with torch.no_grad():
    x16 = torch.randn(16, 3, 224, 224); clip = C.load_clip(torch.device("cpu"), "fp32"); clip.output_tokens = True
    g_clip = gflops(lambda: clip(x16))
    vm = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").eval()
    xv = torch.randn(1, 16, 3, 224, 224); g_vm = gflops(lambda: vm.videomae(pixel_values=xv))
    head = V.V2("LENS-M", L.text_anchors()).eval()
    g_head = gflops(lambda: head(torch.randn(1, 768), torch.randn(1, 16, 17, 10), torch.randn(1, 768)))
    out["LENS-Net"] = dict(gflops=g_clip + g_vm + g_head, params_total=npar(clip) + npar(vm.videomae) + npar(head),
                           params_trained=sum(p.numel() for p in head.parameters() if p.requires_grad), detail=dict(clip16=g_clip, videomae=g_vm, head=g_head))
    for kind in ["videomae", "swin3d_t", "r2plus1d_18"]:
        net = TB.Net(kind).eval(); out[f"{kind} FT"] = dict(gflops=gflops(lambda: net(xv)), params_total=npar(net), params_trained=npar(net))
    ck = torch.load(ROOT / "source_benchmark/runs/binary_hybrid_seed42/best.pt", map_location="cpu", weights_only=False); a = ck["arch"]
    mn = mm.MASTNet(backbone_kind=a["backbone_kind"], pretrained=False, num_classes=2, use_motion=a["use_motion"], motion_pretrained=a["motion_pretrained"],
                    use_interaction=a["use_interaction"], whiten=a["whiten"]).eval()
    out["CNN-BiGRU"] = dict(gflops=gflops(lambda: mn(xv)), params_total=npar(mn), params_trained=npar(mn))
for k, v in out.items(): print(f"{k:14s} {v['gflops']:8.1f} GFLOPs/window  total params {v['params_total']/1e6:7.1f}M  trained {v['params_trained']/1e6:9.4f}M")
json.dump(out, open(ROOT / "workspace/logs/efficiency.json", "w"), indent=2, default=float)
