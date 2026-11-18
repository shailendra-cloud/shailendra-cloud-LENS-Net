#!/usr/bin/env python3
"""Journal E2 cost (PREREG_journal Sec. 2): inference GFLOPs per 1.5-s window for LENS-Net (L14, 16 frames), the Lite
variants L1 (L14, 8 CLIP frames), L2 (B16, 16), L3 (B16, 8) and LENS-AG. Same method as efficiency_lensnet.py (torch FlopCounterMode,
CPU fp32). VideoMAE-B always takes 16 frames (PREREG_journal deviation 1). Writes logs/efficiency_journal.json."""
import json, sys
from pathlib import Path
import torch, open_clip
from torch.utils.flop_counter import FlopCounterMode
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import lensnet_setup as L, lensnet as V
from transformers import VideoMAEForVideoClassification
torch.set_num_threads(8)
def gflops(fn):
    with FlopCounterMode(display=False) as fc: fn()
    return fc.get_total_flops() / 1e9
def npar(m): return sum(p.numel() for p in m.parameters())
def vis(name):
    v = open_clip.create_model_and_transforms(name, pretrained="openai", device="cpu")[0].visual.eval()
    assert type(v.transformer.resblocks[0].mlp.gelu).__name__ == "QuickGELU"; v.output_tokens = True; return v
out = {}
with torch.no_grad():
    vm = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").eval()
    g_vm = gflops(lambda: vm.videomae(pixel_values=torch.randn(1, 16, 3, 224, 224))); p_vm = npar(vm.videomae)
    head = V.V2("LENS-M", L.text_anchors()).eval(); ag = V.V2("LENS-AG", L.text_anchors()).eval()
    g_h = gflops(lambda: head(torch.randn(1, 768), torch.randn(1, 16, 17, 10), torch.randn(1, 768)))
    g_ag = gflops(lambda: ag(torch.randn(1, 768), torch.randn(1, 16, 17, 10), torch.randn(1, 768)))
    for bb, name in (("L14", "ViT-L-14-quickgelu"), ("B16", "ViT-B-16-quickgelu")):
        v = vis(name)
        for nf in (16, 8):
            g = gflops(lambda: v(torch.randn(nf, 3, 224, 224)))
            key = {("L14", 16): "LENS-Net", ("L14", 8): "L1 (L14, 8f)", ("B16", 16): "L2 (B16, 16f)", ("B16", 8): "L3 (B16, 8f)"}[(bb, nf)]
            out[key] = dict(gflops=g + g_vm + g_h, clip=g, videomae=g_vm, params_total_frozen=npar(v) + p_vm)
        del v
    out["LENS-AG"] = dict(gflops=out["LENS-Net"]["clip"] + g_vm + g_ag, head=g_ag, params_trained=sum(p.numel() for p in ag.parameters() if p.requires_grad))
for k, v in out.items(): print(f"{k:16s} {v['gflops']:8.1f} GFLOPs/window")
json.dump(out, open(L.ROOT / "workspace/logs/efficiency_journal.json", "w"), indent=2, default=float)
