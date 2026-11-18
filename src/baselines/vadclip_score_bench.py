#!/usr/bin/env python3
"""Score every final VadCLIP run (selected lr) on all bench windows (row order of features/<bb>/bench/index.csv), so
that P3 transfer targets can be evaluated exactly as for LENS-Net (analyze_baselines_rwf target definition)."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import vadclip_run as R
dev = torch.device("cuda")
for d in sorted((R.ROOT / "workspace/runs_vadclip").glob("VadCLIP-*_seed4?")):
    s = json.load(open(d / "summary.json"))
    if "test_auc" not in s or (d / "predictions_bench.npy").exists(): continue      # final (allow-test) runs only
    bb = s["bb"]; orig = R.VM.clip.load
    R.VM.clip.load = lambda n, device, bb=bb, o=orig: o(R.CLIPNAME[bb], device, download_root=str(R.ROOT / "raw_data/clip_weights"))
    m = R.VM.CLIPVAD(num_class=2, embed_dim=R.EMB[bb], visual_length=16, visual_width=R.EMB[bb], visual_head=1, visual_layers=1,
                     attn_window=4, prompt_prefix=10, prompt_postfix=10, device=dev).to(dev)
    R.VM.clip.load = orig
    m.load_state_dict(torch.load(d / "best.pt", weights_only=True), strict=False)
    X = np.load(R.FEATS[bb] / "bench/cls_frames.npy", mmap_mode="r")
    np.save(d / "predictions_bench.npy", R.score(m, np.asarray(X), dev).astype(np.float32)); print("scored", d.name, flush=True)
