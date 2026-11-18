#!/usr/bin/env python3
"""C4/C5 (PREREG_revision.md): score the revision GPU runs on the seven unseen sets outside XD/UCF (fresh = SCFD+SCVD,
conf = VioPeru+AIRTLab, cfights = CCTV-Fights, conf2 = Movies+UBI-Fights), exactly as score_scfd_scvd.py, score_vioperu_airtlab.py,
score_cctv_fights.py and confirm_footage_gate.py score the published baselines (frame store, 'seq' decode, TTA). Writes <set>/preds/<run>.csv; skips existing.
Usage: score_finetuned_unseen.py <run_dir> [<run_dir> ...]   (LoRA runs from runs_review/, C4 runs from runs_baselines/)"""
import sys, json
from pathlib import Path
import pandas as pd, torch
from torch.utils.data import DataLoader
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import score_scfd_scvd as SF, score_vioperu_airtlab as SC, score_cctv_fights as SCF, confirm_footage_gate as C2
import train_video_baseline as TB, train_finetuned_on_split as RTS
import video_data.frame_store as VF
from video_data import window_data as D
J = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/frames")
SETS = [("fresh", SF.fresh_df, (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/fresh/frames"), SF.P),
        ("conf", SC.conf_df, J, SC.P), ("cfights", SCF.windows, J, SCF.P), ("conf2", C2.windows, J, C2.P)]
dev = torch.device("cuda")
for rd in map(Path, sys.argv[1:]):
    s = json.load(open(rd / "summary.json")); kind = s["model"]
    m = TB.Net(kind).to(dev)
    if s.get("lora"): RTS.add_lora(m); m.to(dev)
    m.load_state_dict(torch.load(rd / "best.pt", map_location=dev), strict=(kind != "xclip")); m.eval()
    for name, dff, root, P in SETS:
        out = P / f"{rd.name}.csv"
        if out.exists(): continue
        w = dff().assign(binary_label=0); VF.STORE_ROOT = root; store = VF.FrameStore(stores=(name,), use_v1=False)
        dl = DataLoader(D.WindowDataset(w, store, False, decode="seq"), batch_size=8, num_workers=8, pin_memory=True)
        pd.DataFrame({"p_violent": D.predict(m, dl, dev, tta=True)}).to_csv(out, index=False); print("scored", rd.name, name, len(w), flush=True)
    del m; torch.cuda.empty_cache()
