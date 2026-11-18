#!/usr/bin/env python3
"""Part F (PREREG_revision.md): score the two video members of AES (VideoMAE-B tuned, VideoMAE-B LoRA; seeds 42-44) on
the untouched confirmation samples, chunk by chunk: rebuild the chunk's frame store with the original build function,
check that its window list is unchanged, score ('seq' decode, TTA, as score_finetuned_unseen.py), remove the store.
Writes conf3/preds/<run>.csv (sample A + AIE, chunks 0-5) and conf3/predsB/<run>.csv (sample B, chunks 10-12)."""
import shutil, subprocess, sys
from pathlib import Path
import pandas as pd, torch
from torch.utils.data import DataLoader
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import train_video_baseline as TB, train_finetuned_on_split as RTS
import video_data.frame_store as VF
from video_data import window_data as D
import json
L = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace"); C3 = L / "conf3"; SCR = Path(__file__).resolve().parent
def script(name): return str(next(SCR.parent.glob(f"*/{name}")))      # scripts live in src/<area>/
FR = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/frames"); TMP = C3 / "aes_chunks"; TMP.mkdir(exist_ok=True)
RUNS = [L / f"runs_baselines/A1_videomae_lr2e-05_seed{s}" for s in (42, 43, 44)] + [L / f"runs_review/A1_videomae_lora_lr0.0001_seed{s}" for s in (42, 43, 44)]
CHUNKS = {"A": [0, 1, 2, 3, 4, 5], "B": [10, 11, 12]}
dev = torch.device("cuda")
models = {}
for rd in RUNS:
    s = json.load(open(rd / "summary.json")); m = TB.Net(s["model"]).to(dev)
    if s.get("lora"): RTS.add_lora(m); m.to(dev)
    m.load_state_dict(torch.load(rd / "best.pt", map_location=dev)); m.eval(); models[rd.name] = m
for part, ks in CHUNKS.items():
    for k in ks:
        todo = [n for n in models if not (TMP / f"{n}_c{k}.csv").exists()]
        if not todo: continue
        ref = pd.read_csv(C3 / f"chunks/c3_{k}.csv"); shutil.copy(C3 / f"chunks/c3_{k}.csv", TMP / f"chunk_c3_{k}_original.csv")
        build = ["confirm_anchor_fusion.py", "build", str(k)] if part == "A" else ["confirm_anchor_fusion_sample_b.py", "build", str(k)]
        subprocess.run([sys.executable, script(build[0]), *build[1:]], check=True, cwd=str(SCR))
        new = pd.read_csv(C3 / f"chunks/c3_{k}.csv")
        cols = ["path", "start_frame", "end_frame", "decode_failed"]
        assert len(new) == len(ref) and (new[cols].values == ref[cols].values).all(), f"chunk {k}: window list changed"
        w = new[new.decode_failed == 0].reset_index(drop=True).assign(binary_label=0)
        VF.STORE_ROOT = FR; store = VF.FrameStore(stores=(f"c3_{k}",), use_v1=False)
        for n in todo:
            dl = DataLoader(D.WindowDataset(w, store, False, decode="seq"), batch_size=8, num_workers=8, pin_memory=True)
            pd.DataFrame({"p_violent": D.predict(models[n], dl, dev, tta=True)}).to_csv(TMP / f"{n}_c{k}.csv", index=False)
            print(f"scored {n} chunk {k}: {len(w)} windows", flush=True)
        subprocess.run([sys.executable, script("confirm_anchor_fusion.py"), "clean", str(k)], check=True, cwd=str(SCR))
        print(f"chunk {k} done, store removed", flush=True)
for part, ks in CHUNKS.items():
    out = C3 / ("preds" if part == "A" else "predsB")
    for n in models:
        pd.concat([pd.read_csv(TMP / f"{n}_c{k}.csv") for k in ks], ignore_index=True).to_csv(out / f"{n}.csv", index=False)
print("ALL_DONE", flush=True)
