"""Part F post-hoc diagnosis (descriptive): mean |p - 0.5| of each AES member's window probabilities on the confirmation
samples (seeds 42-44). Writes review/f_aes_diagnosis.json."""
import sys, json, numpy as np, pandas as pd
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
sys.argv = [sys.argv[0]]
import confirm_anchor_fusion as CA, confirm_anchor_fusion_sample_b as CB
ST = {"CLIP probe": "v2_audited_LP", "VideoMAE-B LoRA": "A1_videomae_lora_lr0.0001", "VideoMAE-B FT (tuned)": "A1_videomae_lr2e-05"}
out = {}
for nm, P, wf, sel in (("XD sample A", CA.P, CA.windows_ok, "XD-train"), ("AIE", CA.P, CA.windows_ok, "AIE"), ("XD sample B", CB.PB, CB.windows_ok, None)):
    _, w = wf(); m = (w.benchmark == sel).values if sel else np.ones(len(w), bool)
    out[nm] = {k: float(np.mean([np.mean(np.abs(pd.read_csv(P / f"{st}_seed{s}.csv").p_violent.values[m] - 0.5)) for s in (42, 43, 44)])) for k, st in ST.items()}
json.dump(out, open(CA.ROOT / "workspace/review/f_aes_diagnosis.json", "w"), indent=1); print(out)
