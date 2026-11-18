"""B7: score the extra-seed heads (seeds 45-51) on the fresh, conf, cfights and conf2 windows from the stored compact
features, into review/preds_seeds/<dir>/<run>.csv (the published preds/ folders are not touched)."""
import sys, numpy as np, pandas as pd, torch
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import lensnet_setup as L, lensnet as V
R = L.ROOT / "workspace"; OUT = R / "review/preds_seeds"
CF = {d: V.Compact(d) for d in ("fresh", "conf", "cfights", "conf2")}
FILES = {"fresh": "fresh_windows.csv", "conf": "conf_windows.csv", "cfights": "cfights_windows.csv", "conf2": "conf2_windows.csv"}
for cfg, tag in (("LENS-M", "LENS-M_vrex"), ("LP", "LP"), ("V2b", "V2b"), ("MP", "MP")):
    for s in range(45, 52):
        n = f"v2_audited_{tag}_seed{s}"
        m = V.V2(cfg, L.text_anchors()); m.load_state_dict(torch.load(L.OUT / n / "best.pt", weights_only=True)); m.eval()
        for d, cf in CF.items():
            w = pd.read_csv(R / d / FILES[d]); nw = int((w.decode_failed == 0).sum())
            p = V.predict(m, cf, np.arange(nw)); (OUT / d).mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"p_violent": p}).to_csv(OUT / d / f"{n}.csv", index=False)
    print("scored", tag, flush=True)
# consistency: re-scoring a published seed this way reproduces its stored predictions
m = V.V2("LENS-M", L.text_anchors()); m.load_state_dict(torch.load(L.OUT / "v2_audited_LENS-M_vrex_seed42/best.pt", weights_only=True)); m.eval()
for d, cf in CF.items():
    w = pd.read_csv(R / d / FILES[d]); nw = int((w.decode_failed == 0).sum())
    p = V.predict(m, cf, np.arange(nw)); ref = pd.read_csv(R / d / "preds/v2_audited_LENS-M_vrex_seed42.csv").p_violent.values
    print(d, "max |diff| vs stored seed-42 predictions:", float(np.abs(p - ref).max()))
