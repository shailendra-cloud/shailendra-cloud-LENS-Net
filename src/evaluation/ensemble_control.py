"""Control for anchor fusion (no training): is the gain of averaging the CLIP probe with LENS-Net more than seed
ensembling? Window-level ROC-AUC on the audited test split and the nine unseen sets for the single model, the
probe + LENS-Net average (A), two and three seeds of LENS-Net (E1, E1b), two probes (E2), LENS-Net + its variant
without motion (E3) and three probes + three LENS-Nets (A3). Prints the table kept as results/ensemble_control.txt.
All nine sets had been seen before, so the comparison is exploratory."""
import sys, json
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import lensnet_setup as L, lensnet as V
from sklearn.metrics import roc_auc_score
R = L.ROOT / "workspace"


def comps(m, cf, rows):
    r = torch.as_tensor(np.asarray(rows)); cls, S, M = cf.get(r)
    with torch.no_grad():
        la = L.SCALE * F.normalize(cls, dim=-1) @ F.normalize(m.w, dim=-1).t()
        corr = m.g * m.head(V.pps_features(S, "V2b")) + m.gm * m.mhead(M)
    return la, corr


def p_of(logit): return torch.softmax(logit, 1)[:, 1].numpy()


def load(cfg, n):
    m = V.V2(cfg, L.text_anchors()); m.load_state_dict(torch.load(L.OUT / n / "best.pt", weights_only=True)); m.eval(); return m


# ---- evaluation sets: (name, compact set, row indices, labels, keep mask)
sets = []
df, _ = L.load_split("audited"); va = df[df.split == "val"].reset_index(drop=True); te = df[df.split == "test"].reset_index(drop=True)
sets.append(("Test", "bench", te.feat_row.values, te.binary_label.values))
ext = pd.read_csv(L.ROOT / "data_tables/manifests/external_windows.csv")
pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True)
for b, nm in (("XD-test", "XD"), ("UCF-test", "UCF")):
    idx = np.where(pool.benchmark == b)[0]; sets.append((nm, "external", idx, pool.binary_label.values[idx]))
w = pd.read_csv(R / "fresh/fresh_windows.csv"); w = w[w.decode_failed == 0].reset_index(drop=True)
import glob
x = pd.read_csv(glob.glob(str(R / "fresh/*exclusion*.csv"))[0]); k = x.excluded_near_duplicate.values == 0
for b in ("SCFD", "SCVD"):
    idx = np.where(k & (w.benchmark == b).values)[0]; sets.append((b, "fresh", idx, w.binary_label.values[idx]))
w = pd.read_csv(R / "conf/conf_windows.csv"); w = w[w.decode_failed == 0].reset_index(drop=True); k = pd.read_csv(R / "conf/near_duplicate_exclusion.csv").excluded_near_duplicate.values == 0
for b in ("VioPeru", "AIRTLab"):
    idx = np.where(k & (w.benchmark == b).values)[0]; sets.append((b, "conf", idx, w.binary_label.values[idx]))
w = pd.read_csv(R / "cfights/cfights_windows.csv"); w = w[w.decode_failed == 0].reset_index(drop=True); k = pd.read_csv(R / "cfights/near_duplicate_exclusion.csv").excluded_near_duplicate.values == 0
idx = np.where(k & (w.category == "CCTV").values)[0]; sets.append(("CCTV-F", "cfights", idx, w.binary_label.values[idx]))
w = pd.read_csv(R / "conf2/conf2_windows.csv"); w = w[w.decode_failed == 0].reset_index(drop=True); k = pd.read_csv(R / "conf2/near_duplicate_exclusion.csv").excluded_near_duplicate.values == 0
for b in ("Movies", "UBI-Fights"):
    idx = np.where(k & (w.benchmark == b).values)[0]; sets.append((b, "conf2", idx, w.binary_label.values[idx]))
CF = {n: V.Compact(n) for n in {s[1] for s in sets} | {"bench"}}
P = {}
for s in (42, 43, 44):
    for tag, cfg, n in (("M", "LENS-M", f"v2_audited_LENS-M_vrex_seed{s}"), ("P", "LP", f"v2_audited_LP_seed{s}"), ("N", "V2b", f"v2_audited_V2b_seed{s}")):
        m = load(cfg, n); P[(tag, s)] = [V.predict(m, CF[cs], rows) for _, cs, rows, _ in sets]
nxt = {42: 43, 43: 44, 44: 42}
combos = {"single LENS-Net": lambda s: [("M", s)], "A: probe + LENS-Net": lambda s: [("P", s), ("M", s)],
          "E1: LENS-Net + LENS-Net (2 seeds)": lambda s: [("M", s), ("M", nxt[s])], "E1b: LENS-Net, 3 seeds": lambda s: [("M", 42), ("M", 43), ("M", 44)],
          "E2: probe + probe (2 seeds)": lambda s: [("P", s), ("P", nxt[s])], "E3: LENS-Net + w/o motion": lambda s: [("M", s), ("N", s)],
          "A3: 3 probes + 3 LENS-Nets": lambda s: [(t, q) for t in ("P", "M") for q in (42, 43, 44)]}
names = [x[0] for x in sets]
print(f"{'ensemble':36s} " + " ".join(f"{n[:7]:>7s}" for n in names) + "  Mean9  Min9")
for k, f in combos.items():
    v = []
    for s in (42, 43, 44):
        ms = f(s); v.append([100 * roc_auc_score(y, np.mean([P[m][i] for m in ms], 0)) for i, (_, _, _, y) in enumerate(sets)])
    v = np.mean(v, 0); m9 = v[1:]
    print(f"{k:36s} " + " ".join(f"{a:7.2f}" for a in v) + f"  {m9.mean():5.2f}  {m9.min():5.2f}")
