"""Exploratory check (no training): do anchor-safe combinations of existing checkpoints help across the nine unseen sets?
A) average of probabilities: anchored CLIP probe (LP) and LENS-Net (LENS-M), same seed
B) LENS-Net with its corrections bounded: logit = l_A + clamp(g_P d_P + g_M d_M, -c, c); c chosen on the audited
   validation split (window AUC) from {0.25, 0.5, 1, 2, 4, inf}
All nine sets have been seen before, so this only decides whether a pre-registered study is worth running."""
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
CS = [0.25, 0.5, 1.0, 2.0, 4.0, float("inf")]
res = {"LENS-Net": [], "CLIP probe": [], "A: average(probe, LENS-Net)": [], "B: bounded corrections (c on val)": [], "chosen c": []}
for s in (42, 43, 44):
    mM = load("LENS-M", f"v2_audited_LENS-M_vrex_seed{s}"); mP = load("LP", f"v2_audited_LP_seed{s}")
    la, co = comps(mM, CF["bench"], va.feat_row.values)
    vauc = {c: roc_auc_score(va.binary_label, p_of(la + co.clamp(-c, c))) for c in CS}
    c_best = max(CS, key=lambda c: (round(vauc[c], 6), -CS.index(c) if c != float("inf") else -99)); res["chosen c"].append(c_best)
    row = {k: [] for k in res if k != "chosen c"}
    for name, cs, rows, y in sets:
        la, co = comps(mM, CF[cs], rows); pM = p_of(la + co)
        pP = V.predict(mP, CF[cs], rows)
        row["LENS-Net"].append(roc_auc_score(y, pM)); row["CLIP probe"].append(roc_auc_score(y, pP))
        row["A: average(probe, LENS-Net)"].append(roc_auc_score(y, (pM + pP) / 2))
        row["B: bounded corrections (c on val)"].append(roc_auc_score(y, p_of(la + co.clamp(-c_best, c_best))))
    for k2 in row: res[k2].append(row[k2])
names = [s[0] for s in sets]
print("chosen c per seed:", res["chosen c"])
print(f"{'model':36s} " + " ".join(f"{n[:7]:>7s}" for n in names) + "  Mean9  Min9")
out = {}
for k2 in ("LENS-Net", "CLIP probe", "A: average(probe, LENS-Net)", "B: bounded corrections (c on val)"):
    v = 100 * np.mean(res[k2], 0); m9 = v[1:]
    out[k2] = dict(zip(names, v.round(2).tolist()), mean9=round(float(m9.mean()), 2), min9=round(float(m9.min()), 2))
    print(f"{k2:36s} " + " ".join(f"{a:7.2f}" for a in v) + f"  {m9.mean():5.2f}  {m9.min():5.2f}")
json.dump(out, open(R / "logs/anchor_safe_exploratory.json", "w"), indent=2)
