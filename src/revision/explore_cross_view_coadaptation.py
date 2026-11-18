"""EXPLORATORY feasibility check (development data only, not pre-registered): cross-view co-adaptation (CVCA).
On each unseen set, without labels: the frozen anchor (semantic view) and an adapted video model (motion view) give
pseudo-labels where both rank a window confidently and agree; light linear heads are then fitted on the target set's
own frozen features (semantic: CLIP CLS + prompt statistics; motion: VideoMAE), cross-fitted over two halves of the
set's videos, and fused with the two teachers. Labels are used only to score. Writes review/explore_coadapt.json."""
import json, sys, zlib
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import saved_predictions as RD, lensnet as V
from saved_predictions import Data, UNSEEN, L
RD.MODELS.update({"VideoMAE-B FT (tuned)": "A1_videomae_lr2e-05", "VideoMAE-B LoRA": "A1_videomae_lora_lr0.0001"})
_orig = RD._run_dir; RD._run_dir = lambda st, sd: (L / f"runs_review/{st}_seed{sd}") if "_lora" in st else _orig(st, sd)
D = Data(); ANCHOR = "CLIP probe"
FEAT = {"XD": "external", "UCF": "external", "SCFD": "fresh", "SCVD": "fresh", "VioPeru": "conf", "AIRTLab": "conf",
        "CCTV-F": "cfights", "Movies": "conf2", "UBI-Fights": "conf2"}
CF = {n: V.Compact(n) for n in set(FEAT.values())}


def views(s):
    S = D.sets[s]; r = S["rows"][S["keep"]]; c = CF[FEAT[s]]
    cls, Sm, M = c.get(torch.as_tensor(r))
    sem = torch.cat([F.normalize(cls, dim=-1), V.pps_features(Sm, "V2b")], 1).numpy()
    mot = F.normalize(M, dim=-1).numpy()
    return sem, mot


def pct(p): return (rankdata(p) - 0.5) / len(p)


def head(X, idx, lab, Xall):
    sc = StandardScaler().fit(X[idx]); clf = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(X[idx]), lab); return clf.predict_proba(sc.transform(Xall))[:, 1]


def cvca(pa, pv, sem, mot, cl, q, rounds=1):
    """returns adapted semantic and motion head scores (cross-fitted over two halves of the videos)"""
    fold = np.array([zlib.crc32(str(c).encode()) % 2 for c in cl]); teach = (pct(pa) + pct(pv)) / 2
    ra, rv = pct(pa), pct(pv)
    for _ in range(rounds):
        hs, hm = np.zeros(len(pa)), np.zeros(len(pa))
        for f in (0, 1):
            tr, te = fold != f, fold == f
            pos = tr & (ra > 1 - q) & (rv > 1 - q); neg = tr & (ra < q) & (rv < q)
            idx = np.where(pos | neg)[0]; lab = pos[idx].astype(int)
            if lab.sum() < 5 or (1 - lab).sum() < 5: hs[te], hm[te] = teach[te], teach[te]; continue
            hs[te] = head(sem, idx, lab, sem)[te]; hm[te] = head(mot, idx, lab, mot)[te]
        ra, rv = pct((pct(pa) + pct(hs)) / 2), pct((pct(pv) + pct(hm)) / 2)    # next round: views refined by their heads
    return hs, hm


auc = lambda y, p: 100 * roc_auc_score(y, p)
VIDEO = {"LoRA": "VideoMAE-B LoRA", "tuned": "VideoMAE-B FT (tuned)"}
res = {}
for vname, vm in VIDEO.items():
    for q in (0.2, 0.3):
        for rounds in (1, 2):
            tag = f"{vname} q{q} r{rounds}"; acc = {k: {} for k in ("teachers mean", "CVCA heads only", "CVCA + teachers", "teachers + motion head")}
            for s in UNSEEN:
                S = D.sets[s]; y = S["y"][S["keep"]]; cl = np.asarray(S["cl"])[S["keep"]]; sem, mot = views(s)
                PA = [p[S["keep"]] for p in D.probs(ANCHOR, s)]; PV = [p[S["keep"]] for p in D.probs(vm, s)]
                a = {k: [] for k in acc}
                for i in range(3):
                    pa, pv = PA[i], PV[i]; hs, hm = cvca(pa, pv, sem, mot, cl, q, rounds)
                    a["teachers mean"].append(auc(y, (pa + pv) / 2))
                    a["CVCA heads only"].append(auc(y, pct(hs) + pct(hm)))
                    a["CVCA + teachers"].append(auc(y, pct(pa) + pct(pv) + pct(hs) + pct(hm)))
                    a["teachers + motion head"].append(auc(y, pct(pa) + pct(pv) + pct(hm)))
                for k in acc: acc[k][s] = float(np.mean(a[k]))
            res[tag] = acc
            base = acc["teachers mean"]
            for k, per in acc.items():
                m9 = np.mean(list(per.values())); mn = min(per.values())
                b = sum(per[s] - base[s] > 0.5 for s in UNSEEN); w = sum(per[s] - base[s] < -0.5 for s in UNSEEN)
                print(f"{tag:18s} {k:24s} mean9 {m9:6.2f}  min9 {mn:6.2f}  +{b}/-{w}  " + " ".join(f"{s[:4]}{per[s]-base[s]:+.1f}" for s in UNSEEN), flush=True)
json.dump(res, open(L / "review/explore_coadapt.json", "w"), indent=1)
