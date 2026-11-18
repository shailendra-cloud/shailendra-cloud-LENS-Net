#!/usr/bin/env python3
"""Error analysis on existing clean predictions (PREREG_journal Sec. 5a-d; descriptive). LENS-Net (LENS-M, P1) vs VideoMAE-B FT (A1),
seeds 42-44. Each seed uses its own validation threshold; rates and AUCs are averaged over seeds.
XD codes: B1 fighting, B2 shooting, B4 riot, B5 abuse, B6 car accident, G explosion (the first code is the primary category).
Writes logs/error_analysis.json."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
R = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace"); SEEDS = (42, 43, 44)
XD = {"B1": "Fighting", "B2": "Shooting", "B4": "Riot", "B5": "Abuse", "B6": "Car accident", "G": "Explosion"}
MODELS = {"LENS-Net": lambda s: R / f"runs/v2_audited_LENS-M_vrex_seed{s}", "VideoMAE-B FT": lambda s: R / f"runs_baselines/A1_videomae_seed{s}"}
CONF = {"LENS-Net": "v2_audited_LENS-M_vrex_seed{}", "VideoMAE-B FT": "A1_videomae_seed{}"}
out = {"external_per_category": {}, "airtlab": {}, "viop": {}, "examples": {}}
thr = {m: {s: json.load(open(f(s) / "summary.json"))["threshold_selected_on_val"] for s in SEEDS} for m, f in MODELS.items()}

# (a) XD / UCF per category: positives of the category vs all negatives of the benchmark
for m, f in MODELS.items():
    per = {}
    for s in SEEDS:
        d = pd.read_csv(f(s) / "predictions_external.csv"); d = d[d.excluded_precaution == 0] if "excluded_precaution" in d else d
        for b, g in d.groupby("benchmark"):
            neg = g[g.binary_label == 0]
            cat = g.category.str.split("+").str[0].map(lambda c: XD.get(c, c)) if b == "XD-test" else g.category
            for c in sorted(set(cat[g.binary_label == 1])):
                pos = g[(g.binary_label == 1) & (cat == c)]
                if len(pos) < 8: continue
                y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]; p = np.r_[pos.p_violent, neg.p_violent]
                r = per.setdefault(f"{b}|{c}", {"n_pos": len(pos), "auc": [], "recall": []})
                r["auc"].append(100 * roc_auc_score(y, p)); r["recall"].append(100 * float((pos.p_violent >= thr[m][s]).mean()))
    out["external_per_category"][m] = {k: dict(n_pos=v["n_pos"], auc=float(np.mean(v["auc"])), recall=float(np.mean(v["recall"]))) for k, v in per.items()}

# (b) AIRTLab action classes, (c) VioPeru false-positive subset, from the confirmation predictions
w = pd.read_csv(R / "conf/conf_windows.csv"); w = w[w.decode_failed == 0].reset_index(drop=True)
AD = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/airt/violence-detection-dataset")
acts = {}
for fn, lab in (("violent-action-classes.csv", "violent"), ("nonviolent-action-classes.csv", "non-violent")):
    t = pd.read_csv(AD / fn, sep=";"); t.columns = [c.strip() for c in t.columns]
    for f_, a in zip(t.FILE, t["ACTION CLASSES"]): acts[f"{lab}/{Path(f_.strip()).stem}"] = [x.strip() for x in str(a).split(",")]
for m in MODELS:
    ps = {s: pd.read_csv(R / "conf/preds" / f"{CONF[m].format(s)}.csv").p_violent.values for s in SEEDS}
    a = w[w.benchmark == "AIRTLab"]; rows = {}
    for i, r in a.iterrows():
        for act in acts.get(r.cluster, []):
            rows.setdefault((r.binary_label, act), []).append(i)
    out["airtlab"][m] = {f"{'miss' if lab else 'false_alarm'}|{act}": dict(n_windows=len(ix), rate=float(np.mean(
        [100 * np.mean((ps[s][ix] >= thr[m][s]) != bool(lab)) for s in SEEDS]))) for (lab, act), ix in sorted(rows.items())}
    v = w[(w.benchmark == "VioPeru") & (w.binary_label == 0)]
    fp = v.category == "false_positives_validation"
    out["viop"][m] = {k: dict(n_windows=int(mask.sum()), false_alarm_rate=float(np.mean([100 * np.mean(ps[s][v.index[mask]] >= thr[m][s]) for s in SEEDS])))
                      for k, mask in (("false_positives_validation", fp.values), ("other non-violent", ~fp.values))}

# (d) fixed-rule examples: 3 most confident false alarms and misses for LENS-Net (seed-averaged probability)
avg = np.mean([pd.read_csv(R / "conf/preds" / f"v2_audited_LENS-M_vrex_seed{s}.csv").p_violent.values for s in SEEDS], 0)
for b in ("VioPeru", "AIRTLab"):
    g = w[w.benchmark == b].assign(p=avg[w.benchmark == b])
    out["examples"][b] = dict(false_alarms=g[g.binary_label == 0].nlargest(3, "p")[["video", "start_frame", "p"]].to_dict("records"),
                              misses=g[g.binary_label == 1].nsmallest(3, "p")[["video", "start_frame", "p"]].to_dict("records"))
json.dump(out, open(R / "logs/error_analysis.json", "w"), indent=2, default=float)
for m, d in out["external_per_category"].items():
    print(m); [print(f"   {k:28s} n={v['n_pos']:4d}  AUC {v['auc']:6.2f}  recall {v['recall']:6.2f}") for k, v in d.items()]
for m in MODELS:
    print(m, "AIRTLab"); [print(f"   {k:28s} n={v['n_windows']:4d}  rate {v['rate']:6.2f}") for k, v in out["airtlab"][m].items()]
    print(m, "VioPeru", {k: round(v["false_alarm_rate"], 2) for k, v in out["viop"][m].items()})
print(json.dumps(out["examples"], indent=1, default=float)[:1500])
