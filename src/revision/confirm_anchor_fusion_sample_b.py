#!/usr/bin/env python3
"""C6 (PREREG_revision.md): second untouched XD-Violence training sample (conf3/xd_train_selection_B.csv), processed with
the functions of confirm_anchor_fusion.py (same windows rule, frame store, features, near-duplicate rule, scoring) under chunk names
c3_10, c3_11, ... so that sample A's files are untouched.
  plan | build K | dedup | score | analyze"""
import sys, json, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
sys.argv, _argv = [sys.argv[0]], sys.argv
import confirm_anchor_fusion as C
ROOT, C3, JD = C.ROOT, C.C3, C.JD
WB, VB, PB, K0 = C3 / "windowsB.csv", C3 / "videosB.csv", C3 / "predsB", 10


def xd_videos_B():
    sel = pd.read_csv(C3 / "xd_train_selection_B.csv")
    return [dict(benchmark="XD-train-B", video=r.name, cluster="XD/" + r.name.split("__")[0], path=str(JD / "xd" / r.path), category=r.labels,
                 binary_label=int(r.binary_label)) for r in sel.itertuples()]


def plan():
    vids = xd_videos_B()
    with ProcessPoolExecutor(16) as ex: rows = [r for pl in ex.map(C.plan_one, vids) for r in pl]
    df = pd.DataFrame(rows); df["chunk"] = K0 + np.arange(len(df)) // C.CHUNK
    df.drop(columns=["binary_label"]).to_csv(WB, index=False); pd.DataFrame(vids).to_csv(VB, index=False)
    print(f"videos {len(vids)}, windows {len(df)}, chunks {sorted(df.chunk.unique())}, videos without window {len(vids) - df.video.nunique()}", flush=True)


def build(k):
    from video_data.build_external_windows import decode_video
    df = pd.read_csv(WB); df = df[df.chunk == k].reset_index(drop=True)
    jobs = [(p, [(r.start_frame, r.end_frame) for r in g.itertuples()]) for p, g in df.groupby("path", sort=False)]
    w = C.VF.StoreWriter(f"c3_{k}"); failed = []
    with ProcessPoolExecutor(16) as ex:
        for (p, wins), frames in zip(jobs, ex.map(decode_video, jobs, chunksize=1)):
            for (s, e), fr in zip(wins, frames):
                if fr is None: failed.append((p, s)); continue
                w.add(C.VF.content_key(p, s, e, decode="seq"), fr)
    w.close(dict(note=f"revision C6 chunk {k}")); fk = set(failed)
    df["decode_failed"] = [int((p, s) in fk) for p, s in zip(df.path, df.start_frame)]
    df.to_csv(C3 / f"chunks/c3_{k}.csv", index=False); print(f"chunk {k}: {len(df)} windows, decode failures {len(failed)}", flush=True)


def chunks():
    ks = sorted(pd.read_csv(WB, usecols=["chunk"]).chunk.unique()); return ks, [pd.read_csv(C3 / f"chunks/c3_{k}.csv") for k in ks]


def windows_ok():
    ks, cs = chunks(); return ks, pd.concat([c[c.decode_failed == 0] for c in cs], ignore_index=True)


def dedup():
    import torch, torch.nn.functional as F, lensnet_setup as L, lensnet as V
    man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "split"])
    sp = pd.read_csv(ROOT / "workspace/splits/std_rwf.csv", usecols=["segment_id", "split"])
    train = set(man[man.split == "train"].segment_id) | set(sp[sp.split == "train"].segment_id)
    b = V.Compact("bench"); tb = F.normalize(b.cls[np.where(man.segment_id.isin(train).values)[0]], dim=-1)
    ks, w = windows_ok(); mx = []
    for k in ks:
        tc = F.normalize(V.Compact(f"c3_{k}").cls, dim=-1)
        mx.append(torch.cat([(tc @ tb[i:i + 4096].t()).max(1).values.unsqueeze(1) for i in range(0, len(tb), 4096)], 1).max(1).values.numpy())
    mx = np.concatenate(mx); assert len(mx) == len(w)
    w.assign(max_cos_to_train=mx, excluded_near_duplicate=(mx >= 0.95).astype(int)).to_csv(C3 / "near_duplicate_exclusionB.csv", index=False)
    print("excluded windows:", int((mx >= 0.95).sum()), "of", len(mx), flush=True)


def score():
    import torch, lensnet_setup as L, lensnet as V
    assert L.BACKBONE == "L14" and L.FRAMES == 16 and L.PSET == "orig"
    ks, _ = chunks(); PB.mkdir(exist_ok=True); feats = [V.Compact(f"c3_{k}") for k in ks]
    for s in (42, 43, 44):
        for cfg, tag in (("LENS-M", "LENS-M_vrex"), ("LP", "LP"), ("V2b", "V2b"), ("MP", "MP")):
            ck = f"v2_audited_{tag}_seed{s}"; m = V.V2(cfg, L.text_anchors()); m.load_state_dict(torch.load(L.OUT / ck / "best.pt", weights_only=True)); m.eval()
            pd.DataFrame({"p_violent": np.concatenate([V.predict(m, f, np.arange(len(f.index))) for f in feats])}).to_csv(PB / f"{ck}.csv", index=False)
    zs = V.V2("LP", L.text_anchors()); zs.eval()
    pd.DataFrame({"p_violent": np.concatenate([V.predict(zs, f, np.arange(len(f.index))) for f in feats])}).to_csv(PB / "ZS.csv", index=False)
    print("scored", len(list(PB.glob("*.csv"))), "prediction files", flush=True)


def analyze():
    from seed_aware_bootstrap import Boot, summarize
    _, w = windows_ok(); x = pd.read_csv(C3 / "near_duplicate_exclusionB.csv"); keep = x.excluded_near_duplicate.values == 0
    vid = pd.read_csv(VB).set_index("video")
    def vscore(stem): return pd.DataFrame({"video": w.video.values[keep], "p": pd.read_csv(PB / f"{stem}.csv").p_violent.values[keep]}).groupby("video").p.max()
    ids = list(vscore("v2_audited_LENS-M_vrex_seed42").index); y = vid.loc[ids, "binary_label"].values; cl = vid.loc[ids, "cluster"].values
    BT = Boot({"B": (y, cl)})
    stems = {"LENS-Net": "v2_audited_LENS-M_vrex", "CLIP probe": "v2_audited_LP", "w/o motion": "v2_audited_V2b", "Motion probe": "v2_audited_MP"}
    P = {m: [vscore(f"{st}_seed{s}").loc[ids].values for s in (42, 43, 44)] for m, st in stems.items()}
    P["LENS-Net-A"] = [(a + b) / 2 for a, b in zip(P["LENS-Net"], P["CLIP probe"])]; P["CLIP zero-shot"] = [vscore("ZS").loc[ids].values]
    H = {m: BT.hier(m, BT.aucs("B", ps)) for m, ps in P.items()}
    res = {"n_videos": len(ids), "n_violent": int(y.sum()), "n_clusters": int(len(set(cl))), "n_windows_kept": int(keep.sum()), "n_windows_excluded": int((~keep).sum()),
           "auc": {m: float(H[m][0]) for m in H}}
    fam = {"K1 LENS-Net-A - LENS-Net (better)": summarize(H["LENS-Net-A"] - H["LENS-Net"], BT.S["B"]["ok"], 0.0),
           "K3 LENS-Net-A - CLIP probe (non-inferior)": summarize(H["LENS-Net-A"] - H["CLIP probe"], BT.S["B"]["ok"], -1.0)}
    fam["K1 LENS-Net-A - LENS-Net (better)"]["p"] = fam["K1 LENS-Net-A - LENS-Net (better)"]["p_one_sided"]
    fam["K3 LENS-Net-A - CLIP probe (non-inferior)"]["p"] = fam["K3 LENS-Net-A - CLIP probe (non-inferior)"]["p_one_sided"]
    order = sorted(fam, key=lambda k: fam[k]["p"]); stop = False
    for i, k in enumerate(order):
        thr = 0.05 / (len(order) - i); fam[k]["holm_threshold"] = thr; fam[k]["holds_after_holm"] = (not stop) and fam[k]["p"] <= thr
        if not fam[k]["holds_after_holm"]: stop = True
    res["claims"] = fam
    json.dump(res, open(ROOT / "workspace/review/c6_confirmB.json", "w"), indent=1)
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res["auc"].items()}))
    for k in order: r = fam[k]; print(f"{k:45s} {r['diff']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] p={r['p']:.4f} holds={r['holds_after_holm']}")
    print({k: v for k, v in res.items() if k.startswith("n_")})


if __name__ == "__main__":
    cmd = _argv[1]; {"plan": plan, "dedup": dedup, "score": score, "analyze": analyze}.get(cmd, lambda: build(int(_argv[2])))()
