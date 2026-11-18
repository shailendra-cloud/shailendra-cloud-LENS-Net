#!/usr/bin/env python3
"""LENS-Net (PREREG deviation 2): anchored CLIP probe + Prompt-Projected Semantic Signals (PPS) through a
zero-initialized gate, optionally with a cross-corpus V-REx penalty. Trains on the compact inputs (CPU).

Configs: LP (Block 0 only), V2a (0+1+2), V2b (0+1+2+3), -G (no gate), -T (temporal mean only), -Sp (CLS only; no grid).
-G/-T/-Sp inherit V-REx from the selected candidate via --vrex.
Test rows are never scored unless --allow-test; audited runs with --allow-test also score XD/UCF (F3).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import lensnet_setup as L
from video_data import window_data as D

torch.set_num_threads(8)


class Compact:
    def __init__(self, name):
        z = np.load(L.FEAT / name / L.compact_file())
        self.cls = torch.from_numpy(z["cls"]); self.S = torch.from_numpy(z["S"].astype(np.float32))
        self.index = pd.read_csv(L.FEAT / name / "index.csv")
        f = L.FEAT / name / "vmae.npy"
        self.M = torch.from_numpy(np.load(f)) if f.exists() else None

    def get(self, r):
        return self.cls[r], self.S[r], (self.M[r] if self.M is not None else None)


def pps_features(S, cfg):
    """S: (B,16,17,10) prompt similarities -> spatial summaries x temporal summaries x prompts."""
    cls_s = S[:, :, 0]
    if cfg == "-Sp":
        sp = cls_s.unsqueeze(2)
    else:
        g = S[:, :, 1:]
        sp = torch.stack([cls_s, g.max(2).values, g.topk(2, dim=2).values.mean(2)], 2)     # (B,16,3,10)
    feats = [sp.mean(1)]
    if cfg != "-T":
        feats += [sp.max(1).values, sp.std(1, unbiased=False), (sp[:, 1:] - sp[:, :-1]).abs().mean(1)]
    return torch.stack(feats, 1).flatten(1)


class V2(nn.Module):
    def __init__(self, cfg, anchors):
        super().__init__()
        self.cfg = cfg
        self.w0 = nn.Parameter(torch.tensor(anchors), requires_grad=False)
        self.w = nn.Parameter(torch.tensor(anchors).clone())
        if cfg in ("LENS-M", "LENS-AG", "LENS-FG", "MP"):
            self.mhead = nn.Sequential(nn.BatchNorm1d(768), nn.Linear(768, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 2))
            self.gm = nn.Parameter(torch.zeros(())) if cfg != "MP" else None
        if cfg in ("LP", "MP"):
            return
        nf = (1 if cfg == "-Sp" else 3) * (1 if cfg == "-T" else 4) * L.NPROMPT
        cfg = "V2b" if cfg in ("LENS-M", "LENS-AG", "LENS-FG") else cfg
        self.head = nn.Sequential(nn.BatchNorm1d(nf), nn.Linear(nf, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 2))
        self.g = nn.Parameter(torch.zeros(())) if cfg != "-G" else None
        if self.cfg == "LENS-FG":              # PREREG_journal Sec. 9: footage-aware gate, g_M * p_raw(x); no trained parameter added
            fe = torch.tensor(np.load(L.FEAT / "text_footage.npy"))      # 4 raw-footage prompts first, then 6 edited-footage prompts
            self.register_buffer("fe", F.normalize(fe, dim=-1), persistent=False)   # not in the state dict: LENS-M checkpoints load unchanged
        if self.cfg == "LENS-AG":              # journal E1 (PREREG_journal Sec. 1): g_M(x) = g_M + w^T h([s_PPS; z_M]), w = 0 at init.
            self.gnet = nn.Sequential(nn.Linear(120 + 32, 16), nn.GELU())   # created last, so all shared parameters get LENS-M's init
            self.gw = nn.Linear(16, 1); nn.init.zeros_(self.gw.weight); nn.init.zeros_(self.gw.bias)

    def forward(self, cls, S, M=None):
        if self.cfg == "MP":
            return self.mhead(M)
        logit = L.SCALE * F.normalize(cls, dim=-1) @ F.normalize(self.w, dim=-1).t()
        if self.cfg == "LP":
            return logit
        if self.cfg == "LENS-AG":
            sp = self.head[0](pps_features(S, "V2b"))                        # s_PPS as the PPS head sees it (after its BatchNorm)
            zm = self.mhead[2](self.mhead[1](self.mhead[0](M)))              # 32-d hidden layer of the motion MLP
            gate = self.gm + self.gw(self.gnet(torch.cat([sp, zm], 1))).squeeze(1)
            return logit + self.g * self.head[1:](sp) + gate.unsqueeze(1) * self.mhead[4](self.mhead[3](zm))
        delta = self.head(pps_features(S, "V2b" if self.cfg in ("LENS-M", "LENS-FG") else self.cfg))
        out = logit + (self.g * delta if self.g is not None else delta)
        if self.cfg == "LENS-M":
            out = out + self.gm * self.mhead(M)
        if self.cfg == "LENS-FG":
            out = out + self.gm * self.p_raw(cls).unsqueeze(1) * self.mhead(M)
        return out

    def p_raw(self, cls):
        """Probability that the clip is raw footage (surveillance / phone) rather than edited footage, from fixed prompts."""
        return torch.softmax(100.0 * F.normalize(cls, dim=-1) @ self.fe.t(), 1)[:, :4].sum(1)

    def anchor_penalty(self):
        return L.ANCHOR_W * ((self.w - self.w0) ** 2).sum()


@torch.no_grad()
def predict(model, feats, rows):
    model.eval(); r = torch.as_tensor(np.asarray(rows))
    return torch.softmax(model(*feats.get(r)), 1)[:, 1].numpy() if len(r) else np.zeros(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["LP", "V2a", "V2b", "-G", "-T", "-Sp", "LENS-M", "LENS-AG", "LENS-FG", "MP"], required=True)
    ap.add_argument("--vrex", type=int, default=None, help="for -G/-T/-Sp: 1 if the selected candidate is V2b")
    ap.add_argument("--split", default="audited")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--allow-test", action="store_true")
    a = ap.parse_args()
    vrex = {"V2b": 1, "V2a": 0, "LP": 0, "LENS-M": 1, "LENS-AG": 1, "LENS-FG": 1, "MP": 0}.get(a.config, a.vrex)
    assert vrex in (0, 1), "--vrex required for ablations"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    name = f"v2_{a.split.replace(':', '_')}_{a.config}{'_vrex' if vrex and a.config not in ('V2b',) else ''}_seed{a.seed}{'_pilot' if a.pilot else ''}"
    od = L.OUT / name; od.mkdir(parents=True, exist_ok=True)
    df, held = L.load_split(a.split)
    tr = df[df.split == "train"].reset_index(drop=True); va = df[df.split == "val"].reset_index(drop=True)
    if held not in (None, "std") and a.pilot:
        te = df[(df.split == "test") & (df.frozen_split == "val")].reset_index(drop=True)
    else:
        te = df[df.split == "test"].reset_index(drop=True) if a.allow_test else df.iloc[:0]
    feats = Compact("bench")
    model = V2(a.config, L.text_anchors())
    w = D.strategy_weights(tr, "hybrid", cap=0.05)
    rows = torch.as_tensor(tr.feat_row.values); ytr = torch.tensor(tr.binary_label.values)
    dsid = torch.tensor(pd.factorize(tr.dataset)[0])
    g = torch.Generator().manual_seed(a.seed); bs = 128; steps = len(tr) // bs; epochs = 30
    nowd = {"w", "g", "gm"}
    groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and n not in nowd], "weight_decay": 0.05},
              {"params": [p for n, p in model.named_parameters() if p.requires_grad and n in nowd], "weight_decay": 0.0}]
    opt = torch.optim.AdamW([x for x in groups if x["params"]], lr=3e-4)
    total, warm = steps * epochs, steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(total - warm, 1))))
    log = open(od / "log.jsonl", "w"); t0 = time.time(); best, best_state, best_ep = -1, None, -1
    for ep in range(epochs):
        model.train()
        idx = torch.multinomial(torch.tensor(w), steps * bs, replacement=True, generator=g)
        for s in range(steps):
            b = idx[s * bs:(s + 1) * bs]; r = rows[b]
            ce = F.cross_entropy(model(*feats.get(r)), ytr[b], label_smoothing=0.1, reduction="none")
            loss = ce.mean() + model.anchor_penalty()
            if vrex:
                per = torch.stack([ce[dsid[b] == k].mean() for k in dsid[b].unique()])
                if len(per) > 1: loss = loss + 1.0 * per.var(unbiased=False)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        pv = predict(model, feats, va.feat_row.values)
        auc = D.binary_metrics(va.binary_label, pv, 0.5)["auc"]
        log.write(json.dumps(dict(epoch=ep + 1, val_auc=auc, gate=None if getattr(model, "g", None) is None else float(model.g))) + "\n"); log.flush()
        if auc > best: best, best_ep, best_state = auc, ep + 1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); torch.save(best_state, od / "best.pt")
    pv = predict(model, feats, va.feat_row.values); thr = D.select_threshold(va.binary_label, pv)
    per = D.per_group(va, pv, thr, "dataset")
    res = dict(config=a.config, vrex=vrex, split=a.split, seed=a.seed, pilot=a.pilot, threshold_selected_on_val=thr, best_epoch=best_ep,
               gate=None if getattr(model, "g", None) is None else float(model.g),
               gate_motion=None if getattr(model, "gm", None) is None else float(model.gm), minutes=(time.time() - t0) / 60,
               n_train=len(tr), n_val=len(va), n_scored_test=len(te),
               val=dict(overall=D.binary_metrics(va.binary_label, pv, thr), per_dataset=per,
                        macro_dataset_auc=D.macro_auc(per)[0], worst_dataset_auc=D.macro_auc(per)[1]))
    va.assign(p_violent=pv).to_csv(od / "predictions_val.csv", index=False)
    if len(te):
        pt = predict(model, feats, te.feat_row.values); key = "heldout_val_rows" if a.pilot else "test"
        res[key] = dict(overall=D.binary_metrics(te.binary_label, pt, thr), per_dataset=D.per_group(te, pt, thr, "dataset"))
        if held is None:
            res[key].update(macro_dataset_auc=D.macro_auc(res[key]["per_dataset"])[0],
                            within_group_auc=D.within_group_auc(te.binary_label, pt, te.component_id))
        te.assign(p_violent=pt).to_csv(od / f"predictions_{key}.csv", index=False)
        if held == "std":
            yv, pvv = L.video_level(va, pv); yt, ptv = L.video_level(te, pt); vthr = D.select_threshold(yv, pvv)
            res["test_video"] = dict(threshold_selected_on_val_videos=vthr, n_videos=len(yt), overall=D.binary_metrics(yt, ptv, vthr))
    if held is None and a.allow_test:                      # F3 external
        ex = Compact("external")
        ext = pd.read_csv(L.ROOT / "data_tables/manifests/external_windows.csv")
        pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True)
        assert len(pool) == len(ex.index) and (pool.path.values == ex.index.path.values).all()
        pe = predict(model, ex, np.arange(len(pool))); keep = (pool.excluded_precaution == 0).values
        res["external"] = {}
        for bname, gg in pool[keep].groupby("benchmark"):
            pp = pe[gg.index.values]; gg = gg.reset_index(drop=True)
            res["external"][bname] = dict(overall=D.binary_metrics(gg.binary_label, pp, thr),
                                          per_category=D.per_group(gg, pp, thr, "category"),
                                          within_video_auc=D.within_group_auc(gg.binary_label, pp, gg.video))
        pool.assign(p_violent=pe).to_csv(od / "predictions_external.csv", index=False)
    json.dump(res, open(od / "summary.json", "w"), indent=2, default=float)
    show = {k: round(res[k]["overall"]["auc"], 2) for k in ("val", "test", "heldout_val_rows") if k in res}
    if "external" in res: show.update({b: round(v["overall"]["auc"], 2) for b, v in res["external"].items()})
    print(name, json.dumps(show), "gate", res["gate"], flush=True)


if __name__ == "__main__":
    main()
