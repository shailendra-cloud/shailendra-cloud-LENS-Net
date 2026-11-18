"""Revision (reviewer response): one loader for every model's per-window predictions on the audited test split and
the nine unseen test sets, with labels, bootstrap clusters and the near-duplicate keep mask. Used by the re-analyses
(per-seed paired tests, multiplicity control, with/without filtering, symmetric fusion). No model is retrained."""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]); L = ROOT / "workspace"
SEEDS = (42, 43, 44)
SETS = ["Test", "XD", "UCF", "SCFD", "SCVD", "VioPeru", "AIRTLab", "CCTV-F", "Movies", "UBI-Fights"]
UNSEEN = SETS[1:]
# model key -> (run-name stem for test/external, stem in the preds/ dirs, family)
MODELS = {
    "LENS-Net": "v2_audited_LENS-M_vrex", "CLIP probe": "v2_audited_LP", "w/o motion": "v2_audited_V2b",
    "Motion probe": "v2_audited_MP", "CLIP zero-shot": "ZS", "VadCLIP": "VadCLIP-L14_audited_lr0.0001",
    "X-CLIP FT": "A1_xclip_lr5e-06", "VideoMAE-B FT": "A1_videomae", "Video Swin-T FT": "A1_swin3d_t",
    "R(2+1)D-18 FT": "A1_r2plus1d_18", "CNN-BiGRU": "MAST-Net",
}


def _run_dir(stem, seed):
    if stem == "ZS": return L / "runs/audited_ZS_seed42"
    if stem == "MAST-Net": return ROOT / f"source_benchmark/eval/binary_hybrid_seed{seed}_cap8"
    if stem.startswith("VadCLIP"): return L / f"runs_vadclip/{stem}_seed{seed}"
    if stem.startswith("A1_"): return L / f"runs_baselines/{stem}_seed{seed}"
    return L / f"runs/{stem}_seed{seed}"


def _ext_frame():
    d = pd.read_csv(L / "runs/v2_audited_LENS-M_vrex_seed42/predictions_external.csv")
    return d


class Data:
    def __init__(self):
        self.sets = {}
        te = pd.read_csv(L / "runs/v2_audited_LENS-M_vrex_seed42/predictions_test.csv")
        self.sets["Test"] = dict(y=te.binary_label.values, cl=te.component_id.values, keep=np.ones(len(te), bool), key=te.segment_id.values)
        ex = _ext_frame()
        base = (ex.binary_label.isin([0, 1])) & (ex.excluded_contamination == 0) & (ex.decode_failed == 0)
        self.ext_base = base.values
        for b, nm in (("XD-test", "XD"), ("UCF-test", "UCF")):
            m = base.values & (ex.benchmark == b).values
            self.sets[nm] = dict(rows=np.where(m)[0], y=ex.binary_label.values[m], cl=ex.video.values[m], keep=np.ones(m.sum(), bool),
                                 excl=np.zeros(m.sum(), bool))
        for d, f, groups in (("fresh", "fresh_windows.csv", [("SCFD", "SCFD"), ("SCVD", "SCVD")]),
                             ("conf", "conf_windows.csv", [("VioPeru", "VioPeru"), ("AIRTLab", "AIRTLab")]),
                             ("cfights", "cfights_windows.csv", [("CCTV-F", None)]),
                             ("conf2", "conf2_windows.csv", [("Movies", "Movies"), ("UBI-Fights", "UBI-Fights")])):
            w = pd.read_csv(L / d / f); w = w[w.decode_failed == 0].reset_index(drop=True)
            x = pd.read_csv(L / d / "near_duplicate_exclusion.csv"); assert len(x) == len(w)
            excl = x.excluded_near_duplicate.values == 1
            cl = w.cluster.values if "cluster" in w else w.video.values
            for nm, bench in groups:
                m = ((w.category == "CCTV") if nm == "CCTV-F" else (w.benchmark == bench)).values
                self.sets[nm] = dict(dir=d, rows=np.where(m)[0], y=w.binary_label.values[m], cl=cl[m], keep=~excl[m], excl=excl[m],
                                     nwin=len(w))

    def p(self, model, seed, s):
        """Probabilities of `model` (seed) on set `s`, aligned with self.sets[s] (all rows, filter via keep)."""
        stem = MODELS[model]; S = self.sets[s]
        if s == "Test":
            d = pd.read_csv(_run_dir(stem, seed) / "predictions_test.csv")
            if "segment_id" in d: d = d.set_index("segment_id").loc[S["key"]]
            return d.p_violent.values
        if s in ("XD", "UCF"):
            d = pd.read_csv(_run_dir(stem, seed) / "predictions_external.csv").p_violent.values
            assert len(d) == len(self.ext_base), (model, len(d))
            return d[S["rows"]]
        if stem == "VadCLIP-L14_audited_lr0.0001" and S["dir"] in ("fresh", "conf"):
            d = pd.read_csv(_run_dir(stem, seed) / f"predictions_{S['dir']}.csv").p_violent.values
        else:
            f = L / S["dir"] / "preds" / (f"{stem}.csv" if stem == "ZS" else f"{stem}_seed{seed}.csv")
            d = pd.read_csv(f).p_violent.values
        assert len(d) == S["nwin"], (model, s, len(d), S["nwin"])
        return d[S["rows"]]

    def seeds(self, model):
        return (42,) if model == "CLIP zero-shot" else SEEDS

    def probs(self, model, s):
        """list of per-seed probability arrays; LENS-Net-A = same-seed average of LENS-Net and the CLIP probe."""
        if model == "LENS-Net-A":
            return [(a + b) / 2 for a, b in zip(self.probs("LENS-Net", s), self.probs("CLIP probe", s))]
        if model.startswith("SYM:"):              # symmetric fusion control: model averaged with the CLIP probe
            m = model[4:]
            return [(a + b) / 2 for a, b in zip(self.probs(m, s), self.probs("CLIP probe", s))]
        return [self.p(model, r, s) for r in self.seeds(model)]


SEEDS10 = tuple(range(42, 52))
TEN = ("LENS-Net", "CLIP probe", "w/o motion", "Motion probe")


class Data10(Data):
    """Same sets; the four frozen-feature heads use seeds 42-51 (B7). Seeds >= 45 of the unseen sets beyond XD/UCF
    come from review/preds_seeds/ (scored from the stored compact features)."""
    def p(self, model, seed, s):
        if seed < 45 or s in ("Test", "XD", "UCF"): return super().p(model, seed, s)
        S = self.sets[s]; d = pd.read_csv(L / "review/preds_seeds" / S["dir"] / f"{MODELS[model]}_seed{seed}.csv").p_violent.values
        assert len(d) == S["nwin"]; return d[S["rows"]]

    def seeds(self, model):
        return SEEDS10 if model in TEN else super().seeds(model)
