#!/usr/bin/env python3
"""Binary pipeline shared pieces: window datasets, v1-identical transforms, sampling strategies, metrics."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from video_data import cnn_bigru_model as mm
from video_data.frame_store import content_key

CLASSES = ("NON_VIOLENCE", "VIOLENCE")


# ------------------------------------------------------------------ transforms
def _decode(b):
    arr = np.frombuffer(b, np.uint8) if isinstance(b, (bytes, bytearray)) else np.asarray(b)
    return cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def train_clip_tensor(jpegs, rng, n_model=mm.N_FRAMES_MODEL, img_size=mm.IMG_SIZE):
    """v1 CroppedClipDataset training transform: stratified temporal jitter, scale-crop 0.8-1.0, h-flip."""
    T = len(jpegs)
    if n_model >= T:
        tsel = np.clip(np.arange(n_model), 0, T - 1)
    else:
        b = np.linspace(0, T, n_model + 1)
        tsel = np.array([rng.randint(int(b[i]), max(int(b[i + 1]), int(b[i]) + 1)) for i in range(n_model)]).clip(0, T - 1)
    frames = [_decode(jpegs[int(t)]) for t in tsel]
    H, W = frames[0].shape[:2]
    sc = rng.uniform(0.8, 1.0); ch, cw = int(H * sc), int(W * sc)
    y0 = rng.randint(0, H - ch + 1); x0 = rng.randint(0, W - cw + 1); flip = rng.rand() < 0.5
    out = []
    for f in frames:
        f = cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        if flip: f = f[:, ::-1]
        out.append(f.astype(np.float32))
    x = np.stack(out, 0) / 255.0
    x = (x - mm.IMAGENET_MEAN) / mm.IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).float()


class WindowDataset(Dataset):
    """Yields (clip tensor, binary label, row index). Frames come from a FrameStore; labels from the manifest."""
    def __init__(self, df, store, train, decode="seek"):
        self.df = df.reset_index(drop=True)
        self.v1_rows = (self.df.v1_cache_row.fillna(-1).astype(int).values if "v1_cache_row" in self.df
                        else np.full(len(self.df), -1))
        self.keys = [content_key(p, s, e, decode) for p, s, e in zip(self.df.path, self.df.start_frame, self.df.end_frame)]
        self.labels = self.df.binary_label.astype(int).values
        self.store, self.train = store, train
        self.rng = np.random.RandomState(0)
        missing = sum(not store.has(int(r), k) for r, k in zip(self.v1_rows, self.keys))
        if missing:
            raise RuntimeError(f"{missing} of {len(self.df)} windows have no frames in the store")

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        jp = self.store.jpegs(int(self.v1_rows[i]), self.keys[i])
        x = train_clip_tensor(jp, self.rng) if self.train else mm.clip_tensor_from_jpegs(jp)
        return x, int(self.labels[i]), i


def worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.RandomState((torch.initial_seed() + worker_id) % (2 ** 32))


# ------------------------------------------------------------------ sampling strategies
def strategy_weights(train_df, strategy, cap=0.05):
    """Per-window sampling weights for the training split. Every strategy gives each binary label equal
    total mass (0.5), so no class weights are needed. The evidence unit is the near-duplicate group
    (component_id): windows sharing footage are one piece of evidence.

      unit     every group of a label gets equal mass, regardless of dataset
      dataset  every dataset of a label gets equal mass; groups equal within a dataset
      hybrid   dataset mass ~ sqrt(windows), group mass ~ sqrt(windows) within its dataset,
               and no single group may exceed `cap` of its label's mass
    """
    df = train_df.reset_index(drop=True)
    w = np.zeros(len(df))
    for lab in (0, 1):
        m = (df.binary_label.values == lab)
        g = df[m]
        n_c = g.groupby("component_id").size()
        if strategy == "unit":
            mass = pd.Series(1.0, index=n_c.index)
        elif strategy == "dataset":
            ds_of = g.groupby("component_id").dataset.first()
            per_ds = ds_of.value_counts()
            mass = 1.0 / ds_of.map(per_ds)
        elif strategy == "hybrid":
            ds_of = g.groupby("component_id").dataset.first()
            ds_windows = g.dataset.value_counts()
            ds_mass = np.sqrt(ds_windows) / np.sqrt(ds_windows).sum()
            sq = np.sqrt(n_c)
            mass = sq / ds_of.map(sq.groupby(ds_of).sum()) * ds_of.map(ds_mass)
            mass = mass / mass.sum()
            for _ in range(100):                                    # cap and redistribute
                over = mass > cap
                if not over.any(): break
                excess = (mass[over] - cap).sum(); mass[over] = cap
                free = ~over
                mass[free] += excess * mass[free] / mass[free].sum()
        else:
            raise ValueError(strategy)
        mass = mass / mass.sum() * 0.5
        w[m] = g.component_id.map(mass).values / g.component_id.map(n_c).values
    return w


def effective_mass(train_df, w):
    df = train_df.reset_index(drop=True).assign(w=w)
    by_ds = (df.groupby("dataset").w.sum() / df.w.sum() * 100).round(1).to_dict()
    comp = df.groupby(["binary_label", "component_id"]).w.sum()
    top = float((comp / df.groupby("binary_label").w.sum().reindex(comp.index.get_level_values(0)).values).max() * 100)
    eff_n = float(df.w.sum() ** 2 / (df.w ** 2).sum())
    return {"dataset_share_pct": by_ds, "max_group_share_of_label_pct": round(top, 2), "effective_sample_size": round(eff_n)}


# ------------------------------------------------------------------ metrics
def binary_metrics(y, p, thr):
    y = np.asarray(y).astype(int); p = np.asarray(p, float); yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    both = len(set(y.tolist())) == 2
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else float("nan")
    return dict(n=int(len(y)), n_violence=int(y.sum()), n_non_violence=int(len(y) - y.sum()), threshold=float(thr),
                auc=float(roc_auc_score(y, p) * 100) if both else float("nan"),
                ap=float(average_precision_score(y, p) * 100) if both else float("nan"),
                accuracy=float((tp + tn) / len(y) * 100), precision=float(prec * 100), recall=float(rec * 100),
                f1=float(2 * prec * rec / (prec + rec) * 100) if (prec + rec) and rec == rec else 0.0,
                fpr=float(fp / (fp + tn) * 100) if fp + tn else float("nan"),
                fnr=float(fn / (fn + tp) * 100) if fn + tp else float("nan"),
                confusion=[[int(tn), int(fp)], [int(fn), int(tp)]])


def select_threshold(y, p):
    """Threshold maximising F1 on the given (validation) scores; ties go to the higher threshold."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    cands = np.unique(np.quantile(p, np.linspace(0, 1, 1001)))
    best_t, best_f = 0.5, -1.0
    for t in cands:
        f = binary_metrics(y, p, t)["f1"]
        if f > best_f or (f == best_f and t > best_t): best_t, best_f = float(t), f
    return best_t


def per_group(df, p, thr, col):
    out = {}
    for k, g in df.groupby(col):
        idx = g.index.values
        out[str(k)] = binary_metrics(df.binary_label.values[idx], np.asarray(p)[idx], thr)
    return out


def macro_auc(per):
    aucs = [m["auc"] for m in per.values() if m["auc"] == m["auc"]]
    return (float(np.mean(aucs)), float(np.min(aucs))) if aucs else (float("nan"), float("nan"))


def within_group_auc(y, p, groups):
    """Pairwise AUC counted only within groups holding both labels (background held constant)."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float); groups = np.asarray(groups)
    num = den = 0.0; ng = 0
    for g in np.unique(groups):
        m = groups == g; yy = y[m]
        npos, nneg = int(yy.sum()), int(len(yy) - yy.sum())
        if npos and nneg:
            num += roc_auc_score(yy, p[m]) * npos * nneg; den += npos * nneg; ng += 1
    return (float(num / den * 100) if den else float("nan")), ng


@torch.no_grad()
def predict(model, loader, device, tta=True, p_violent_fn=None, use_amp=True):
    """Return per-row P(VIOLENCE) in dataset order."""
    model.eval()
    n = len(loader.dataset); out = np.zeros(n, np.float64)
    for x, _, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=bool(use_amp and device.type == "cuda")):
            lg = model(x)
            prob = torch.softmax(lg.float(), 1)
            if tta:
                prob = (prob + torch.softmax(model(torch.flip(x, dims=[-1])).float(), 1)) / 2
        pv = p_violent_fn(prob) if p_violent_fn else prob[:, 1]
        out[idx.numpy()] = pv.cpu().numpy()
    return out
