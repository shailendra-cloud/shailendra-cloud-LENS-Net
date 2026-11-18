#!/usr/bin/env python3
"""CNN-BiGRU baseline (internal name: MAST-Net): model definition and frame sampler.

The class and function names keep the internal name of the baseline (MASTNet, load_mastnet), because the
released run folders and prediction files use it (for example MAST-Net_seed42.csv). The paper calls this
model the CNN-BiGRU detector.

Fidelity contract
-----------------
Everything below is copied from MAST_NET.py, the original (v1) training script of the baseline, which is
not part of this repository, with ONE kind of change: the architecture switches MAST_NET.py read from its
global CFG object (use_motion, motion_pretrained, use_interaction, whiten, grad_checkpoint, img_size) are
explicit constructor arguments. Parameter names, module order and the forward computation are unchanged,
so v1 checkpoints load with strict=True and produce identical outputs.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import models as tvm

try:
    import timm
    HAS_TIMM = True
except ImportError:  # pragma: no cover
    HAS_TIMM = False

# Preprocessing constants -- identical to MAST_NET.py CFG defaults.
N_FRAMES_CACHE = 32
N_FRAMES_MODEL = 16
CACHE_SIZE = 256
IMG_SIZE = 224
JPEG_QUALITY = 92
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================================================
#  Model components (verbatim from MAST_NET.py run_training)
# ============================================================================================
class BackboneWrapper(nn.Module):
    def __init__(self, net, feat_dim, kind):
        super().__init__(); self.net, self.feat_dim, self.kind = net, feat_dim, kind

    def forward(self, x):
        if self.kind == "timm":
            o = self.net(x)
            if o.ndim == 4: o = F.adaptive_avg_pool2d(o, 1).flatten(1)
            elif o.ndim == 3: o = o.mean(1)
            return o
        return self.net(x).flatten(1)


def build_backbone(kind, pretrained=True, img_size=IMG_SIZE):
    if kind == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Identity(); return BackboneWrapper(m, 2048, "resnet")
    if kind == "efficientvit_b0":
        m = timm.create_model("efficientvit_b0.r224_in1k", pretrained=pretrained, num_classes=0)
        with torch.no_grad():
            feat_dim = m(torch.zeros(1, 3, img_size, img_size)).shape[-1]
        return BackboneWrapper(m, feat_dim, "timm")
    if kind == "effnetv2s":
        if HAS_TIMM:
            for nm in ["tf_efficientnetv2_s.in21k_ft_in1k", "efficientnetv2_rw_s.ra2_in1k"]:
                try:
                    m = timm.create_model(nm, pretrained=pretrained, num_classes=0)
                    return BackboneWrapper(m, m.num_features, "timm")
                except Exception: continue
        m = tvm.efficientnet_v2_s(weights=tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Identity(); return BackboneWrapper(m, 1280, "resnet")
    raise ValueError(kind)


def set_trainable(bb, mode="partial"):
    if mode == "frozen":
        for p in bb.parameters(): p.requires_grad = False
        bb.eval()
    elif mode == "full":
        for p in bb.parameters(): p.requires_grad = True
    elif mode == "partial":
        ps = list(bb.parameters()); cut = int(len(ps) * 0.70)
        for i, p in enumerate(ps): p.requires_grad = i >= cut
    return bb


class MotionEncoder(nn.Module):
    def __init__(self, out_dim=256, width=32):
        super().__init__()
        def blk(i, o, s=2):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.SiLU(True),
                                 nn.Conv2d(o, o, 3, 1, 1, groups=o, bias=False), nn.BatchNorm2d(o), nn.SiLU(True))
        self.stem, self.b1 = blk(3, width), blk(width, width * 2)
        self.b2, self.b3 = blk(width * 2, width * 4), blk(width * 4, width * 8)
        self.proj = nn.Sequential(nn.Conv2d(width * 8, out_dim, 1, bias=False), nn.BatchNorm2d(out_dim), nn.SiLU(True))

    def forward(self, x):
        return F.adaptive_avg_pool2d(self.proj(self.b3(self.b2(self.b1(self.stem(x))))), 1).flatten(1)


class PretrainedMotionEncoder(nn.Module):
    """Same role and output width as MotionEncoder, backed by ImageNet weights."""
    def __init__(self, out_dim=256, img_size=IMG_SIZE, pretrained=True):
        super().__init__()
        self.net = timm.create_model("efficientvit_b0.r224_in1k", pretrained=pretrained, num_classes=0)
        with torch.no_grad():
            fd = self.net(torch.zeros(1, 3, img_size, img_size)).shape[-1]
        self.proj = nn.Sequential(nn.LayerNorm(fd), nn.Linear(fd, out_dim), nn.SiLU(True))

    def forward(self, x):
        return self.proj(self.net(x))


class AttentionPool(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        w = torch.softmax(self.score(x), dim=1)
        return (x * w).sum(1), w.squeeze(-1)


class GradientReversal(torch.autograd.Function):
    """DANN gradient reversal (Ganin & Lempitsky, 2016)."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class InteractionEncoder(nn.Module):
    """Encodes the per-frame (B, T, 8) person-interaction geometry descriptor."""
    def __init__(self, d_out=64, d_in=8, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, d_out), nn.GELU())

    def forward(self, g):
        return self.net(g)


class MASTNet(nn.Module):
    """v1 MAST-Net. Identical to MAST_NET.py except that the CFG switches are arguments.

    use_motion / motion_pretrained / use_interaction change the parameter set and are recoverable
    from a checkpoint's keys. `whiten` has NO parameters, so it cannot be recovered from weights:
    a whitened checkpoint loads cleanly into an unwhitened model and silently gives wrong
    outputs. Always pass it explicitly (load_mastnet records where it came from).
    """
    def __init__(self, backbone_kind="effnetv2s", d_model=512, gru_hidden=256, gru_layers=2,
                 heads=4, dropout=0.3, pretrained=True, freeze_mode="full", num_classes=7,
                 num_domains=2, blur_aug_prob=0.5, *, use_motion=True, motion_pretrained=False,
                 use_interaction=False, whiten=False, grad_checkpoint=False, img_size=IMG_SIZE):
        super().__init__()
        self.arch = dict(backbone_kind=backbone_kind, num_classes=num_classes, use_motion=use_motion,
                         motion_pretrained=motion_pretrained, use_interaction=use_interaction,
                         whiten=bool(whiten))
        self.blur_aug_prob = blur_aug_prob
        self.whiten = bool(whiten)
        self.backbone = set_trainable(build_backbone(backbone_kind, pretrained, img_size), freeze_mode)
        self.freeze_mode = freeze_mode
        if grad_checkpoint and hasattr(self.backbone.net, "set_grad_checkpointing"):
            try: self.backbone.net.set_grad_checkpointing(True)
            except Exception: pass
        d_rgb = self.backbone.feat_dim
        if not use_motion:
            self.motion = None
        elif motion_pretrained:
            self.motion = PretrainedMotionEncoder(256, img_size=img_size, pretrained=pretrained)
        else:
            self.motion = MotionEncoder(256)
        self.interaction = InteractionEncoder(64) if use_interaction else None
        d_in = d_rgb + (256 if use_motion else 0) + (64 if use_interaction else 0)
        self.fuse = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_model), nn.GELU(), nn.Dropout(dropout))
        self.rnn = nn.GRU(d_model, gru_hidden, num_layers=gru_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if gru_layers > 1 else 0.0)
        d_seq = gru_hidden * 2
        self.pool = AttentionPool(d_seq)
        self.norm = nn.LayerNorm(d_seq)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_seq, d_seq // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_seq // 2, num_classes))
        self.domain_head_cdan = nn.Sequential(
            nn.Linear(d_seq * num_classes, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(64, num_domains))
        self.domain_heads_mada = nn.ModuleList([
            nn.Sequential(nn.Linear(d_seq, 64), nn.ReLU(inplace=True),
                          nn.Dropout(dropout), nn.Linear(64, num_domains))
            for _ in range(num_classes)])
        self.domain_head = nn.Sequential(nn.Linear(d_seq, 64), nn.ReLU(inplace=True),
                                         nn.Dropout(dropout), nn.Linear(64, num_domains))

    def _encode(self, x, g=None):
        B, T = x.shape[:2]
        # motion is differenced from the raw input, before any appearance-only transform
        d = x[:, 1:] - x[:, :-1]; d = torch.cat([torch.zeros_like(x[:, :1]), d], dim=1)
        flat = x.flatten(0, 1)
        if self.whiten:
            v = flat.view(B, T, flat.shape[1], -1)
            mu = v.mean(dim=(1, 3), keepdim=True)
            sd = v.std(dim=(1, 3), keepdim=True).clamp_min(1e-5)
            flat = ((v - mu) / sd).view_as(flat)
        if self.training and self.blur_aug_prob > 0:
            clip_mask = torch.rand(B, device=x.device) < self.blur_aug_prob
            if clip_mask.any():
                sigma = float(np.random.uniform(1.0, 2.5))
                blurred = TF.gaussian_blur(flat, kernel_size=[9, 9], sigma=sigma)
                m = clip_mask.view(B, 1, 1, 1, 1).expand(B, T, *flat.shape[1:]).reshape(flat.shape)
                flat = torch.where(m, blurred, flat)
        if self.freeze_mode == "frozen":
            with torch.no_grad(): f_rgb = self.backbone(flat)
        else:
            f_rgb = self.backbone(flat)
        f_rgb = f_rgb.view(B, T, -1)
        if self.motion is not None:
            f = torch.cat([f_rgb, self.motion(d.flatten(0, 1)).view(B, T, -1)], dim=-1)
        else:
            f = f_rgb
        if self.interaction is not None:
            if g is None:
                g = torch.zeros(B, T, 8, device=x.device, dtype=f.dtype)
            f = torch.cat([f, self.interaction(g.to(f.dtype))], dim=-1)
        seq, _ = self.rnn(self.fuse(f))
        z, _ = self.pool(seq)
        return z

    def forward(self, x, g=None):
        return self.head(self.norm(self._encode(x, g)))

    def forward_with_domain(self, x, grl_lambda=1.0):
        # v1 behaviour kept exactly: this path calls _encode without g.
        zn = self.norm(self._encode(x))
        class_logits = self.head(zn)
        z_rev = GradientReversal.apply(zn, grl_lambda)
        domain_logits = self.domain_head(z_rev)
        return class_logits, domain_logits, z_rev


# ============================================================================================
#  Checkpoint loading
# ============================================================================================
def infer_arch(state, checkpoint_path=None, meta=None, whiten=None):
    """Recover constructor flags for a checkpoint. v1 checkpoints carry no arch metadata."""
    keys = list(state.keys())
    saved = (meta or {}).get("arch") if isinstance(meta, dict) else None
    arch = dict(saved) if isinstance(saved, dict) else {}
    arch.setdefault("use_interaction", any(k.startswith("interaction.") for k in keys))
    arch.setdefault("use_motion", any(k.startswith("motion.") for k in keys))
    arch.setdefault("motion_pretrained", any(k.startswith("motion.net.") for k in keys))
    arch.setdefault("num_classes", int(state["head.4.weight"].shape[0]))
    arch.setdefault("num_domains", int(state["domain_head.3.weight"].shape[0]))
    if whiten is not None:
        arch["whiten"], src = bool(whiten), "argument"
    elif "whiten" in arch:
        src = "checkpoint metadata"
    else:
        name = Path(checkpoint_path).name if checkpoint_path else ""
        arch["whiten"], src = ("_whiten" in name), "filename (v1 checkpoints store no arch metadata)"
    arch["whiten_source"] = src
    return arch


def load_mastnet(checkpoint_path, device="cpu", strict=True, backbone_kind="efficientvit_b0", whiten=None):
    """Build a MASTNet matching a checkpoint and load it. Returns (model, info)."""
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
    arch = infer_arch(state, checkpoint_path, ck if isinstance(ck, dict) else None, whiten)
    model = MASTNet(backbone_kind=backbone_kind, pretrained=False, num_classes=arch["num_classes"],
                    num_domains=arch["num_domains"], use_motion=arch["use_motion"],
                    motion_pretrained=arch["motion_pretrained"],
                    use_interaction=arch["use_interaction"], whiten=arch["whiten"])
    result = model.load_state_dict(state, strict=strict)
    model.to(device).eval()
    info = {"arch": arch, "classes": ck.get("classes") if isinstance(ck, dict) else None,
            "epoch": ck.get("epoch") if isinstance(ck, dict) else None,
            "best_f1": ck.get("best_f1") if isinstance(ck, dict) else None,
            "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}
    return model, info


@torch.no_grad()
def predict_proba_tta(model, x, g=None, use_amp=True):
    """Softmax averaged over the clip and its horizontal flip (v1 evaluate_tta)."""
    with torch.autocast(device_type="cuda", enabled=bool(use_amp and x.is_cuda)):
        lg = model(x, g)
        lf = model(torch.flip(x, dims=[-1]), g)
    return (torch.softmax(lg.float(), 1) + torch.softmax(lf.float(), 1)) / 2


# ============================================================================================
#  Frame sampler: shared by cache building, evaluation and inference
# ============================================================================================
def sample_clip_frames(path, n_frames=N_FRAMES_CACHE, cache_size=CACHE_SIZE, start_f=None, end_f=None):
    """Decode n_frames evenly spaced over [start_f, end_f] (inclusive), resized to cache_size.

    Same logic as MAST_NET._cache_one_video up to the JPEG step. Returns BGR uint8 frames or None.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release(); return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); frames = []
    lo = 0 if start_f is None else max(0, int(start_f))
    hi = max(total - 1, 0) if end_f is None else min(int(end_f), max(total - 1, 0))
    if hi < lo: lo, hi = 0, max(total - 1, 0)
    if total > 0:
        for i in np.linspace(lo, hi, n_frames).round().astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok: break
            frames.append(fr)
    cap.release()
    if not frames: return None
    while len(frames) < n_frames: frames.append(frames[-1].copy())
    frames = frames[:n_frames]
    return [cv2.resize(fr, (cache_size, cache_size), interpolation=cv2.INTER_AREA) for fr in frames]


def encode_jpeg_frames(frames, quality=JPEG_QUALITY):
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    buf = []
    for fr in frames:
        good, e = cv2.imencode(".jpg", fr, enc)
        if not good: return None
        buf.append(e.tobytes())
    return buf


def cache_one_clip(args):
    """Drop-in replacement for MAST_NET._cache_one_video: same argument tuple, same bytes."""
    if len(args) == 6:
        path, n_frames, cache_size, jpeg_quality, start_f, end_f = args
    else:
        path, n_frames, cache_size, jpeg_quality = args
        start_f = end_f = None
    cv2.setNumThreads(1)
    frames = sample_clip_frames(path, n_frames, cache_size, start_f, end_f)
    return None if frames is None else encode_jpeg_frames(frames, jpeg_quality)


def temporal_indices(t_cache, n_model=N_FRAMES_MODEL):
    """Evaluation-time temporal selection (CroppedClipDataset._sample_temporal, train=False)."""
    if n_model >= t_cache:
        return np.clip(np.arange(n_model), 0, t_cache - 1)
    return np.linspace(0, t_cache - 1, n_model).round().astype(int)


def jpegs_from_blob(blob, offsets_row):
    return [blob[offsets_row[t]:offsets_row[t + 1]] for t in range(len(offsets_row) - 1)]


def clip_tensor_from_jpegs(jpegs, n_model=N_FRAMES_MODEL, img_size=IMG_SIZE):
    """Evaluation transform (CroppedClipDataset.__getitem__, train=False) -> float (T,3,H,W)."""
    tsel = temporal_indices(len(jpegs), n_model)
    frames = []
    for t in tsel:
        b = jpegs[int(t)]
        arr = np.frombuffer(b, dtype=np.uint8) if isinstance(b, (bytes, bytearray)) else b
        frames.append(cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    frames = [cv2.resize(f, (img_size, img_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) for f in frames]
    x = np.stack(frames, 0) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).float()


def clip_tensor_from_video(path, start_f=None, end_f=None):
    """Inference path: the exact training preprocessing, including the q92 JPEG round-trip the
    cached training frames went through, so inference inputs match the training distribution."""
    frames = sample_clip_frames(path, N_FRAMES_CACHE, CACHE_SIZE, start_f, end_f)
    if frames is None: return None
    jpegs = encode_jpeg_frames(frames, JPEG_QUALITY)
    return None if jpegs is None else clip_tensor_from_jpegs(jpegs)
