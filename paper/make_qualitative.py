#!/usr/bin/env python3
"""Qualitative figure: LENS-Net (3-seed mean) and fine-tuned VideoMAE-B violence probability over entire unseen videos.
Selection rule (fixed before any scoring): first eligible video (alphabetical) with duration 20-120 s in each of
XD-test B1 (fighting), XD-test B4 (riot), UCF-test Fighting, UCF-test Shoplifting (non-violent hard negative).
Eligible = in the external evaluation pool, not excluded for contamination or precaution."""
import json, re, sys
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch, torch.nn.functional as F
ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[1]); LN = ROOT / "workspace"; OUT = Path(__file__).parent / "figs"
sys.path.insert(0, str(ROOT)); sys.path[:0] = [str(LN.parent / "src" / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import lensnet_setup as L, lensnet as V, clip_tail as C, train_video_baseline as TB
from video_data import cnn_bigru_model as mm
from video_data.build_external_windows import decode_video
from video_data import label_schema as S

ext = pd.read_csv(ROOT / "data_tables/manifests/external_windows.csv")
pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0) & (ext.excluded_precaution == 0)]
vids = pool.groupby(["benchmark", "video"]).agg(path=("path", "first"), category=("category", "first"), fps=("fps", "first"), total=("total_frames", "first")).reset_index()
vids["dur"] = vids.total / vids.fps
SEL = [("XD-test", "B1", "XD-Violence: fighting"), ("XD-test", "B4", "XD-Violence: riot"), ("UCF-test", "Fighting", "UCF-Crime: fighting"),
       ("UCF-test", "Shoplifting", "UCF-Crime: shoplifting (non-violent)")]
chosen = []
for b, cat, title in SEL:
    c = vids[(vids.benchmark == b) & (vids.category == cat) & vids.dur.between(20, 120)].sort_values("video")
    if len(c): chosen.append((title, c.iloc[0]))
    else: print("no eligible video for", b, cat)

def gt_intervals(bench, name):
    if bench == "UCF-test":
        for line in open(ROOT / "raw_data/annotations/ucf_temporal_annotation.txt"):
            t = line.split()
            if t and t[0] == name:
                n = list(map(int, t[2:6])); return [(n[i], n[i + 1]) for i in (0, 2) if n[i] >= 0 and n[i + 1] > n[i]]
        return []
    for line in open(ROOT / "raw_data/annotations/xd_violence_annotations.txt"):
        t = line.split()
        if t and t[0] == name:
            n = list(map(int, t[1:])); return [(n[i], n[i + 1]) for i in range(0, len(n) - 1, 2)]
    return []

dev = torch.device("cuda")
clip_v = C.load_clip(dev, "fp16"); clip_v.output_tokens = True; proj = clip_v.proj
from transformers import VideoMAEForVideoClassification
vmae = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").to(dev).half().eval()
E = torch.tensor(np.load(L.FEAT / "text_prompts.npy"))
lens_models = []
for s in (42, 43, 44):
    m = V.V2("LENS-M", L.text_anchors()); m.load_state_dict(torch.load(LN / f"runs/v2_audited_LENS-M_vrex_seed{s}/best.pt")); m.eval(); lens_models.append(m)
ft = []
for d in sorted((LN / "runs_baselines").glob("A1_videomae_seed*")):
    if (d / "summary.json").exists():
        m = TB.Net("videomae").to(dev); m.load_state_dict(torch.load(d / "best.pt", map_location=dev)); m.eval(); ft.append(m)
CM = np.array([0.48145466, 0.4578275, 0.40821073], np.float32); CS = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

@torch.no_grad()
def score_windows(jpeg_lists):
    out_l, out_v = [], []
    for jp in jpeg_lists:
        fr = [cv2.cvtColor(cv2.imdecode(np.frombuffer(jp[int(t)], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for t in mm.temporal_indices(len(jp), 16)]
        f224 = np.stack([cv2.resize(f, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255 for f in fr])
        xc = torch.from_numpy(((f224 - CM) / CS).transpose(0, 3, 1, 2)).to(dev).half()
        cls, pt = clip_v(xc)
        g = F.adaptive_avg_pool2d(pt.view(-1, 16, 16, 1024).permute(0, 3, 1, 2).float(), 4).flatten(2).transpose(1, 2).half() @ proj
        z = F.normalize(torch.cat([cls.unsqueeze(1), g], 1).float(), dim=-1).cpu()            # (16,17,768)
        clsm, Ssig = z[:, 0].mean(0, keepdim=True), (z @ E.t()).unsqueeze(0)
        xi = mm.clip_tensor_from_jpegs(jp).unsqueeze(0).to(dev)                               # ImageNet-normalized (1,16,3,224,224)
        h = vmae.videomae(pixel_values=xi.half())[0]; mfeat = vmae.fc_norm(h.mean(1)).float().cpu()
        out_l.append(np.mean([torch.softmax(m(clsm, Ssig, mfeat), 1)[0, 1].item() for m in lens_models]))
        with torch.autocast("cuda", dtype=torch.float16):
            out_v.append(np.mean([torch.softmax(m(xi).float(), 1)[0, 1].item() for m in ft]) if ft else np.nan)
    return np.array(out_l), np.array(out_v)

res = []
for title, r in chosen:
    span = int(round(S.WINDOW_SECONDS * r.fps)); wins = [(s, s + span - 1) for s in range(0, int(r.total) - span + 1, span)][:120]
    frames = decode_video((r.path, wins)); ok = [i for i, f in enumerate(frames) if f is not None]
    pl, pv = score_windows([frames[i] for i in ok])
    t = np.array([(wins[i][0] + wins[i][1]) / 2 / r.fps for i in ok])
    thumbs = []
    for tt in np.linspace(0.08, 0.92, 5):
        k = ok[int(tt * (len(ok) - 1))]; jp = frames[k][len(frames[k]) // 2]
        thumbs.append(cv2.cvtColor(cv2.imdecode(np.frombuffer(jp, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)[::2, ::2])
    res.append(dict(title=title, video=r.video, bench=r.benchmark, t=t, lens=pl, vmae=pv, gt=[(a / r.fps, b / r.fps) for a, b in gt_intervals(r.benchmark, r.video)],
                    thumbs=thumbs, thumb_t=[(wins[ok[int(tt * (len(ok) - 1))]][0] + span / 2) / r.fps for tt in np.linspace(0.08, 0.92, 5)], dur=r.dur))
    print(title, r.video, f"{r.dur:.0f}s", len(ok), "windows; mean P lens", pl.mean().round(3), "gt", res[-1]["gt"], flush=True)
np.save(OUT / "qualitative_data.npy", np.array(res, dtype=object), allow_pickle=True)
json.dump([dict(title=x["title"], video=x["video"], bench=x["bench"], gt=x["gt"], t=x["t"].tolist(), lens=x["lens"].tolist(), vmae=x["vmae"].tolist()) for x in res],
          open(OUT / "qualitative_data.json", "w"), indent=1)
