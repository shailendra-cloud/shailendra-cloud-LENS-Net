#!/usr/bin/env python3
"""Part G (PREREG_revision.md): further untouched confirmation - UCF-Crime TRAINING split (violent classes + normal
sample) and the HMDB51 human-interaction group (selection lists fixed in conf4/). Each batch: download its videos from
Kaggle (one file at a time) -> windows (conf3 rule, at most 64 per video) -> frame store c3_K with K = 20 + batch (so the
existing loaders, feature extractors and cleaners read it unchanged) -> CLIP compact and VideoMAE features -> scores of
every Part F candidate -> frame store and raw videos removed (features and scores are kept).
  confirm_fusion_rules_untouched.py batch I | all | dedup | clipft_c3 K
Labels are read only by the analysis script."""
import base64, io, json, os, shutil, subprocess, sys, time, urllib.parse, urllib.request, zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
SCR = Path(__file__).resolve().parent
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
def script(name): return str(next(SCR.parent.glob(f"*/{name}")))      # scripts live in src/<area>/
sys.argv, _argv = [sys.argv[0]], sys.argv
import confirm_anchor_fusion as CA
ROOT = CA.ROOT; L4 = ROOT / "workspace/conf4"; RAW = ROOT / "raw_data/journal/conf4"; PRED = L4 / "preds"
FR = ROOT / "raw_data/journal/frames"; PY = sys.executable; K0 = 20; BATCH_BYTES = 3.5e9


def videos():
    h = pd.read_csv(L4 / "hmdb51_selection.csv").assign(batch=0)
    u = pd.read_csv(L4 / "ucf_train_selection.csv").sort_values(["binary_label", "category", "video"]).reset_index(drop=True)
    u["cluster"] = "UCF/" + u.video.str.replace(r"\.mp4$", "", regex=True)
    b, acc, out = 1, 0.0, []
    for x in u.bytes:
        if acc + x > BATCH_BYTES and acc > 0: b, acc = b + 1, 0.0
        out.append(b); acc += x
    u["batch"] = out
    v = pd.concat([h, u], ignore_index=True)
    v["path"] = [str(RAW / ("hmdb" if r.benchmark == "HMDB51" else "ucf") / r.video) for r in v.itertuples()]
    return v


def kaggle_get(ref, fname, dest):
    if dest.exists() and dest.stat().st_size > 0: return
    c = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
    auth = base64.b64encode(f"{c['username']}:{c['key']}".encode()).decode()
    url = f"https://www.kaggle.com/api/v1/datasets/download/{ref}/{urllib.parse.quote(fname, safe='')}"
    dest.parent.mkdir(parents=True, exist_ok=True); tmp = dest.with_suffix(dest.suffix + ".part")
    for t in range(4):                                  # streamed to disk: some UCF normal videos are several GB
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": "Basic " + auth}), timeout=600) as r, open(tmp, "wb") as fh:
                shutil.copyfileobj(r, fh, 16 << 20)
            break
        except Exception:
            if t == 3: raise
            time.sleep(10 * (t + 1))
    with open(tmp, "rb") as fh: zipped = fh.read(2) == b"PK"
    if zipped:
        with zipfile.ZipFile(tmp) as z, z.open(z.namelist()[0]) as src, open(dest, "wb") as fh: shutil.copyfileobj(src, fh, 16 << 20)
        tmp.unlink()
    else: tmp.rename(dest)


def build_store(k, df):
    from video_data.build_external_windows import decode_video
    import video_data.frame_store as VF
    VF.STORE_ROOT = FR
    jobs = [(p, [(r.start_frame, r.end_frame) for r in g.itertuples()]) for p, g in df.groupby("path", sort=False)]
    w = VF.StoreWriter(f"c3_{k}"); failed = []
    with ProcessPoolExecutor(16) as ex:
        for (p, wins), frames in zip(jobs, ex.map(decode_video, jobs, chunksize=1)):
            for (s, e), fr in zip(wins, frames):
                if fr is None: failed.append((p, s)); continue
                w.add(VF.content_key(p, s, e, decode="seq"), fr)
    w.close(dict(note=f"Part G confirmation chunk {k}"))
    fk = set(failed); df = df.assign(decode_failed=[int((p, s) in fk) for p, s in zip(df.path, df.start_frame)])
    df.to_csv(CA.C3 / f"chunks/c3_{k}.csv", index=False); df.to_csv(L4 / f"windows_c{k}.csv", index=False)
    print(f"chunk {k}: {len(df)} windows, decode failures {len(failed)}", flush=True)
    return df[df.decode_failed == 0].reset_index(drop=True)


def score_heads(k):
    import torch, lensnet_setup as L, lensnet as V, vadclip_run as VR
    f = V.Compact(f"c3_{k}"); rows = np.arange(len(f.index))
    for s in (42, 43, 44):
        for cfg, tag in (("LENS-M", "LENS-M_vrex"), ("LP", "LP"), ("V2b", "V2b"), ("MP", "MP")):
            n = f"v2_audited_{tag}_seed{s}"; m = V.V2(cfg, L.text_anchors()); m.load_state_dict(torch.load(L.OUT / n / "best.pt", weights_only=True)); m.eval()
            pd.DataFrame({"p_violent": V.predict(m, f, rows)}).to_csv(PRED / f"{n}_c{k}.csv", index=False)
    zs = V.V2("LP", L.text_anchors()); zs.eval(); pd.DataFrame({"p_violent": V.predict(zs, f, rows)}).to_csv(PRED / f"ZS_c{k}.csv", index=False)
    dev = torch.device("cuda"); orig_load = VR.VM.clip.load
    VR.VM.clip.load = lambda nm, device: orig_load(VR.CLIPNAME["L14"], device, download_root=str(ROOT / "raw_data/clip_weights"))
    X = np.load(VR.FEATS["L14"] / f"c3_{k}/cls_frames.npy")
    for s in (42, 43, 44):
        d = ROOT / f"workspace/runs_vadclip/VadCLIP-L14_audited_lr0.0001_seed{s}"
        m = VR.VM.CLIPVAD(num_class=2, embed_dim=768, visual_length=16, visual_width=768, visual_head=1, visual_layers=1,
                          attn_window=4, prompt_prefix=10, prompt_postfix=10, device=dev).to(dev)
        r = m.load_state_dict(torch.load(d / "best.pt", weights_only=True), strict=False)
        assert not r.unexpected_keys and all(x.startswith("clipmodel.") for x in r.missing_keys)
        pd.DataFrame({"p_violent": VR.score(m, X, dev)}).to_csv(PRED / f"{d.name}_c{k}.csv", index=False)
        del m; torch.cuda.empty_cache()


def score_video_models(k, w, out_dir=PRED, which=("vmae", "clipft")):
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    import video_data.frame_store as VF
    from video_data import window_data as D
    dev = torch.device("cuda"); VF.STORE_ROOT = FR
    if "vmae" in which:
        import train_video_baseline as TB, train_finetuned_on_split as RTS
        store = VF.FrameStore(stores=(f"c3_{k}",), use_v1=False); ww = w.assign(binary_label=0)
        for rd in [ROOT / f"workspace/runs_baselines/A1_videomae_lr2e-05_seed{s}" for s in (42, 43, 44)] + \
                  [ROOT / f"workspace/runs_review/A1_videomae_lora_lr0.0001_seed{s}" for s in (42, 43, 44)]:
            out = out_dir / f"{rd.name}_c{k}.csv"
            if out.exists(): continue
            sm = json.load(open(rd / "summary.json")); m = TB.Net(sm["model"]).to(dev)
            if sm.get("lora"): RTS.add_lora(m); m.to(dev)
            m.load_state_dict(torch.load(rd / "best.pt", map_location=dev)); m.eval()
            dl = DataLoader(D.WindowDataset(ww, store, False, decode="seq"), batch_size=8, num_workers=8, pin_memory=True)
            pd.DataFrame({"p_violent": D.predict(m, dl, dev, tta=True)}).to_csv(out, index=False); del m; torch.cuda.empty_cache()
    if "clipft" in which:
        import clip_tail as C, lensnet_setup as L, lensnet as V2m, extract_clip_tokens as E
        PROMPTS = torch.tensor(np.load(L.FEAT / "text_prompts.npy"))
        df, st, dec = E.load_set(f"c3_{k}"); assert len(df) == len(w), (len(df), len(w))
        ds = E.Windows(df[["path", "start_frame", "end_frame"]], st, dec)
        vh = C.load_clip(dev, "fp16")
        for s in (42, 43, 44):
            out = out_dir / f"CLIP-FT_audited_seed{s}_c{k}.csv"
            if out.exists(): continue
            tail = C.Tail(C.load_clip(dev, "fp32")).to(dev); head = V2m.V2("LP", L.text_anchors()).to(dev)
            ck = torch.load(ROOT / f"workspace/runs_review/CLIP-FT_audited_seed{s}/best.pt", map_location=dev)
            tail.load_state_dict(ck["tail"]); head.load_state_dict(ck["head"]); tail.eval(); head.eval(); res = np.zeros(len(ds))
            with torch.no_grad():
                for x, ii in DataLoader(ds, batch_size=8, num_workers=12, pin_memory=True):
                    B = x.shape[0]
                    with torch.autocast("cuda", dtype=torch.float16):
                        t = tail(C.head(vh, x.to(dev, non_blocking=True).half().flatten(0, 1)).float()).view(B, 16, 17, 768)
                    z = F.normalize(t.float(), dim=-1)
                    res[ii.numpy()] = torch.softmax(head(z[:, :, 0].mean(1), z @ PROMPTS.to(dev).t()).float(), 1)[:, 1].cpu().numpy()
            pd.DataFrame({"p_violent": res}).to_csv(out, index=False); del tail, head; torch.cuda.empty_cache()
        del vh; torch.cuda.empty_cache()


def wait_disk(gb=8):
    while shutil.disk_usage(ROOT).free / 1e9 < gb:
        print(f"waiting for disk space ({shutil.disk_usage(ROOT).free / 1e9:.1f} GB free)", flush=True); time.sleep(300)


def batch(i):
    k = K0 + i; PRED.mkdir(parents=True, exist_ok=True)
    if (L4 / f"done_c{k}").exists(): print("batch", i, "already done"); return
    v = videos(); v = v[v.batch == i]; wait_disk(8)
    for r in v.itertuples(): kaggle_get(r.kaggle_ref, r.kaggle_file, Path(r.path))
    print(f"batch {i}: {len(v)} videos downloaded", flush=True)
    vids = [dict(benchmark=r.benchmark, video=r.video, cluster=r.cluster, path=r.path, category=r.category) for r in v.itertuples()]
    with ProcessPoolExecutor(16) as ex: rows = [x for pl in ex.map(CA.plan_one, vids) for x in pl]
    df = pd.DataFrame(rows).assign(chunk=k); w = build_store(k, df)
    for cmd in (["extract_compact.py", "--set", f"c3_{k}"], ["extract_vmae.py", f"c3_{k}"]):
        subprocess.run([PY, script(cmd[0]), *cmd[1:]], check=True, cwd=str(SCR))
    score_heads(k); score_video_models(k, w)
    subprocess.run([PY, script("confirm_anchor_fusion.py"), "clean", str(k)], check=True, cwd=str(SCR))
    for r in v.itertuples(): Path(r.path).unlink(missing_ok=True)
    (L4 / f"done_c{k}").touch(); print(f"batch {i} done (chunk {k}), store and raw videos removed", flush=True)


def clipft_c3(k):
    """CLIP-FT on a Part F chunk (missing there): rebuild its store with the original builder, score, remove."""
    out_dir = CA.C3 / ("preds_clipft" if k < 10 else "predsB_clipft"); out_dir.mkdir(exist_ok=True)
    if all((out_dir / f"CLIP-FT_audited_seed{s}_c{k}.csv").exists() for s in (42, 43, 44)): return
    ref = pd.read_csv(CA.C3 / f"chunks/c3_{k}.csv"); wait_disk(6)
    subprocess.run([PY, script("confirm_anchor_fusion.py" if k < 10 else "confirm_anchor_fusion_sample_b.py"), "build", str(k)], check=True, cwd=str(SCR))
    new = pd.read_csv(CA.C3 / f"chunks/c3_{k}.csv"); cols = ["path", "start_frame", "end_frame", "decode_failed"]
    assert len(new) == len(ref) and (new[cols].values == ref[cols].values).all(), f"chunk {k}: window list changed"
    score_video_models(k, new[new.decode_failed == 0].reset_index(drop=True), out_dir=out_dir, which=("clipft",))
    subprocess.run([PY, script("confirm_anchor_fusion.py"), "clean", str(k)], check=True, cwd=str(SCR)); print(f"CLIP-FT chunk {k} done", flush=True)


def dedup():
    import torch, torch.nn.functional as F, lensnet as V
    man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "split"])
    sp = pd.read_csv(ROOT / "workspace/splits/std_rwf.csv", usecols=["segment_id", "split"])
    train = set(man[man.split == "train"].segment_id) | set(sp[sp.split == "train"].segment_id)
    b = V.Compact("bench"); tb = F.normalize(b.cls[np.where(man.segment_id.isin(train).values)[0]], dim=-1)
    ks = sorted(int(p.name[6:]) for p in L4.glob("done_c*")); parts = []
    for k in ks:
        w = pd.read_csv(L4 / f"windows_c{k}.csv"); w = w[w.decode_failed == 0].reset_index(drop=True)
        tc = F.normalize(V.Compact(f"c3_{k}").cls, dim=-1); assert len(tc) == len(w)
        mx = torch.cat([(tc @ tb[i:i + 4096].t()).max(1).values.unsqueeze(1) for i in range(0, len(tb), 4096)], 1).max(1).values.numpy()
        parts.append(w.assign(chunk=k, max_cos_to_train=mx, excluded_near_duplicate=(mx >= 0.95).astype(int)))
    x = pd.concat(parts, ignore_index=True); x.to_csv(L4 / "near_duplicate_exclusion.csv", index=False)
    print("excluded windows:", x.groupby("benchmark").excluded_near_duplicate.sum().to_dict(), "of", x.groupby("benchmark").size().to_dict(), flush=True)


if __name__ == "__main__":
    cmd = _argv[1]
    if cmd == "batch": batch(int(_argv[2]))
    elif cmd == "all":
        for i in sorted(videos().batch.unique()): batch(int(i))
        for k in (0, 1, 2, 3, 4, 5, 10, 11, 12): clipft_c3(k)
        dedup(); print("ALL_DONE", flush=True)
    elif cmd == "dedup": dedup()
    elif cmd == "clipft_c3": clipft_c3(int(_argv[2]))
