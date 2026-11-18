"""D1 (PREREG_revision.md): stratified sample of 500 window pairs for human validation of the two overlap criteria,
rendered as blind labelling sheets (PDF) + a CSV to fill in. The stratum/score key is written separately."""
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch
os.environ["LENS_BACKBONE"] = "L14"
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import extract_clip_tokens as X, lensnet_setup as L
import video_data.frame_store as VF
from PIL import Image, ImageDraw, ImageFont
ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]); OUTD = ROOT / "workspace/labelling"; KEY = ROOT / "workspace/review"
rng = np.random.default_rng(2026)
man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "video_id", "dataset", "split", "path", "start_frame", "end_frame", "v1_cache_row"])
fine = pd.read_csv(ROOT / "data_tables/audit/near_duplicate_edges_fine.csv")
ver, rej = fine[fine.grad128 >= 0.30], fine[fine.grad128 < 0.30]
pairs = []
def take(df, n, stratum):
    d = df.sample(n, random_state=int(rng.integers(1e9)))
    pairs.extend(dict(stratum=stratum, a=("bench", x.segment_a), b=("bench", x.segment_b), score=float(x.grad128)) for x in d.itertuples())
take(ver, 150, "verified_edge"); take(rej, 100, "rejected_candidate")
vid = man.video_id.values; seg = man.segment_id.values; k = 0
while k < 50:
    i, j = rng.integers(0, len(man), 2)
    if vid[i] != vid[j]: pairs.append(dict(stratum="random_pair", a=("bench", seg[i]), b=("bench", seg[j]), score=float("nan"))); k += 1
# unseen windows vs their nearest training window (mean CLIP embedding, as in the 0.95 rule)
zb = np.load(L.FEAT / "bench" / L.compact_file())["cls"]; trmask = (man.split == "train").values
B = torch.nn.functional.normalize(torch.tensor(zb[trmask]), dim=-1); trseg = seg[trmask]
cand_hi, cand_mid = [], []
for name, wf in (("fresh", "fresh/fresh_windows.csv"), ("cfights", "cfights/cfights_windows.csv"), ("conf2", "conf2/conf2_windows.csv")):
    w = pd.read_csv(ROOT / "workspace" / wf); w = w[w.decode_failed == 0].reset_index(drop=True)
    z = torch.nn.functional.normalize(torch.tensor(np.load(L.FEAT / name / L.compact_file())["cls"]), dim=-1)
    cs, ix = (z @ B.t()).max(1)
    for r in np.where(cs.numpy() >= 0.90)[0]:
        (cand_hi if cs[r] >= 0.95 else cand_mid).append(dict(a=(name, int(r)), b=("bench", trseg[int(ix[r])]), score=float(cs[r])))
for lst, n, st in ((cand_hi, 150, "cosine_removed"), (cand_mid, 50, "cosine_0.90-0.95")):
    sel = rng.choice(len(lst), min(n, len(lst)), replace=False); pairs.extend(dict(stratum=st, **lst[i]) for i in sel)
order = rng.permutation(len(pairs)); pairs = [pairs[i] for i in order]
for i, p in enumerate(pairs): p["pair_id"] = f"P{i + 1:03d}"
# ---- frames
stores = {}
def frames3(src):
    name, key = src
    if name == "bench":
        r = man[man.segment_id == key].iloc[0]
        if "bench" not in stores: stores["bench"] = X.FrameStore(stores=("supp",), use_v1=True)
        st = stores["bench"]; jp = st.jpegs(int(r.v1_cache_row) if not pd.isna(r.v1_cache_row) else -1, VF.content_key(r.path, r.start_frame, r.end_frame, "seek"))
    else:
        df, st, dec = X.load_set(name); r = df.iloc[key]; jp = st.jpegs(-1, VF.content_key(r.path, r.start_frame, r.end_frame, dec))
    out = []
    for t in (0, len(jp) // 2, len(jp) - 1):
        b = jp[t]; a = np.frombuffer(b, np.uint8) if isinstance(b, (bytes, bytearray)) else np.asarray(b)
        out.append(Image.fromarray(cv2.cvtColor(cv2.imdecode(a, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)).resize((240, 180)))
    return out
font = ImageFont.load_default()
pages, page = [], None
for i, p in enumerate(pairs):
    if i % 4 == 0:
        page = Image.new("RGB", (1240, 1754), "white"); ImageDraw.Draw(page).text((40, 20), "Same-source labelling: for each pair write 1 = same clip, 2 = same camera or scene, 3 = different source", fill="black", font=font); pages.append(page)
    y0 = 60 + (i % 4) * 420; d = ImageDraw.Draw(page); d.text((40, y0), p["pair_id"], fill="black", font=font)
    for row, src in enumerate((p["a"], p["b"])):
        for c, im in enumerate(frames3(src)): page.paste(im, (40 + c * 250, y0 + 20 + row * 190))
    d.text((800, y0 + 120), "A (top) vs B (bottom)", fill="black", font=font)
pages[0].save(OUTD / "labelling_sheets.pdf", save_all=True, append_images=pages[1:], resolution=150)
pd.DataFrame({"pair_id": [p["pair_id"] for p in pairs], "label_1same_2scene_3different": ""}).to_csv(OUTD / "labels_to_fill.csv", index=False)
pd.DataFrame([dict(pair_id=p["pair_id"], stratum=p["stratum"], a=f"{p['a'][0]}:{p['a'][1]}", b=f"{p['b'][0]}:{p['b'][1]}", score=p["score"]) for p in pairs]).to_csv(KEY / "label_key_HIDDEN.csv", index=False)
print("pairs", len(pairs), pd.Series([p["stratum"] for p in pairs]).value_counts().to_dict(), "pages", len(pages))
