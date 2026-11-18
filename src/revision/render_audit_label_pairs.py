"""D1: one JPEG per pair (A: three frames on top, B: three frames below) for the labelling page, from the same pairs and
frames as labelling_sheets.pdf (key: review/label_key_HIDDEN.csv). Only pair ids P001..P{N} are rendered."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
os.environ["LENS_BACKBONE"] = "L14"
import sys as _s; from pathlib import Path as _P; _R = _P(__file__).resolve().parents[2]; _s.path[:0] = [str(_R / "src" / d) for d in ("revision", "model", "features", "datasets", "baselines", "evaluation")] + [str(_R), str(_R / "paper")]
import extract_clip_tokens as X
import video_data.frame_store as VF
from PIL import Image, ImageDraw, ImageFont
ROOT = __import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]); OUT = Path(sys.argv[1]); N = int(sys.argv[2])
key = pd.read_csv(ROOT / "workspace/review/label_key_HIDDEN.csv").head(N)
man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv", usecols=["segment_id", "path", "start_frame", "end_frame", "v1_cache_row"]).set_index("segment_id")
ORIG = VF.STORE_ROOT; bench = None; cache = {}
def frames3(src):
    global bench
    name, k = src.split(":", 1)
    if name == "bench":
        VF.STORE_ROOT = ORIG
        if bench is None: bench = X.FrameStore(stores=("supp",), use_v1=True)
        r = man.loc[k]; jp = bench.jpegs(int(r.v1_cache_row) if not pd.isna(r.v1_cache_row) else -1, VF.content_key(r.path, r.start_frame, r.end_frame, "seek"))
    else:
        if name not in cache: VF.STORE_ROOT = ORIG; cache[name] = X.load_set(name)
        df, st, dec = cache[name]; r = df.iloc[int(k)]; jp = st.jpegs(-1, VF.content_key(r.path, r.start_frame, r.end_frame, dec))
    out = []
    for t in (0, len(jp) // 2, len(jp) - 1):
        b = jp[t]; a = np.frombuffer(b, np.uint8) if isinstance(b, (bytes, bytearray)) else np.asarray(b)
        out.append(Image.fromarray(cv2.cvtColor(cv2.imdecode(a, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)).resize((400, 300)))
    return out
try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
except Exception: font = ImageFont.load_default()
for r in key.itertuples():
    if (OUT / f"{r.pair_id}.jpg").exists(): continue        # keep images already labelled from
    img = Image.new("RGB", (3 * 400 + 4 * 12 + 60, 2 * 300 + 3 * 12), (24, 24, 28)); d = ImageDraw.Draw(img)
    for row, src in enumerate((r.a, r.b)):
        y = 12 + row * 312; d.text((18, y + 130), "AB"[row], fill=(240, 240, 240), font=font)
        for c, im in enumerate(frames3(src)): img.paste(im, (60 + c * 412, y))
    img.save(OUT / f"{r.pair_id}.jpg", quality=82)
print("rendered", len(key))
