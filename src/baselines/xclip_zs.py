#!/usr/bin/env python3
"""PREREG_journal 6C(i): X-CLIP B/32 zero-shot. Texts = the 10 original LENS-Net prompts in the 2 original templates.
Window score = sigmoid(mean logit of violent texts - mean logit of non-violent texts); 8 of the 16 evaluation frames,
CLIP normalisation. Writes predictions for the audited test, external pool, fresh and conf sets."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("model", "features", "datasets", "baselines", "evaluation")]
import lensnet_setup as L, extract_clip_tokens as E
ROOT = L.ROOT; sys.path.insert(0, str(ROOT))
from video_data import window_data as D
import video_data.frame_store as VF
from transformers import XCLIPModel, AutoTokenizer
dev = torch.device("cuda")
m = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32").to(dev).half().eval()
tk = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
texts = [t.format(p) for c in (0, 1) for p in L.PROMPT_SETS["orig"][c] for t in L.TEMPLATES]
isv = torch.tensor([c == 1 for c in (0, 1) for p in L.PROMPT_SETS["orig"][c] for t in L.TEMPLATES], device=dev)
enc = tk(texts, padding=True, return_tensors="pt").to(dev)
im_m = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 1, 3, 1, 1); im_s = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 1, 3, 1, 1)
c_m = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 1, 3, 1, 1); c_s = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 1, 3, 1, 1)
def run(df, store, dec):
    out = np.zeros(len(df))
    with torch.no_grad():
        for x, _, i in DataLoader(D.WindowDataset(df.assign(binary_label=0), store, False, decode=dec), batch_size=8, num_workers=8):
            x = ((x.to(dev)[:, ::2] * im_s + im_m) - c_m) / c_s
            lg = m(input_ids=enc.input_ids, attention_mask=enc.attention_mask, pixel_values=x.half(), return_dict=True).logits_per_video.float()
            out[i.numpy()] = torch.sigmoid(lg[:, isv].mean(1) - lg[:, ~isv].mean(1)).cpu().numpy()
    return out
od = ROOT / "workspace/runs_baselines/XCLIP-ZS"; od.mkdir(parents=True, exist_ok=True)
man = pd.read_csv(ROOT / "data_tables/manifests/source_windows.csv"); te = man[man.split == "test"].reset_index(drop=True)
te.assign(p_violent=run(te, VF.FrameStore(stores=("supp",), use_v1=True), "seek")).to_csv(od / "predictions_test.csv", index=False)
ext = pd.read_csv(ROOT / "data_tables/manifests/external_windows.csv")
pool = ext[(ext.binary_label.isin([0, 1])) & (ext.excluded_contamination == 0) & (ext.decode_failed == 0)].reset_index(drop=True)
pool.assign(p_violent=run(pool, VF.FrameStore(stores=("external",), use_v1=False), "seq")).to_csv(od / "predictions_external.csv", index=False)
for s, pdir in (("fresh", "fresh/preds"), ("conf", "conf/preds")):
    df, store, dec = E.load_set(s)
    pd.DataFrame({"p_violent": run(df, store, dec)}).to_csv(ROOT / "workspace" / pdir / "XCLIP-ZS.csv", index=False)
print("X-CLIP zero-shot done", flush=True)
