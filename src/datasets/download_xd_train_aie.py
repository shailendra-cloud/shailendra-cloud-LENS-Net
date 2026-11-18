#!/usr/bin/env python3
"""Download the anchor-fusion confirmation data: the fixed XD-Violence training subset (Hugging Face mirror
jherng/xd-violence, files listed in conf3/xd_train_selection.csv) and the AIE dataset (Kaggle arnaldovitor/aie-dataset)."""
import os
import pandas as pd
from huggingface_hub import hf_hub_download
D = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/conf3"); R = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace")
sel = pd.read_csv(R / "conf3/xd_train_selection.csv")
for i, p in enumerate(sel.path):
    hf_hub_download("jherng/xd-violence", p, repo_type="dataset", local_dir=D / "xd")
    if i % 50 == 0: print("xd", i, flush=True)
print("XD DONE", flush=True)
os.environ["KAGGLEHUB_CACHE"] = str(D / "kaggle")
import kagglehub
print("AIE", kagglehub.dataset_download("arnaldovitor/aie-dataset"), flush=True)
print("ALL DONE", flush=True)
