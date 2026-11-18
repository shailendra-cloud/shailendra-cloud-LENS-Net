"""C6: download the second XD-Violence training sample (conf3/xd_train_selection_B.csv) from the same mirror."""
import pandas as pd
from huggingface_hub import hf_hub_download
D = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "raw_data/journal/conf3"); R = (__import__("pathlib").Path(__import__("os").environ.get("LENSNET_ROOT") or __import__("pathlib").Path(__file__).resolve().parents[2]) / "workspace")
sel = pd.read_csv(R / "conf3/xd_train_selection_B.csv")
for i, p in enumerate(sel.path):
    hf_hub_download("jherng/xd-violence", p, repo_type="dataset", local_dir=D / "xd")
    if i % 50 == 0: print("xd-B", i, flush=True)
print("XD-B DONE", flush=True)
