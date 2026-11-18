# Audit of the public evaluation splits

`audit_public_splits.py` computes the same-source overlap under the public evaluation splits (Section 4, Fig. 3 and
Table 8c of the paper) and writes `results/public_split_audit.json`.

`PROTOCOL_public_split_audit.md` is the protocol as written on 23 September 2026, before any count was computed. It is
kept unchanged and therefore uses the folder names of the authors' working project:

| In the protocol | In this repository |
|---|---|
| `dataset_v2/manifests/segments_v2_binary.csv` | `data_tables/manifests/source_windows.csv` (the released file has placeholder video paths, so its md5 is not the one quoted) |
| `dataset_v2/manifests/near_duplicate_edges.csv` | `data_tables/manifests/near_duplicate_edges.csv` |
| `inputs/rwf2000_kaggle_files.json` | `data_tables/audit/rwf2000_kaggle_files.json` |
| `results/rwf_official_split_videos.csv` | `data_tables/audit/rwf_official_split_videos.csv` |
| `results/rwf_prefix_groups.csv` | not included; the prefix counts are under `B_prefix` and `B_agreement` in `results/public_split_audit.json` |
| `logs/protocol_md5.txt` and the name-chain manifests | not included |
