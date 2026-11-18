# Data

No dataset, frame or feature is distributed with this repository. Obtain each dataset from its original source and
respect its licence (most allow research use only and forbid redistribution). The roles follow Table 2 of the paper.
The per-window predictions of every model are released separately (`reproduce/get_predictions.sh`).

## Source corpora (training, validation and audited test)
| Dataset | Source |
|---|---|
| RLVS (Real Life Violence Situations) | Soliman et al., ICICIS 2019 (Kaggle release) |
| RWF-2000 | Cheng et al., ICPR 2020 (request from the authors) |
| Hockey Fights | Bermejo Nievas et al., CAIP 2011 |
| Bus Violence | Ciampi et al., Sensors 2022 |
| VID | Mahi et al., Data in Brief 2024 |

## Unseen test sets (never used for training or model selection)
| Dataset | Role | Source |
|---|---|---|
| XD-Violence (test videos, temporal annotations) | development | Wu et al., ECCV 2020 — roc-ng.github.io/XD-Violence |
| UCF-Crime (test videos, temporal annotations) | development | Sultani et al., CVPR 2018 |
| SCFD (Surveillance Camera Fight Dataset) | fresh (motion stream) | Akti et al., IPTA 2019 — github.com/seymanurakti/fight-detection-surv-dataset |
| SCVD | fresh (motion stream) | Aremu et al., SAI 2024 — Kaggle `toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd` (version 3) |
| VioPeru | untouched | Huillcen Baca et al., Sensors 2024 — github.com/hhuillcen/VioPeru |
| AIRTLab | untouched | Bianculli et al., Data in Brief 2020 — github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos |
| CCTV-Fights | untouched | Perez et al., ICASSP 2019 — ROSE Lab, NTU (access request; `download_cctv_fights.py` downloads with your own login) |
| Movies | untouched | Bermejo Nievas et al., CAIP 2011 — Kaggle `naveenk903/movies-fight-detection-dataset` |
| UBI-Fights | untouched | Degardin and Proença, IJCB 2020 — socia-lab.di.ubi.pt/EventDetection (annotations: `fetch_ubi_annotations.py`) |

## Confirmation of the fusion rules (video-level labels, untouched)
| Dataset | Videos | Source and selection list |
|---|---|---|
| XD-Violence training sample A | 600 | Hugging Face mirror `jherng/xd-violence`; `data_tables/xd_train_aie/xd_train_selection.csv` |
| XD-Violence training sample B | 326 | same mirror, disjoint from sample A; `data_tables/xd_train_aie/xd_train_selection_B.csv` |
| AIE | 647 | da Silva and Pereira, J. Brazilian Computer Society 2024 — Kaggle `arnaldovitor/aie-dataset` |
| UCF-Crime training subset | 388 | Kaggle `bypktt/ucf-crimes` and `minhajuddinmeraj/anomalydetectiondatasetucf`; `data_tables/untouched_fusion/ucf_train_selection.csv` lists the 417 videos drawn, of which 29 lost every window to the near-duplicate filter |
| HMDB51 interaction classes | 765 | Kuehne et al., ICCV 2011 — Kaggle `kodurujagruthaaditya/hmdb51`; `data_tables/untouched_fusion/hmdb51_selection.csv` |

## Third-party code
- VadCLIP (Apache-2.0): clone github.com/nwpu-zxr/VadCLIP into `raw_data/journal/ext/VadCLIP` for `vadclip_run.py`.

## Expected locations (relative to the repository root)
| What | Where |
|---|---|
| Window manifests, same-source edges (included) | `data_tables/manifests/` |
| Frame stores (built by the `video_data` utilities and `build_scfd_scvd.py` / `build_vioperu_airtlab.py`) | `source_benchmark/frames/`, `raw_data/{fresh,journal}/frames/` |
| Extracted features (built by `extract_compact.py`, `extract_vmae.py`) | `workspace/features/{l14,b16}/<set>/` |
| Run outputs and checkpoints | `workspace/runs*/` |

## Window tables included here
- `data_tables/manifests/source_windows.csv`: every 1.5-s window of the source corpora, with its split and
  same-source component (the source-controlled split).
- `data_tables/manifests/near_duplicate_edges.csv`: the verified same-source links between windows.
- `data_tables/audit/`: inputs of the public-split audit (Section 4) and the fine-scored same-source edges.
  The human labels of the audit validation (Section 4.3) are in `results/revision/human_labels_*.csv`, with the
  unblinding key `label_key_HIDDEN.csv`.
- `data_tables/scfd_scvd/`, `vioperu_airtlab/`, `cctv_fights/`, `movies_ubi_fights/`, `xd_train_aie/`,
  `untouched_fusion/`, `framelevel_xd_ucf/`, `ambiguous_windows/`: windows of the unseen and untouched sets and their
  near-duplicate checks.

## Rebuilding frames and features from the videos
Checking the numbers of the paper needs only the released predictions (`reproduce/get_predictions.sh`). To extract
features again from the videos, note three points.

- **Paths.** The `path` columns of the tables in `data_tables/` hold placeholders for the authors' folders
  (`/path/to/LENS-Net/...`), and the copies of these tables in the predictions archive hold paths relative to the
  repository (`raw_data/...`). A frame store identifies a window by a hash of its path, start frame and end frame
  (`video_data/frame_store.py`), so a store and the table that indexes it must be built with the same paths. The build
  scripts write their own tables, so run them rather than editing the released ones.
- **Unseen and untouched sets.** `video_data/build_external_windows.py` (XD-Violence and UCF-Crime test windows), the
  `build_*.py` scripts of `src/datasets/`, `confirm_footage_gate.py`, `confirm_anchor_fusion.py`,
  `framelevel_xd_ucf.py` and `src/revision/confirm_fusion_rules_untouched.py` build the windows and frame stores of
  these sets from the downloads under `raw_data/`.
- **Source corpora.** The five source corpora were used through a merged collection with unified file names
  (`FINAL_DATASET`). The manifest identifies a video by its name in that collection (`video_id`, `original_filename`).
  A column with the file names of the original release exists for RWF-2000 only (`orig_name` in
  `data_tables/audit/rwf_official_split_videos.csv`). For 20,356 of the 24,501 source windows the frames come from a
  cache made during that earlier curation (column `v1_cache_row`). The curation scripts, their manifests and the
  cache are not part of this repository, so `video_data/build_source_manifest.py` and the feature extraction of the
  source windows cannot be re-run from the public downloads alone. `video_data/build_frame_store.py` builds the
  store of the other 4,145 windows, and `cache_one_clip` in `video_data/cnn_bigru_model.py` is the frame sampler and
  JPEG encoder used for both.
