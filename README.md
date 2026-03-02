<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<h1 align="center">PointWorld</h1>

<p align="center">
Training and Evaluation Pipeline for "PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation".
</p>

<p align="center">
  <a href="https://github.com/NVlabs/PointWorld"><img src="https://img.shields.io/badge/Code-NVlabs%2FPointWorld-0969da.svg" alt="Code Repository"></a>
  <a href="https://point-world.github.io/"><img src="https://img.shields.io/badge/Project-Website-2ea44f.svg" alt="Project Website"></a>
  <a href="https://arxiv.org/pdf/2601.03782"><img src="https://img.shields.io/badge/Paper-PDF-1f6feb.svg" alt="Paper PDF"></a>
  <a href="https://arxiv.org/abs/2601.03782"><img src="https://img.shields.io/badge/arXiv-2601.03782-b31b1b.svg" alt="arXiv"></a>
  <a href="https://youtu.be/XPOsCwrYdk0"><img src="https://img.shields.io/badge/Video-YouTube-red.svg" alt="Video"></a>
  <img src="https://img.shields.io/badge/Branch-main-blue.svg" alt="main branch">
</p>

<p align="center">
  <a href="https://wenlong.page">Wenlong Huang</a><sup>1,†</sup>,
  <a href="https://scholar.google.com/citations?user=48Y9F-YAAAAJ&hl=en">Yu-Wei Chao</a><sup>2</sup>,
  <a href="https://cs.gmu.edu/~amousavi/">Arsalan Mousavian</a><sup>2</sup>,
  <a href="https://mingyuliu.net/">Ming-Yu Liu</a><sup>2</sup>,
  <a href="https://homes.cs.washington.edu/~fox/">Dieter Fox</a><sup>2</sup>,
  <a href="https://kaichun-mo.github.io/">Kaichun Mo</a><sup>2,*</sup>,
  <a href="https://profiles.stanford.edu/fei-fei-li">Li Fei-Fei</a><sup>1,*</sup>
  <br/>
  <sup>1</sup>Stanford University, <sup>2</sup>NVIDIA
  <br/>
  <sup>*</sup>Equal advising &nbsp;|&nbsp; <sup>†</sup>Work done partly at NVIDIA
</p>

<p align="center">
  <a href="https://point-world.github.io/">
    <img src="https://point-world.github.io/media/preview-card-1080p.jpg" alt="PointWorld teaser" width="100%"/>
  </a>
</p>

<p align="center"><em>
PointWorld is a large pre-trained 3D world model that predicts full-scene 3D point flows from partially observable RGB-D captures and robot actions, also represented as 3D point flows.
</em></p>

If you find this work useful in your research, please cite using the following BibTeX:

```bibtex
@article{huang2026pointworld,
  title={PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation},
  author={Huang, Wenlong and Chao, Yu-Wei and Mousavian, Arsalan and Liu, Ming-Yu and Fox, Dieter and Mo, Kaichun and Li, Fei-Fei},
  journal={arXiv preprint arXiv:2601.03782},
  year={2026}
}
```

<a id="table-of-contents"></a>
## 🗂️ Table of Contents

- [Important Notes](#important-notes)
- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Known Limitations](#known-limitations)
- [Acknowledgements](#acknowledgements)
- [Contributing](#contributing)

<a id="important-notes"></a>
## 📌 Important Notes

- Precomputed datasets and pretrained checkpoints are still under internal review at NVIDIA and are expected to be released in the next 1-2 months.
- `main` is the training/evaluation code branch for release.
- `data` is the dataset preparation pipeline branch.
- Please first prepare the data using the `data` branch. Then return to `main` for training and evaluation.

<a id="setup"></a>
## 🛠️ Setup

### Environment

The `main` branch provides a self-contained conda setup with no local editable dependencies.
Recommended baseline for reproducibility in `main`:
- Linux `x86_64`
- Python `3.10`
- NVIDIA driver compatible with CUDA 12.4 wheels

Recommended setup:

```bash
# from repo root
conda env create -n pointworld-env -f environments/train_eval.yml
conda activate pointworld-env
# timm is used for PTv3 DropPath; install without pulling extra transitive deps
python -m pip install timm==1.0.19 --no-deps
# keep urdfpy-compatible graph deps on a Python 3.10-safe networkx release
python -m pip install networkx==3.4.2 --no-deps
```

If you also need visualization extras:

```bash
conda env update -n pointworld-env -f environments/train_eval_viz.yml --prune
# timm is used for PTv3 DropPath; install without pulling extra transitive deps
python -m pip install timm==1.0.19 --no-deps
# keep urdfpy-compatible graph deps on a Python 3.10-safe networkx release
python -m pip install networkx==3.4.2 --no-deps
```

Dependency layout:
- `environments/requirements.txt`: canonical base dependency list for train/eval.
- `environments/train_eval_viz.yml`: optional visualization extras (`matplotlib`, `open3d`, `viser`).

### Third-Party Dependency (DINOv3)

Request access via the [official DINOv3 release page](https://github.com/facebookresearch/dinov3) first, then use the provided download URL.

```bash
git submodule update --init --recursive
mkdir -p third_party/dinov3/checkpoints
wget -O third_party/dinov3/checkpoints/<dinov3_vitl16_pretrain_*.pth> \
  "<URL_FROM_DINOV3_ACCESS_EMAIL>"
```

### Dataset Path Convention

Use this directory layout for generated datasets consumed by `main`:
- DROID WDS: `/path/to/droid/wds`
- BEHAVIOR WDS: `/path/to/behavior/wds`

The `arguments.py` defaults now follow this convention under `LOCAL_DATASET_DIR`:
- `droid` -> `${LOCAL_DATASET_DIR}/droid/wds`
- `behavior` -> `${LOCAL_DATASET_DIR}/behavior/wds`

<a id="training"></a>
## 🏋️ Training

### PTv3 Architecture Variant

PointWorld release now supports three PTv3 variants:
- `small`
- `base` (default)
- `large`

Set the variant explicitly with `--ptv3_size=<small|base|large>` in training/evaluation commands when needed.

### Single-Domain Training (DROID)

```bash
python train.py \
  --domains=droid \
  --data_dirs=/path/to/droid/wds \
  --norm_stats_path=stats/droid \
  --batch_size=<BATCH_SIZE> \
  --num_workers=<NUM_WORKERS> \
  --eval_num_workers=<EVAL_NUM_WORKERS> \
  --eval_freq=-1
```

Replace `/path/to/droid/wds` and worker/batch settings with values that match your machine.

### Single-Domain Training (BEHAVIOR)

```bash
python train.py \
  --domains=behavior \
  --data_dirs=/path/to/behavior/wds \
  --norm_stats_path=stats/droid_behavior \
  --batch_size=<BATCH_SIZE> \
  --num_workers=<NUM_WORKERS> \
  --eval_num_workers=<EVAL_NUM_WORKERS> \
  --eval_freq=-1
```

### Multi-Domain Training (DROID + BEHAVIOR)

```bash
python train.py \
  --domains=droid,behavior \
  --data_dirs=/path/to/droid/wds,/path/to/behavior/wds \
  --norm_stats_path=stats/droid_behavior \
  --batch_size=<BATCH_SIZE> \
  --num_workers=<NUM_WORKERS> \
  --eval_num_workers=<EVAL_NUM_WORKERS> \
  --eval_freq=-1
```

### DDP Training Template

```bash
torchrun \
  --standalone \
  --nproc_per_node=<NUM_GPUS> \
  train.py \
  --distributed=true \
  <your_train_args>
```

<a id="evaluation"></a>
## 📊 Evaluation

By default, release evaluation targets the `test` split.

### Expert Model Training For DROID Filtered Metrics (Optional)

This step is only required if you want reliable filtered metrics on the DROID domain (`full_eval/test/filtered_l2_moved/mean`) and for reproducing the results in the paper.

```bash
python train.py \
  --domains=droid \
  --data_dirs=/path/to/droid/wds \
  --norm_stats_path=stats/droid \
  --train_splits=test \
  --exp_name=droid-test-expert \
  --batch_size=<BATCH_SIZE> \
  --num_workers=<NUM_WORKERS> \
  --eval_num_workers=<EVAL_NUM_WORKERS> \
  --eval_freq=-1
```

### 1. DROID Evaluation (Annotation-Aware)

The key paper metric is:

- `full_eval/test/filtered_l2_moved/mean`

To evaluate filtered metrics, generate expert confidence locally first.

1. Set the expert checkpoint path (for example, from the `--train_splits=test` run above):

```bash
EXPERT_MODEL_PATH=/path/to/train_logs/droid-test-expert/model-last.pt
```

2. Generate confidence annotations on DROID test split:

```bash
python eval.py \
  --model_path "${EXPERT_MODEL_PATH}" \
  --domains=droid \
  --data_dirs=/path/to/droid/wds \
  --run_confidence_annotation=true \
  --confidence_thres=0.8 \
  --batch_size=1 \
  --eval_num_batches=-1
```

This writes `expert_confidence-seed=42.h5` under `/path/to/droid/wds/test/`.

3. Evaluate a target checkpoint using the generated confidence annotation:

```bash
MODEL_PATH=/path/to/train_logs/<run_name>/model-last.pt
```

```bash
python eval.py \
  --model_path "${MODEL_PATH}" \
  --domains=droid \
  --data_dirs=/path/to/droid/wds \
  --confidence_thres=0.8 \
  --batch_size=1 \
  --eval_num_batches=-1
```

For quicker iteration, you can set `--eval_num_batches=<N>` (for example `100`) instead of full-dataset evaluation.

### 2. BEHAVIOR Evaluation (Simulation-Only)

BEHAVIOR evaluation does not require the expert-confidence annotation because the data is noiseless.

```bash
MODEL_PATH=/path/to/train_logs/<run_name>/model-last.pt
```

```bash
python eval.py \
  --model_path "${MODEL_PATH}" \
  --domains=behavior \
  --data_dirs=/path/to/behavior/wds \
  --norm_stats_path=stats/droid_behavior \
  --batch_size=1 \
  --eval_num_batches=-1
```

<a id="visualization"></a>
## 🎞️ Visualization

PointWorld visualization is built on top of [`viser`](https://github.com/nerfstudio-project/viser), which provides the live 3D viewer and GUI controls.

Use evaluation-time visualization by setting `--eval_viz_num > 0`:

```bash
python eval.py \
  --model_path "${MODEL_PATH}" \
  --domains=droid \
  --data_dirs=/path/to/droid/wds \
  --batch_size=1 \
  --eval_num_batches=100 \
  --eval_viz_num=8 \
  --viewer_port=8080
```

When running, open `http://localhost:8080` in your browser.

Visualization includes these controls:
- `Frame`: step through temporal evolution (frame-by-frame) across the sequence.
- `Ground-truth`: switch between model prediction and GT trajectories.
- `Upsample`: toggle between coarse and upsampled point rendering.
- `Scene flow density` and `Robot flow density`: reduce/increase the number of rendered flow vectors.
- `Scene Flow Thickness` and `Robot Flow Thickness`: adjust vector thickness for readability.
- `Point size`: adjust rendered point cloud size.
- `Full overlay opacity`: control overlay transparency.

Runtime behavior:
- After each visualized sample, the CLI prompts `Press ENTER to continue ...` (type `q` to stop).
- This prompt requires an interactive TTY (a real terminal stdin). If stdin is redirected/captured, the prompt may fail.
- In headless setups, SSH with a terminal attached and forward the viewer port if needed.

If you want to run evaluation without visualization, set `--eval_skip_viz=true` (or leave `--eval_viz_num=-1`).

<a id="known-limitations"></a>
## ⚠️ Known Limitations

- Eval outputs are not deterministic on GPU; small run-to-run variation is expected even with fixed seeds.
- Partial-batch comparisons (`eval_num_batches < full dataset`) are sensitive to `num_workers` and `eval_num_workers`; match these settings when comparing runs.



<a id="acknowledgements"></a>
## 🙏 Acknowledgements

We gratefully acknowledge the authors and maintainers of third-party projects that this repository depends on or adapts.
Modifications have been made where noted, and the original license terms remain in effect.

Third-party OSS attribution and license references for distributed or adapted code are documented in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

| Repository / Project | Usage in this repo | License |
|---|---|---|
| [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | Scene encoder backbone submodule (`third_party/dinov3/`) | [DINOv3 License](third_party/dinov3/LICENSE.md) |
| [Pointcept/PointTransformerV3](https://github.com/Pointcept/PointTransformerV3) | Vendored/adapted PTv3 components (`ptv3/`) | [MIT](ptv3/LICENSE) |
| [facebookresearch/sonata](https://github.com/facebookresearch/sonata) | PTv3 lineage reference for adapted components | [Apache-2.0](https://github.com/facebookresearch/sonata/blob/main/LICENSE) |
| [StanfordVL/OmniGibson](https://github.com/StanfordVL/OmniGibson) | Adapted transform utilities (`transform_utils.py`, `deploy/transform_utils_torch.py`) | [MIT](https://github.com/StanfordVL/OmniGibson/blob/main/LICENSE) |
| [UT-Austin-RPL/deoxys_control](https://github.com/UT-Austin-RPL/deoxys_control) | Additional adapted transform routines noted in `transform_utils.py` | [Apache-2.0](https://github.com/UT-Austin-RPL/deoxys_control/blob/main/LICENSE) |

<a id="contributing"></a>
## 🤝 Contributing

All external contributions must follow `CONTRIBUTING.md` in this repository.
In particular, commits must be signed off (`git commit -s`) to satisfy DCO requirements.
