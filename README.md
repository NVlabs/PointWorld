<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

<h1 align="center">PointWorld Data Branch</h1>

<p align="center">
Data annotation and generation pipeline for PointWorld.
</p>

<p align="center">
  <a href="https://github.com/NVlabs/PointWorld"><img src="https://img.shields.io/badge/Code-NVlabs%2FPointWorld-0969da.svg" alt="Code Repository"></a>
  <a href="https://point-world.github.io/"><img src="https://img.shields.io/badge/Project-Website-2ea44f.svg" alt="Project Website"></a>
  <a href="https://arxiv.org/pdf/2601.03782"><img src="https://img.shields.io/badge/Paper-PDF-1f6feb.svg" alt="Paper PDF"></a>
  <a href="https://arxiv.org/abs/2601.03782"><img src="https://img.shields.io/badge/arXiv-2601.03782-b31b1b.svg" alt="arXiv"></a>
  <a href="https://youtu.be/XPOsCwrYdk0"><img src="https://img.shields.io/badge/Video-YouTube-red.svg" alt="Video"></a>
  <img src="https://img.shields.io/badge/Branch-data-orange.svg" alt="data branch">
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
- [Quick Walkthrough](#quick-walkthrough)
- [Setup](#setup)
- [Download Third-Party Checkpoints](#third-party-checkpoints)
- [Full Data Generation Pipelines](#full-data-generation-pipelines)
- [Build Train/Eval Datasets from Generated H5](#build-train-eval-datasets-from-generated-h5)
- [Visualize Generated H5 Clips](#visualize-generated-h5-clips)
- [Acknowledgements](#acknowledgements)
- [Contributing](#contributing)


<a id="important-notes"></a>
## 📌 Important Notes

- Precomputed datasets and pretrained checkpoints are still under internal review at NVIDIA and are expected to be released in the next 1-2 months.
- `data` is the dataset preparation pipeline (this branch), and `main` is training/evaluation code.
- Please first prepare the data using the `data` branch. Then return to `main` for training and evaluation.

<a id="quick-walkthrough"></a>
## 🧭 Quick Walkthrough

1. Set up the environment in [Setup](#setup).
2. Download checkpoints for the required third-party annotation models in [Third-Party Checkpoints](#third-party-checkpoints).
3. Run generation pipelines in [Full Data Generation Pipelines](#full-data-generation-pipelines).
4. Run integrity, create split manifest, and convert to WDS in [Build Train/Eval Datasets from Generated H5](#build-train-eval-datasets-from-generated-h5).
5. Train/evaluate in the `main` branch with generated train/eval datasets.

```text
[DROID raw scenes + BEHAVIOR raw episodes]
                  |
                  v
    [Run full generation pipeline]
                  |
                  v
      [Generated H5 datasets]
                  |
                  v
      [data_integrity_check.py]
                  |
                  v
        [make_wds_manifest.py]
                  |
                  v
          [convert_wds.py]
                  |
                  v
       [WDS train/test shards]
                  |
                  v
[`main` branch training / evaluation input]
```

<a id="setup"></a>
## 🛠️ Setup


### Prerequisites and Storage Planning

- Install `gsutil` (Google Cloud CLI / Cloud Storage tools): https://cloud.google.com/storage/docs/gsutil_install
- Storage Planning:
  - The full pipeline is multi-terabyte scale.
  - Plan additional free space for temporary files and retries.
  - If rerunning full generation with intermediates, required storage can exceed `10 TB`.

### Streaming Cache

- For faster and more efficient processing, our data pipeline uses on-demand streaming for DROID (`gs://...`) and BEHAVIOR (`hf://...` list entries).
- Please set a shared cache root. With this set, files are re-used across different stages:
```bash
export POINTWORLD_CACHE_DIR=/path/to/local/cache/dir
```

### Python environment

```bash
conda create -n pointworld-data python=3.10
conda activate pointworld-data
git submodule update --init --recursive
python -m pip install -r requirements.txt
# urdfpy is required for DROID generation, but its metadata pins networkx==2.2
# (incompatible with python>=3.10 and scikit-image>=0.24). Install it in a staged way.
python -m pip install --no-deps urdfpy==0.0.22
python -m pip install -r environments/requirements_urdfpy_runtime.txt
pip install -e third_party/co-tracker
pip install -e third_party/vggt
```

### ZED Python API

Follow the official Stereolabs instructions: https://github.com/stereolabs/zed-python-api. After installation, verify:

```bash
python -c "import pyzed.sl as sl; print(sl.Camera.get_sdk_version())"
```

### Docker setup (for BEHAVIOR pipeline)

Build the BEHAVIOR runtime image:

```bash
docker build -f docker/dockerfile_behavior -t pointworld-behavior:v3.7.2 .
```

Initialize OmniGibson datasets and assets once (plan for about `40 GB` under `OG_DATA_ROOT`):

```bash
# Required: set OG_DATA_ROOT to a host path for OmniGibson assets (using a dedicated directory is recommended).
export OG_DATA_ROOT=/path/to/your/og_data
mkdir -p "$OG_DATA_ROOT"

docker run --rm --gpus all --ipc=host \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e OMNIGIBSON_DATA_PATH=/data \
  -v "$OG_DATA_ROOT":/data \
  pointworld-behavior:v3.7.2 \
  bash -lc 'cd /BEHAVIOR-1K && ./setup.sh --bddl --omnigibson --dataset --accept-nvidia-eula --accept-dataset-tos --confirm-no-conda'
```

With the pinned image in this repository, this initializes:
- `$OG_DATA_ROOT/behavior-1k-assets`
- `$OG_DATA_ROOT/omnigibson-robot-assets`
- `$OG_DATA_ROOT/2025-challenge-task-instances`
- `$OG_DATA_ROOT/omnigibson.key`

This path is the official BEHAVIOR-1K v3.7.2 setup flow, so Docker and source installs follow the same install entrypoint.
The image applies a minimal patch to `setup.sh` so container-provided Isaac env vars are accepted (required for `stanfordvl/omnigibson` base images).

### Optional: Enroot conversion from Docker image

If you run on Slurm/cluster environments that prefer Enroot, convert the Docker image:

```bash
enroot import -o pointworld-behavior.sqsh dockerd://pointworld-behavior:v3.7.2
```


<a id="third-party-checkpoints"></a>
## 📥 Download Third-Party Checkpoints

Download required third-party vision model checkpoints:

```bash
mkdir -p checkpoints/foundationstereo checkpoints/cotracker checkpoints/vggt

# CoTracker3
wget -O checkpoints/cotracker/scaled_online.pth \
  https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth

# VGGT-1B
wget -O checkpoints/vggt/model.pt \
  https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt

# FoundationStereo
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing" \
  -O checkpoints/foundationstereo
```

<a id="full-data-generation-pipelines"></a>
## 🧪 Full Data Generation Pipelines

<a id="distributed-execution-for-full-data-generation-optional"></a>
### Distributed Execution for Full Data Generation (Optional)

The full-generation workload is multi-terabyte scale. For practical end-to-end runtime, use distributed execution (for example, Slurm multi-job or multi-GPU/multi-node setups) when compute is available.

Built-in parallel flags:
- `real/compute_depth.py`: `--rank`, `--world_size`
- `real/compute_extrinsics.py`: `--rank`, `--world_size`
- `real/compute_2d_flows.py`: `--rank`, `--world_size`
- `real/convert_2d_flows_to_3d.py`: `--rank`, `--world_size`
- `simulation/behavior_3d_flows.py`: distributed list mode via `--input_list`, `--output_root`, `--rank`, `--world_size`

Alternatively, you can choose to process a subset of the data by creating a smaller list of source files to process for both DROID and BEHAVIOR. Specifically, you may create a copy of `real/droid_paths.txt` or `simulation/behavior_paths.txt`, edit them to include only the episodes you want to process, and provide them as arguments to the corresponding scripts.

### DROID

```bash
export DROID_ROOT=/path/to/processed/droid/outputs

# Optional: regenerate `real/gripper2wrist_transforms.json`
# python real/compute_gripper2wrist.py \
#   --scenes_file real/droid_paths.txt \
#   --output_dir real \
#   --world_size 1 \
#   --rank 0

# Depth
python real/compute_depth.py \
  --input real/droid_paths.txt \
  --output_dir "$DROID_ROOT" \
  --foundation_stereo_ckpt checkpoints/foundationstereo/23-51-11/model_best_bp2.pth \
  --foundation_stereo_cfg assets/foundationstereo/23-51-11/cfg.yaml

# Extrinsics
python real/compute_extrinsics.py \
  --input real/droid_paths.txt \
  --output_dir "$DROID_ROOT" \
  --vggt_model_path checkpoints/vggt/model.pt

# Optional but recommended: filter scenes by extrinsics quality before 2D tracking.
# This avoids spending extra GPU time tracking scenes that will be discarded.
python real/filter_paths_by_extrinsics_quality.py \
  --input real/droid_paths.txt \
  --output "$DROID_ROOT/droid_paths_final_loss_lt_0.10.txt" \
  --output_dir "$DROID_ROOT" \
  --max_final_loss 0.10

# Reuse one canonical scene list for both 2D and 3D stages.
# If you skipped the optional filter step above, use `real/droid_paths.txt`.
export DROID_FLOW_INPUT="$DROID_ROOT/droid_paths_final_loss_lt_0.10.txt"
# export DROID_FLOW_INPUT="real/droid_paths.txt"

# 2D tracks
python real/compute_2d_flows.py \
  --input "$DROID_FLOW_INPUT" \
  --output_dir "$DROID_ROOT" \
  --cotracker_ckpt checkpoints/cotracker/scaled_online.pth

# 3D flows
python real/convert_2d_flows_to_3d.py \
  --input "$DROID_FLOW_INPUT" \
  --output_dir "$DROID_ROOT" \
  --extrinsics_source optimized
```

When `convert_2d_flows_to_3d.py` is given `--input`, it validates required per-scene artifacts (`2d_flows`, `depth`, `cameras`) up front and fails fast on any missing/incomplete dependency.

### BEHAVIOR

Replace the paths below with your own local paths:
- `POINTWORLD_REPO`: local checkout of this repository
- `OG_DATA_ROOT`: OmniGibson assets root initialized above
- `POINTWORLD_CACHE_DIR`: shared cache root for DROID + BEHAVIOR streaming
- `BEHAVIOR_ROOT`: root for BEHAVIOR generated outputs

```bash
export POINTWORLD_REPO=/path/to/point-world-repo
export OG_DATA_ROOT=/path/to/your/og_data
export POINTWORLD_CACHE_DIR=/path/to/local/cache/dir
export BEHAVIOR_ROOT=/path/to/processed/behavior/outputs
mkdir -p "$POINTWORLD_CACHE_DIR" "$BEHAVIOR_ROOT/flows"

docker run --rm --gpus all --ipc=host --ulimit core=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e OMNIGIBSON_HEADLESS=1 \
  -e OMNIGIBSON_DATA_PATH=/data \
  -e OMNIGIBSON_DATASET_PATH=/data/behavior-1k-assets \
  -e OMNIGIBSON_ASSET_PATH=/data/omnigibson-robot-assets \
  -e OMNIGIBSON_KEY_PATH=/data/omnigibson.key \
  -e POINTWORLD_CACHE_DIR=/cache/pointworld \
  -v "$POINTWORLD_REPO":/workspace/point-world \
  -v "$OG_DATA_ROOT":/data \
  -v "$POINTWORLD_CACHE_DIR":/cache/pointworld \
  -v "$BEHAVIOR_ROOT":/workspace/behavior \
  pointworld-behavior:v3.7.2 \
  bash -lc "cd /workspace/point-world && \
  python simulation/behavior_3d_flows.py \
    --headless \
    --input_list simulation/behavior_paths.txt \
    --output_root /workspace/behavior/flows \
    --rank 0 \
    --world_size 1"
```

On the first run, OmniGibson/Isaac extension sync, shader compilation, and scene material setup may take several minutes before frame processing begins.

<a id="build-train-eval-datasets-from-generated-h5"></a>
## 📦 Build Train/Eval Datasets from Generated H5

<a id="distributed-execution-for-dataset-build-optional"></a>
### Distributed Execution for Dataset Build (Optional)

- `convert_wds.py` supports distributed execution via `--rank` and `--world_size`.
- `convert_wds.py` requires an explicit `--manifest` path.
- Split behavior in `convert_wds.py`:
  - split membership is selected globally first from the provided manifest;
  - `rank/world_size` only partition work inside the already-selected train/test sets.
- `data_integrity_check.py` is intended to run on a single machine in one launch.
  - It supports local multiprocessing with `--num_mp_workers`.
  - It is not designed as a cross-machine distributed job.

### DROID

```bash
python data_integrity_check.py \
  --input_dir /path/to/droid/flows-fs-optimize \
  --domain droid

python make_wds_manifest.py \
  --input_dir /path/to/droid/flows-fs-optimize \
  --domain droid \
  --output_manifest /path/to/droid/flows-fs-optimize/wds_manifest_seed42_test0.1.json

python convert_wds.py \
  --input_dir /path/to/droid/flows-fs-optimize \
  --output_dir /path/to/droid/wds \
  --domain droid \
  --manifest /path/to/droid/flows-fs-optimize/wds_manifest_seed42_test0.1.json
```

To match the test split from the paper, pass the release manifest directly instead of generating one:

```bash
python convert_wds.py \
  --input_dir /path/to/droid/flows-fs-optimize \
  --output_dir /path/to/droid/wds \
  --domain droid \
  --manifest manifests/droid_paper_split_manifest.json
```

Manifest matching is strict and fail-fast: if local integrity clips do not match the manifest universe, regenerate a local manifest from your local `integrity_check.json`.

### BEHAVIOR

```bash
python data_integrity_check.py \
  --input_dir /path/to/behavior/flows \
  --domain behavior

python make_wds_manifest.py \
  --input_dir /path/to/behavior/flows \
  --domain behavior \
  --output_manifest /path/to/behavior/flows/wds_manifest_seed42_test0.1.json

python convert_wds.py \
  --input_dir /path/to/behavior/flows \
  --output_dir /path/to/behavior/wds \
  --domain behavior \
  --manifest /path/to/behavior/flows/wds_manifest_seed42_test0.1.json
```

To match the test split from the paper, pass the release manifest directly instead of generating one:

```bash
python convert_wds.py \
  --input_dir /path/to/behavior/flows \
  --output_dir /path/to/behavior/wds \
  --domain behavior \
  --manifest manifests/behavior_paper_split_manifest.json
```

<a id="visualize-generated-h5-clips"></a>
## 🎥 Visualize Generated H5 Clips (Optional)

Use `visualization/visualize_generated_h5.py` to open a [viser](https://github.com/nerfstudio-project/viser) viewer for one generated clip directly from `.h5`/`.hdf5` outputs (not final WDS shards).

- `--h5_dir` is required.
- `--h5_name` is optional. If omitted, one H5 file is chosen randomly from `--h5_dir` (recursive search).
- `--clip_key` is optional. If omitted, one clip is chosen randomly from the selected H5 file.
- `--seed` controls random H5/clip selection when `--h5_name` and/or `--clip_key` is omitted.
  Use different seeds to browse different random samples; reuse the same seed for reproducible selection.
- `--max_robot_points` controls the maximum URDF-sampled robot points used for robot-flow visualization (default: `500`).
- Robot-flow sampling is gripper-only by default for both DROID and BEHAVIOR generated H5 visualization.
- The viewer requires root H5 attribute `domain` (`droid` or `behavior`) and uses it to select URDF internally.
- This viewer is ground-truth-only for generated data (no prediction/GT toggle in the UI).

### Random clip from DROID generated H5

```bash
python visualization/visualize_generated_h5.py \
  --h5_dir /path/to/droid/flows-fs-optimize
```

### Random clip from BEHAVIOR generated H5

```bash
python visualization/visualize_generated_h5.py \
  --h5_dir /path/to/behavior/flows
```

To browse different random samples, set `--seed`.
Examples: `--seed 1`, `--seed 2`, `--seed 42` (same seed => same random file/clip choice).

To visualize a specific file and clip, add `--h5_name` and `--clip_key`.
Examples: `--h5_name task-0000_episode_00000050.h5`, `--clip_key 115:126`.

To override viewer binding, add `--viewer_host` and `--viewer_port`.
Examples: `--viewer_host 0.0.0.0`, `--viewer_port 8080`.

<a id="acknowledgements"></a>
## 🙏 Acknowledgements

We gratefully acknowledge the authors and maintainers of third-party projects that this repository depends on or adapts.
Modifications have been made where noted, and the original license terms remain in effect.

Third-party OSS attribution and license references for distributed or adapted code are documented in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

| Repository / Project | Usage in this repo | License |
|---|---|---|
| [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker) | 2D point-track generation (`third_party/co-tracker/`, `real/compute_2d_flows.py`) | [CC BY-NC 4.0](third_party/co-tracker/LICENSE.md) |
| [NVlabs/FoundationStereo](https://github.com/NVlabs/FoundationStereo) | Depth prediction backend (`third_party/FoundationStereo/`, `real/compute_depth.py`) | [FoundationStereo License](third_party/FoundationStereo/LICENSE) |
| [LiheYoung/Depth-Anything](https://github.com/LiheYoung/Depth-Anything) | Included via FoundationStereo depth module (`third_party/FoundationStereo/depth_anything/`) | [Apache-2.0](third_party/FoundationStereo/depth_anything/LICENSE.txt) |
| [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Included via FoundationStereo feature stack (`third_party/FoundationStereo/dinov2/`) | [Apache-2.0](third_party/FoundationStereo/dinov2/LICENSE) |
| [openai/CLIP](https://github.com/openai/CLIP) | Included through DINOv2 third-party components (`third_party/FoundationStereo/dinov2/dinov2/thirdparty/CLIP/`) | [MIT](third_party/FoundationStereo/dinov2/dinov2/thirdparty/CLIP/LICENSE) |
| [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | Camera extrinsics estimation backend (`third_party/vggt/`, `real/compute_extrinsics.py`) | [VGGT License](third_party/vggt/LICENSE.txt) |
| [StanfordVL/OmniGibson](https://github.com/StanfordVL/OmniGibson) | BEHAVIOR simulation backend for data generation (`simulation/behavior_3d_flows.py`) | [MIT](https://github.com/StanfordVL/OmniGibson/blob/main/LICENSE) |
| [droid-dataset/droid_policy_learning](https://github.com/droid-dataset/droid_policy_learning) | Source dataset specification/reference for DROID raw scenes and policy-learning release tooling | [MIT](https://github.com/droid-dataset/droid_policy_learning/blob/master/LICENSE) |

<a id="contributing"></a>
## 🤝 Contributing

All external contributions must follow `CONTRIBUTING.md` in this repository.
In particular, commits must be signed off (`git commit -s`) to satisfy DCO requirements.
