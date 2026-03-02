<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Third-Party OSS Licenses

This document lists third-party OSS code that is included in or adapted by this repository for release workflows.

## Included Components

| Component | Upstream | Local Path | License Reference |
|---|---|---|---|
| CoTracker | https://github.com/facebookresearch/co-tracker | `third_party/co-tracker/` | `third_party/co-tracker/LICENSE.md` (Creative Commons Attribution-NonCommercial 4.0) |
| FoundationStereo | https://github.com/NVlabs/FoundationStereo | `third_party/FoundationStereo/` | `third_party/FoundationStereo/LICENSE` |
| Depth Anything (included via FoundationStereo) | https://github.com/LiheYoung/Depth-Anything | `third_party/FoundationStereo/depth_anything/` | `third_party/FoundationStereo/depth_anything/LICENSE.txt` (Apache-2.0) |
| DINOv2 (included via FoundationStereo) | https://github.com/facebookresearch/dinov2 | `third_party/FoundationStereo/dinov2/` | `third_party/FoundationStereo/dinov2/LICENSE` (Apache-2.0) |
| CLIP (included via DINOv2 thirdparty) | https://github.com/openai/CLIP | `third_party/FoundationStereo/dinov2/dinov2/thirdparty/CLIP/` | `third_party/FoundationStereo/dinov2/dinov2/thirdparty/CLIP/LICENSE` (MIT) |
| VGGT | https://github.com/facebookresearch/vggt | `third_party/vggt/` | `third_party/vggt/LICENSE.txt` |

## Adapted Components

| Component | Upstream | Local Path | License Reference |
|---|---|---|---|
| OmniGibson transform utilities (adapted portions) | https://github.com/StanfordVL/OmniGibson | `transform_utils.py` | Upstream `LICENSE` (MIT): https://github.com/StanfordVL/OmniGibson/blob/main/LICENSE |

## Notes

- Third-party license files are kept in their component directories and should be distributed together with the corresponding third-party code.
- This file is release-facing documentation and does not replace obligations in each upstream license.
