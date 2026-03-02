# Third-Party OSS Licenses and Notices

This repository includes and/or adapts third-party open-source software (OSS).
The components below remain subject to their original license terms.

## Included or Adapted Components

| Component | Local Path(s) | Upstream | License | License Reference |
|---|---|---|---|---|
| DINOv3 | `third_party/dinov3/` | https://github.com/facebookresearch/dinov3 | DINOv3 License | `third_party/dinov3/LICENSE.md` |
| PointTransformerV3 (PTv3 lineage) | `ptv3/` | https://github.com/Pointcept/PointTransformerV3 | MIT | `ptv3/LICENSE` |
| Sonata (reference lineage for PTv3 integration) | `ptv3/` integration lineage | https://github.com/facebookresearch/sonata | Apache-2.0 | https://github.com/facebookresearch/sonata/blob/main/LICENSE |
| OmniGibson (adapted transforms) | `transform_utils.py`, `deploy/transform_utils_torch.py` | https://github.com/StanfordVL/OmniGibson | MIT | https://github.com/StanfordVL/OmniGibson/blob/main/LICENSE |
| deoxys_control (adapted routines) | `transform_utils.py` (annotated in-file) | https://github.com/UT-Austin-RPL/deoxys_control | Apache-2.0 | https://github.com/UT-Austin-RPL/deoxys_control/blob/main/LICENSE |

## Distribution Notes

- Project primary license: `LICENSE` (Apache-2.0).
- Third-party code keeps upstream attribution and licensing requirements.
- If additional copied/adapted third-party code is introduced, update this file and link the corresponding license source.

