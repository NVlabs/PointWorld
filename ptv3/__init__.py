"""
PTv3 package provenance:
- Pointcept PTv3 upstream: https://github.com/Pointcept/PointTransformerV3
- Sonata upstream: https://github.com/facebookresearch/sonata

This directory keeps upstream attribution/licensing and intentionally does not
add NVIDIA SPDX file headers to vendored third-party source files.
"""

from . import ptv3
from . import module
from . import structure
from . import utils

__all__ = ["ptv3", "module", "structure", "utils"]
