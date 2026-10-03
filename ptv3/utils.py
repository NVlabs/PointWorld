"""
General utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import random
import numpy as np
import torch
try:
    import torch.backends.cudnn as cudnn
except (ImportError, AttributeError):  # XPU-only builds
    cudnn = None
from datetime import datetime


@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )


@torch.no_grad()
def bincount2offset(bincount):
    return torch.cumsum(bincount, dim=0)


@torch.no_grad()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)


@torch.no_grad()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def get_random_seed():
    seed = (
        os.getpid()
        + int(datetime.now().strftime("%S%f"))
        + int.from_bytes(os.urandom(2), "big")
    )
    return seed


def set_seed(seed=None):
    if seed is None:
        seed = get_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed(seed)
    if cudnn is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
