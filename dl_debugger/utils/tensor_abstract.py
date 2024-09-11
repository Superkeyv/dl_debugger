# copyright ZJX

import numpy as np
import torch

from torch import Tensor
from typing import Dict


@torch.compile
def _tensor_hash_stage1(t:Tensor) -> Tensor:
    t = t.reshape(-1)
    t_rev = t.flip(0)
    feat = torch.zeros(2).to(t)
    feat[0] = (t * t_rev).abs().sum()
    feat[1] = t.square().sum()
    return feat


@torch.compile
def _tensor_hash_stage2(feat:Tensor) -> Tensor:
    assert feat.numel() == 2
    return feat.sqrt().mean()


@torch.no_grad()
def tensor_hash(t:Tensor) -> float:
    """ mapping a any tensor to float."""
    if t.numel() == 0:
        return np.nan
    if t.numel() == 1:
        return t.item()

    # Convert dtype to float32 may introduces additional precision errors,
    # so we use default precision
    if not t.is_floating_point():
        t = t.float()
    t = _tensor_hash_stage1(t)
    t = _tensor_hash_stage2(t)
    return t.item()


@torch.no_grad()
def tensor_fingerprint(t:Tensor) -> Dict[str, float]:
    """ use serveral floats describe a tensor"""
    if t.numel() == 0:
        return {}

    t = t.to(torch.float32)
    t_min, t_max = t.aminmax()
    t_mean = t.mean()
    t_norm = t.norm()
    t_hash = tensor_hash(t)
    return {
        'min': t_min.item(),
        'max': t_max.item(),
        'mean': t_mean.item(),
        'norm': t_norm.item(),
        'hash': t_hash,
    }
