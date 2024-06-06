import numpy as np
import torch

from torch import Tensor


def reproducible_init_method(w:Tensor, loc=0, scale=1, inplace=False):
    """ reproducible init method """
    if not isinstance(w, Tensor):
        return w

    seed = w.nelement() % 97
    rng = np.random.Generator(np.random.MT19937(seed=seed))
    dst = rng.normal(size=w.shape, loc=loc, scale=scale)
    dst = torch.from_numpy(dst)

    if inplace:
        with torch.no_grad():
            if isinstance(w, torch.nn.Parameter):
                w.data.copy_(dst)
            else:
                w.copy_(dst)
        return w
    return dst


def reproducible_init_method_(w:Tensor, loc=0, scale=1):
    """ in-place init method """
    return reproducible_init_method(w=w, loc=loc, scale=scale, inplace=True)
