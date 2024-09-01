# copyright ZJX

""" Convert any struct to String. """

import torch

from typing import Mapping

from .data_struct import ListTuple
from .tensor_abstract import tensor_hash


def tensor_to_str(t:torch.Tensor) -> str:
    rv = tensor_hash(t)
    return f'{rv:.5e}'


def any_to_str(obj) -> str:
    if isinstance(obj, torch.Tensor):
        try:
            return tensor_to_str(obj)
        except: # pylint: disable=W0702
            return 'Err'

    if obj is None:
        return 'None'

    if isinstance(obj, ListTuple):
        rp = [any_to_str(t) for t in obj]
        return ', '.join(rp)

    if isinstance(obj, Mapping):
        rp = [f'{k}={any_to_str(v)}' for k, v in obj.items()]
        return ', '.join(rp)

    return str(obj)
