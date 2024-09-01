# copyright ZJX

""" auto mark nvtx for every module."""

import functools
import re
import torch

from torch.cuda import nvtx

from .utils import logger, HookManager


def instrument_w_nvtx(func):
    """decorator that causes an NVTX range to be recorded for the duration of the
    function call."""
    def wrapped_fn(*args, **kwargs):
        nvtx.range_push(func.__qualname__)
        ret_val = func(*args, **kwargs)
        nvtx.range_pop()
        return ret_val

    return wrapped_fn


HM = HookManager('nvtx')

def register_nvtx_mark(model:torch.nn.Module,
                       pattern:str='.*'):

    def fwd_pre_hook(name, *args, **kwds):
        nvtx.range_push(name+'_fwd')

    def bwd_pre_hook(name, *args, **kwds):
        nvtx.range_push(name+'_bwd')

    def post_hook(*args, **kwds):
        nvtx.range_pop()

    for name, mod in model.named_modules():
        if not re.fullmatch(pattern, name):
            continue

        fn = functools.partial(fwd_pre_hook, name)
        hdl = mod.register_forward_pre_hook(fn, with_kwargs=True)
        HM.reg_handle(name, hdl)

        fn = functools.partial(bwd_pre_hook, name)
        hdl = mod.register_full_backward_pre_hook(fn)
        HM.reg_handle(name, hdl)

        hdl = mod.register_forward_hook(post_hook, with_kwargs=True)
        HM.reg_handle(name, hdl)
        hdl = mod.register_full_backward_hook(post_hook)
        HM.reg_handle(name, hdl)

    logger.info('nvtx hook number: %d', HM.get_handle_count())


def clean_hook():
    """ remove hook."""
    HM.clear()
