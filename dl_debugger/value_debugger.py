# copyright ZJX

""" check whether invalid value appear. """

import re
import sys
import torch
import torch.nn as nn

from collections.abc import Mapping
from functools import partial

from .utils import (
    HookManager, ListTuple, logger,
    check_nested_tensor_any, check_nested_tensor_all,
)


################## define some basic op ##################
isnan = lambda x: torch.isnan(x).any().item()
isfinite = lambda x: torch.isfinite(x).any().item()

check_nested_tensor_any_isnan = partial(
    check_nested_tensor_any, fn=isnan
)
check_nested_tensor_all_isfinite = partial(
    check_nested_tensor_all, fn=isfinite
)


################## define value check hook manager ##################
HM = HookManager('inf nan check')

def register_check_model_fwd_nan(model: nn.Module,
                                 pattern:str='.*',
                                 only_training_module:bool=False):
    """ check nan/inf appear at infer stage and print stack info.
    
    >>> model = ResNet50(...)
    >>> register_check_model_fwd_nan(model)
    >>> result = model(inp_x, inp_y)
    """

    def __hook_check_isnan(module, input_) -> None:
        if only_training_module and not (
            module.training and torch.is_grad_enabled()
        ):
            return
        
        if isinstance(input_, ListTuple):
            for i, v in enumerate(input_):
                if check_nested_tensor_all_isfinite(v):
                    continue
                logger.debug('in-param [%d] with [inf/nan]', i,
                             stack_info=True,
                             stacklevel=2)
                sys.exit(1)
        if isinstance(input_, Mapping):
            for k, v in input_.items():
                if check_nested_tensor_all_isfinite(v):
                    continue
                logger.debug('in-param [%s] with [inf/nan]', k,
                             stack_info=True,
                             stacklevel=2)
                sys.exit(1)
        else:
            if not check_nested_tensor_all_isfinite(input_):
                logger.debug('in-param with [inf/nan]',
                            stack_info=True,
                            stacklevel=2)
                sys.exit(1)

    for name, mod in model.named_modules():
        if not re.fullmatch(pattern, name):
            continue

        hdl = mod.register_forward_pre_hook(__hook_check_isnan)
        HM.reg_handle(name, hdl)

    logger.info('value debugger hook number: %d', HM.get_handle_count())
    return model


def clean_hook():
    """ remove hook manager. """
    HM.clear()


def check_model_param_infinite(model: nn.Module):
    """ check nan/inf appear at model params. 
    
    >>> model = ResNet50(...)
    >>> check_model_param_infinite(model)
    >>> result = model(inp_x, inp_y)
    """

    has = False
    for name, param in model.parameters():
        if isnan(param):
            has = True
            logger.warning(' nan detect # %s', name)

    return has
