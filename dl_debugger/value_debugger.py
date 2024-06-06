""" check whether invalid value appear. """

import sys
import torch
import torch.nn as nn

from collections.abc import Mapping, Callable
from functools import partial

from .utils import logger, nested_check_tensor, HookManager, ListTuple


def __all_is(x: torch.Tensor, fn: Callable):
    assert fn in [torch.isfinite, torch.isnan, torch.isinf, 
                  torch.isposinf, torch.isneginf, torch.isreal]
    return fn(x).any().item()
    

__is_inf_or_nan = partial(__all_is, fn=torch.isfinite)

__recursion_check_tensor_isfinite = partial(
    nested_check_tensor, fn=partial(
        __all_is, fn=torch.isfinite))


HM = HookManager('inf nan check')

def check_model_forward_infinite(model: nn.Module, only_training_module:bool=False):
    """ check nan/inf appear at infer stage and print stack info.
    
    >>> model = ResNet50(...)
    >>> check_model_forward_infinite(model)
    >>> result = model(inp_x, inp_y)
    """

    def __hook_check_isfinite(module, input_) -> None:
        if only_training_module and not (
            module.training and torch.is_grad_enabled()
        ):
            return
        
        if isinstance(input_, ListTuple):
            for i, v in enumerate(input_):
                if not __recursion_check_tensor_isfinite(v):
                    continue
                logger.debug('in-param [%d] with [inf/nan]', i,
                             stack_info=True,
                             stacklevel=2)
                sys.exit(1)
        if isinstance(input_, Mapping):
            for k, v in input_.items():
                if not __recursion_check_tensor_isfinite(v):
                    continue
                logger.debug('in-param [%s] with [inf/nan]', k,
                             stack_info=True,
                             stacklevel=2)
                sys.exit(1)
        else:
            if __recursion_check_tensor_isfinite(input_):
                logger.debug('in-param with [inf/nan]',
                            stack_info=True,
                            stacklevel=2)
                sys.exit(1)

    for name, mod in model.named_modules():
        hdl = mod.register_forward_pre_hook(__hook_check_isfinite)
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
        if __is_inf_or_nan(param):
            has = True
            logger.warning(' nan detect # %s', name)

    return has
