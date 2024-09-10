# copyright ZJX

""" insert hook for model to check forward and backward details."""

import functools
import re
import torch

from .utils import logger, HookManager, any_to_str


def register_fwd_hook(model:torch.nn.Module,
                      pattern:str='.*',
                      filename:str='fwd_hook',
                      only_training_module=False):
    hm = HookManager(filename)

    def fwd_hook(name, module, args, kwargs, output):
        if only_training_module and not (
            module.training and torch.is_grad_enabled()
        ):
            return
        
        # TODO need fix, when args has varlen list or dict
        p1 = any_to_str(args)
        p2 = any_to_str(kwargs)
        p3 = any_to_str(output)
        logger.debug(f' {name} # {p1} {p2} | {p3}')
        hm.add_to_table(name, {
            'input': (args, kwargs),
            'output': output})
        
    def fwd_pre_hook(name, module, args, kwargs):
        p1 = any_to_str(args)
        p2 = any_to_str(kwargs)
        logger.debug(f' {name} # {p1} {p2}')

    for name, mod in model.named_modules():
        if not re.fullmatch(pattern, name):
            continue

        # fn_pre = functools.partial(fwd_pre_hook, name)
        # mod.register_forward_pre_hook(fn_pre, with_kwargs=True)
        fn = functools.partial(fwd_hook, name)
        hdl = mod.register_forward_hook(fn, with_kwargs=True)
        hm.reg_handle(name, hdl)

    logger.info('fwd hook number: %d', hm.get_handle_count())


def register_bwd_hook(model:torch.nn.Module,
                      pattern:str='.*',
                      filename:str='bwd_hook'):
    hm = HookManager(filename)

    def bwd_hook(name, module, grad_input, grad_output):
        p1 = any_to_str(grad_input)
        p2 = any_to_str(grad_output)
        logger.debug(f' {name} # {p1} | {p2}')
        hm.add_to_table(name, {
            'input': grad_input,
            'output': grad_output})
        
    def bwd_pre_hook(name, module, grad_output):
        p1 = any_to_str(grad_output)
        logger.debug(f' {name} # {p1}')

    for name, mod in model.named_modules():
        if not re.fullmatch(pattern, name):
            continue

        # fn_pre = functools.partial(bwd_pre_hook, name)
        # mod.register_full_backward_pre_hook(fn_pre)
        fn = functools.partial(bwd_hook, name)
        hdl = mod.register_full_backward_hook(fn)
        hm.reg_handle(name, hdl)

    logger.info('bwd hook number: %d', hm.get_handle_count())


BWD_HOOK_NODE_HM = None

def register_bwd_hook_graph_node(node:torch.Tensor,
                                 name,
                                 filename='autograd_san_bwd_node'):
    global BWD_HOOK_NODE_HM
    if BWD_HOOK_NODE_HM is None:
        BWD_HOOK_NODE_HM = HookManager(filename)

    if not node.requires_grad:
        logger.warning("tensor don't requires grad, %s", name)
        return
    
    if not isinstance(node.grad_fn, torch.autograd.graph.Node):
        logger.info("tensor not in graph, %s", name)
        return

    if BWD_HOOK_NODE_HM.is_registered(name):
        return
    
    def bwd_hook(grad_inputs, grad_outputs):
        assert BWD_HOOK_NODE_HM, "BWD_HOOK_NODE_HM can't be None"
        p1 = any_to_str(grad_inputs)
        p2 = any_to_str(grad_outputs)
        logger.debug(f' {name} # {p1} | {p2}')
        BWD_HOOK_NODE_HM.add_to_table(name, {
            'input': grad_inputs,
            'output': grad_outputs})
        BWD_HOOK_NODE_HM.remove_hook(name)
        
    hdl = node.grad_fn.register_hook(bwd_hook)
    BWD_HOOK_NODE_HM.reg_handle(name, hdl)
