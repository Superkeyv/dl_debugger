# copyright ZJX

import pathlib
import torch

def describe_memory(stage: str, rank: int, dump_path: str, device=None):
    with open(dump_path + f'/memory_check_rank{rank}.txt', 'a') as f:
        print(f'>>> {stage}: \n{torch.cuda.memory_summary(device)}\n')
        print(f'>>> {stage}: \n{torch.cuda.memory_summary(device)}\n', file=f)

def init_memory_sanitizer():
    from . import load
    load()

    srcpath = pathlib.Path(__file__).parent.absolute()
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        srcpath / 'build' / 'alloctor_hook.so',
        'ds_malloc', 'ds_free'
    )
    torch.cuda.memory.change_current_allocator(new_alloc)


def free_all_cuda_memory():
    from .build.alloctor_hook import free_all_mem
    free_all_mem()