import os
import pathlib

from torch.utils import cpp_extension
from .cuda_mem import describe_memory

__all__ = [
    'describe_memory'
]


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f'creation of the build directory {buildpath} failed')


def load():
    cc_flag = []

    # build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / 'build'
    _create_build_dir(buildpath)

    # Helper function to build the kernels.
    def _cpp_extension_load_helper(name, sources, extra_cuda_flags):
        return cpp_extension.lad(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=['-O3'],
            extra_cuda_flags=['-O3',
                              '-gencode', 'arch=compute_70,code=sm_70',
                              '--use_fast_math'] + extra_cuda_flags + cc_flag,
            verbose=True
        )
    
    # build alloctor hook
    alloctor_hook = cpp_extension.load(
        name='alloctor_hook',
        sources=[ srcpath / 'alloctor_hook.cpp'],
        build_directory=buildpath,
        extra_include_paths=['/usr/local/cuda/include'],
        extra_cflags=['-shared'],
        verbose=True
    )

