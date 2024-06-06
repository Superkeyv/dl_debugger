import matplotlib.pyplot as plt
import numpy as np
import torch

from dl_debugger.utils import tensor_hash


def test_tensor_hash_senstive(M, N, dtype):
    base = torch.normal(0, 10, (M, N), dtype=dtype)
    base_hash = tensor_hash(base)

    x_p = []
    y_p = []

    for i in range(5):
        scale = 10**(-i+1)
        noise = torch.normal(0, scale, (M, N))
        noise_hash = tensor_hash(base+noise)

        ref_dB = -i*10
        rerr = np.abs((noise_hash-base_hash) / base_hash)
        dB = 10 * np.log10(rerr)

        if np.isnan(dB):
            print(f'{M:4d}x{N:4d} dB: {ref_dB:2d} base_hash or noise_hash=inf')
        elif np.isinf(dB):
            print(f'{M:4d}x{N:4d} db: {ref_dB:2d} -inf')
        else:
            dB = int(np.round(dB))
            print(f'{M:4d}x{N:4d} db :{ref_dB:2d} {dB:2d}')
            x_p.append(ref_dB)
            y_p.append(dB)

    return x_p, y_p



for dtype in [torch.float32, torch.bfloat16]:
    x = []
    y = []

    for M in [32, 64, 128, 512, 1024, 2048]:
        for N in [32, 64, 128, 512, 1024, 2048]:
            x_p, y_p = test_tensor_hash_senstive(M, N, dtype)
            x.extend(x_p)
            y.extend(y_p)

    print(f'total points {len(x)}')

    fig = plt.scatter(x, y)
    plt.title(f'tensor hash dB {dtype}')
    plt.xlabel('base dB')
    plt.ylabel('hash dB')
    plt.savefig(f'tensor_hash_cmp_{dtype}.png')
