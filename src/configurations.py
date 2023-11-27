import os
import sys

import torch


def gen_configurations(n, vocab=[-1, 1]):
    def _gen(n, arr, res):
        if len(res) == n:
            arr.append(res)
        else:
            for c in vocab:
                _gen(n, arr, res + [c])

    # Generate all configurations
    arr = []
    _gen(n, arr, [])
    arr = torch.FloatTensor(arr).transpose(0, 1)

    return arr


def load_configurations(n):
    if not os.path.exists(f'configurations/{n}.pth'):
        os.makedirs('configurations', exist_ok=True)
        x = gen_configurations(n)
        torch.save(x, f'configurations/{n}.pth')
    else:
        x = torch.load(f'configurations/{n}.pth')
    return x


if __name__ == '__main__':
    n = int(sys.argv[1])
    gen_configurations(n)
