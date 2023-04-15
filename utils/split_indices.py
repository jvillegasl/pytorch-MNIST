import numpy as np

from typing import List

__all__ = ['split_indices']


def split_indices(n: int, val_pct: float = 0.1, seed: int = 99):
    n_val = int(val_pct*n)
    np.random.seed(seed)
    idxs: List[int] = np.random.permutation(n).astype(int).tolist()

    return idxs[n_val:], idxs[:n_val]
