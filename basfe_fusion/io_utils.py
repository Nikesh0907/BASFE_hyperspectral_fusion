import os
from typing import Tuple, Optional
import numpy as np
import scipy.io as sio


def load_mat_first_array(path: str) -> np.ndarray:
    """Load a .mat file and return the first ndarray found.

    This is a robust fallback when variable names differ across datasets.
    """
    m = sio.loadmat(path)
    for k, v in m.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray):
            return v.astype(np.float32)
    raise ValueError(f"No ndarray found in {path}")


def mat_save(path: str, key: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sio.savemat(path, {key: arr})


def minmax01(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    mn = a.min()
    mx = a.max()
    if mx <= mn:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)
