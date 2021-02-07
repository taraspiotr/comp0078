from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.spatial.distance import cdist


Kernel = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _check_kernel_input(x: np.ndarray, t: np.ndarray) -> None:
    if x.ndim == 1:
        x = np.expand_dims(x, 0)
    if t.ndim == 1:
        t = np.expand_dims(t, 0)

    assert x.ndim == t.ndim == 2, f"Wrong input dimensions: {x.ndim, t.ndim}"
    assert x.shape[1] == t.shape[1], f"Dimension mismatch: {x.shape, t.shape}"


def get_polynomial_kernel(d: int) -> Kernel:
    def _kernel(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.power(x @ t.T, d)

    def kernel(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        _check_kernel_input(x, t)
        return _kernel(x, t)

    return kernel


def get_gaussian_kernel(c: float) -> Kernel:
    def _kernel(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.exp(-c * cdist(x, t))

    def kernel(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        _check_kernel_input(x, t)
        return _kernel(x, t)

    return kernel
