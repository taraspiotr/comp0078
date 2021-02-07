from typing import List, Any, Generator, Dict, Tuple, Optional

import os
from itertools import product

import numpy as np
import psutil


def product_dict(**kwargs: List[Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Thanks to:
    https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


def limit_cpu() -> None:
    p = psutil.Process(os.getpid())
    p.nice(19)


def read_data(path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    data = np.loadtxt(path)
    x, y = data[:, 1:], data[:, 0].astype(np.int)
    num_classes = 10

    return x, y, num_classes


def train_test_split(
    *x: np.ndarray, test_size: float = 0.2, random_state: Optional[int] = None
) -> List[np.ndarray]:
    rand_gen = np.random.RandomState(random_state)
    n = len(x[0])
    test_ind = rand_gen.choice(n, int(n * test_size), replace=False)
    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_ind] = True
    split_x = []
    for arr in x:
        split_x.append(arr[np.logical_not(test_mask)])
        split_x.append(arr[test_mask])

    return split_x


class KFold:
    def __init__(
        self, n_splits: int, shuffle: bool = False, random_state: Optional[int] = None
    ):
        self._n_splits = n_splits
        self._shuffle = shuffle
        self._rand_gen = np.random.RandomState(random_state)

    def split(
        self, x: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n = len(x)
        inds = self._rand_gen.permutation(n) if self._shuffle else np.arange(n)
        split_size = int(n / self._n_splits)

        for i in range(self._n_splits):
            train_ind = np.concatenate(
                [inds[: i * split_size], inds[(i + 1) * split_size :]]
            )
            valid_ind = inds[i * split_size : (i + 1) * split_size]
            yield train_ind, valid_ind
