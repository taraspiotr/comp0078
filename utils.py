from typing import List, Any, Generator, Dict, Tuple, Optional, Callable

import os
from itertools import product

import numpy as np
import psutil
from tqdm import tqdm


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


def _kfold(
    train_func: Callable,
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    grid: List[Dict[str, Any]],
    random_state: int,
) -> Dict[str, Any]:
    best_err, best_params = 1, dict()
    for params in tqdm(grid):
        kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
        errors = []
        for train_ind, valid_ind in kfold.split(x):
            x_train, y_train = x[train_ind], y[train_ind]
            x_valid, y_valid = x[valid_ind], y[valid_ind]
            _, valid_err = train_func(
                x_train, y_train, num_classes, params, x_valid=x_valid, y_valid=y_valid
            )
            errors.append(valid_err)
        mean_err = np.mean(errors)
        if mean_err < best_err:
            best_err = mean_err
            best_params = params
    return best_params


def get_cm(model: Any, x: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes))
    for pred, gt in zip(model.pred(x), y):
        if pred != gt:
            cm[pred, gt] += 1
            cm[gt, pred] += 1
    return cm / len(y)


def run_kfold(
    train_func: Callable,
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    grid: List[Dict[str, Any]],
    random_state: int,
) -> Tuple[Dict[str, Any], float, np.ndarray]:
    # pylint: disable=unbalanced-tuple-unpacking
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state
    )
    best_params = _kfold(train_func, x_train, y_train, num_classes, grid, random_state)

    model, (_, test_err) = train_func(
        x_train,
        y_train,
        num_classes,
        best_params,
        x_valid=x_test,
        y_valid=y_test,
        return_model=True,
    )
    cm = get_cm(model, x_test, y_test, num_classes)

    return best_params, test_err, cm
