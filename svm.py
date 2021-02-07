from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Union

import cvxpy as cp
import numpy as np

from utils import train_test_split
from kernel import Kernel, get_gaussian_kernel, get_polynomial_kernel


def solve_svm(K: np.ndarray, y: np.ndarray) -> np.ndarray:

    n = len(y)
    ones = np.ones(n)
    zeros = np.zeros(n)

    try:
        alpha = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(
                (1 / 2) * cp.quad_form(alpha, np.diag(y) @ K @ np.diag(y))
                - ones @ alpha
            ),
            [-alpha <= zeros, y @ alpha == 0],
        )
        prob.solve()
        return alpha.value
    except cp.error.SolverError:
        return np.zeros(n)


class BinarySVM:
    def __init__(self, kernel: Kernel):
        self._kernel = kernel
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._X = X
        self._y = y * 2 - 1
        K = self._kernel(X, X)
        self._alpha = solve_svm(K, self._y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        K = self._kernel(X, self._X)
        return (K @ (self._alpha * self._y)) > 0


class SVM:
    def __init__(self, num_classes: int, kernel: Kernel):
        self.num_classes = num_classes
        self._kernel = kernel
        self._classifiers = np.array(
            [BinarySVM(kernel) for _ in range(self.num_classifiers)]
        )

    @property
    def num_classifiers(self) -> int:
        return self.num_classes * (self.num_classes - 1) // 2

    def classifiers_for(self, ind: int) -> np.ndarray:
        return np.arange(ind * (ind - 1) // 2, ind * (ind - 1) // 2 + ind)

    def classifiers_against(self, ind: int) -> np.ndarray:
        return np.cumsum(range(self.num_classes - 1))[ind:] + ind

    def clf_ind(self, i: int, j: int) -> int:
        return i * (i - 1) // 2 + j

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for i in range(self.num_classes):
            for j in range(i):
                ind = (y == i) | (y == j)
                self._classifiers[self.clf_ind(i, j)].fit(X[ind], y[ind] == i)

    def transform(self, X: np.ndarray, output_logits: bool = False) -> np.ndarray:
        pred = np.stack([clf.transform(X) for clf in self._classifiers], axis=0)
        votes = []
        for y in range(self.num_classes):
            clf_for, clf_agn = self.classifiers_for(y), self.classifiers_against(y)
            votes.append(
                np.concatenate([pred[clf_for] > 0, pred[clf_agn] <= 0], axis=0)
            )
        votes = np.sum(np.stack(votes, axis=0), axis=1)
        return votes if output_logits else np.argmax(votes, axis=0)

    def pred(self, X: np.ndarray, output_logits: bool = False) -> np.ndarray:
        return self.transform(X, output_logits)


def _train(
    kernel: Kernel,
    num_classes: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
) -> Tuple[Any, Tuple[float, float]]:
    svm = SVM(num_classes, kernel)
    svm.fit(x_train, y_train)

    train_error = np.mean(svm.transform(x_train) != y_train)
    valid_error = np.mean(svm.transform(x_valid) != y_valid)

    return svm, (train_error, valid_error)


def train_svm(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    params: Dict[str, Any],
    random_state: Optional[int] = None,
    x_valid: Optional[np.ndarray] = None,
    y_valid: Optional[np.ndarray] = None,
    return_model: bool = False,
) -> Union[Tuple[Any, Tuple[float, float]], Tuple[float, float]]:
    if x_valid is None:
        # pylint: disable=unbalanced-tuple-unpacking
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, test_size=0.2, random_state=random_state
        )
    else:
        x_train, y_train = x, y
    kernel_type = {
        "polynomial": get_polynomial_kernel,
        "gaussian": get_gaussian_kernel,
    }[params["kernel"]]
    model, (train_err, valid_err) = _train(
        kernel_type(params["kernel_param"]),
        num_classes,
        x_train,
        y_train,
        x_valid,
        y_valid,
    )
    return (model, (train_err, valid_err)) if return_model else (train_err, valid_err)
