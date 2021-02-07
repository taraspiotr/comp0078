from __future__ import annotations

from typing import Type, Dict, Any, Optional, Tuple, Union

from copy import deepcopy

import numpy as np

from utils import train_test_split
from kernel import Kernel, get_gaussian_kernel, get_polynomial_kernel


class Perceptron:
    def __init__(self, num_classes: int, x: np.ndarray, y: np.ndarray, kernel: Kernel):
        self.num_classes = num_classes

        self._kernel = kernel
        self._num_datapoints = len(x)
        self._x = x
        self._y = y
        self._gram = self._kernel(x, x)

    def epoch(self) -> Perceptron:
        raise NotImplementedError

    def pred(self, x: np.ndarray, output_logits: bool = False) -> np.ndarray:
        raise NotImplementedError


class OneVsAllPerceptron(Perceptron):
    def __init__(self, num_classes: int, x: np.ndarray, y: np.ndarray, kernel: Kernel):

        super().__init__(num_classes, x, y, kernel)
        self._alpha = np.zeros((self.num_classes, self._num_datapoints))

    def epoch(self) -> Perceptron:
        for i in range(self._num_datapoints):
            y_pred = self._alpha @ self._gram[i].T
            alpha = (self._y[i] == np.arange(self.num_classes)).astype(np.int) - (
                y_pred > 0
            ).astype(np.int)
            self._alpha[:, i] += alpha
        return self

    def pred(self, x: np.ndarray, output_logits: bool = False) -> np.ndarray:
        k = self._kernel(x, self._x)
        pred = self._alpha @ k.T
        return pred if output_logits else np.argmax(pred, axis=0)


class OneVsOnePerceptron(Perceptron):
    def __init__(self, num_classes: int, x: np.ndarray, y: np.ndarray, kernel: Kernel):

        super().__init__(num_classes, x, y, kernel)
        self._num_classifiers = num_classes * (num_classes - 1) // 2
        self._alpha = np.zeros((self._num_classifiers, self._num_datapoints))

    def classifiers_for(self, ind: int) -> np.ndarray:
        return np.arange(ind * (ind - 1) // 2, ind * (ind - 1) // 2 + ind)

    def classifiers_against(self, ind: int) -> np.ndarray:
        return np.cumsum(range(self.num_classes - 1))[ind:] + ind

    def epoch(self) -> Perceptron:
        for i in range(self._num_datapoints):
            y = self._y[i]
            clf_for, clf_agn = self.classifiers_for(y), self.classifiers_against(y)
            pred = (
                np.concatenate([self._alpha[clf_for], -self._alpha[clf_agn]])
                @ self._gram[i]
            )
            self._alpha[clf_for, i] += (pred[: len(clf_for)] <= 0).astype(np.int)
            self._alpha[clf_agn, i] -= (pred[len(clf_for) :] <= 0).astype(np.int)
        return self

    def pred(self, x: np.ndarray, output_logits: bool = False) -> np.ndarray:
        k = self._kernel(x, self._x)
        pred = self._alpha @ k.T
        votes = []
        for y in range(self.num_classes):
            clf_for, clf_agn = self.classifiers_for(y), self.classifiers_against(y)
            votes.append(
                np.concatenate([pred[clf_for] > 0, pred[clf_agn] <= 0], axis=0)
            )
        votes = np.sum(np.stack(votes, axis=0), axis=1)
        return votes if output_logits else np.argmax(votes, axis=0)


def error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean(y_pred != y_true)


def _train(
    perceptron_class: Type[Perceptron],
    kernel: Kernel,
    num_classes: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    max_iterations: int,
    max_iterations_since_best: int,
) -> Tuple[Any, Tuple[float, float]]:
    model = perceptron_class(num_classes, x_train, y_train, kernel)
    best_model, best_train_err, best_valid_err = None, 1.0, 1.0
    it_since_best = 0
    for _ in range(max_iterations):
        model.epoch()
        train_err = error(model.pred(x_train), y_train)

        if train_err < best_train_err:
            best_model = deepcopy(model)
            best_train_err = train_err
            best_valid_err = error(model.pred(x_valid), y_valid)
            it_since_best = 0

        it_since_best += 1
        if it_since_best > max_iterations_since_best:
            break

    return best_model, (best_train_err, best_valid_err)


def train_perceptron(
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
        params["perceptron_type"],
        kernel_type(params["kernel_param"]),
        num_classes,
        x_train,
        y_train,
        x_valid,
        y_valid,
        100,
        3,
    )
    return (model, (train_err, valid_err)) if return_model else (train_err, valid_err)
