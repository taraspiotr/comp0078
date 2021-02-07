from typing import Type, Dict, Any, Optional, Tuple, Union, List

import numpy as np

from utils import train_test_split

Tensor = Type[np.ndarray]


def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x: Tensor) -> Tensor:
    return sigmoid(x) * (1 - sigmoid(x))


def cross_entropy_loss(y_pred: Tensor, y_true: Tensor) -> float:
    return np.mean(-np.log(softmax(y_pred)[np.arange(y_true.shape[0]), y_true]))


def cross_entropy_loss_grad(y_pred: Tensor, y_true: Tensor) -> Tensor:
    q = softmax(y_pred)
    q[np.arange(y_true.shape[0]), y_true] -= 1
    return q / y_true.shape[0]


def softmax(x: Tensor, axis: int = 1) -> Tensor:
    x = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(x)
    return exps / np.sum(exps, axis=axis, keepdims=True)


class Layer:
    def __init__(self) -> None:
        self._input: Tensor = None
        self._output: Tensor = None

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

    @property
    def input(self) -> Tensor:
        return self._input

    @property
    def output(self) -> Tensor:
        return self._output


class LinearLayer(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._init_weights()
        self.w_grad = np.zeros_like(self.w)
        self.b_grad = np.zeros_like(self.b)

    def _init_weights(self) -> None:
        d = np.sqrt(1 / self.input_dim)
        self.w = np.random.rand(self.input_dim, self.output_dim) * d * 2 - d
        self.b = np.random.rand(self.output_dim) * d * 2 - d

    def forward(self, x: Tensor) -> Tensor:
        self._input = x
        self._output = x @ self.w + self.b
        return self._output

    def backward(self, grad: Tensor) -> Tensor:
        self.b_grad += np.sum(grad, axis=0)
        self.w_grad += self._input.T @ grad

        return grad @ self.w.T


class MultiLayerPerceptron:
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int]):
        self._linear_layers = []
        sizes = [input_dim] + hidden_sizes + [output_dim]
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            self._linear_layers.append(LinearLayer(in_dim, out_dim))

    def forward(self, x: Tensor, output_logits: bool = True) -> Tensor:
        for layer in self._linear_layers[:-1]:
            x = sigmoid(layer.forward(x))
        x = self._linear_layers[-1].forward(x)

        return x if output_logits else softmax(x)

    def backward(self, grad: Tensor) -> Tensor:
        grad = self._linear_layers[-1].backward(grad)
        for layer in reversed(self._linear_layers[:-1]):
            grad = grad * sigmoid_grad(layer.output)
            grad = layer.backward(grad)

        return grad

    @property
    def linear_layers(self) -> List[LinearLayer]:
        return self._linear_layers

    def pred(self, x: Tensor) -> Tensor:
        return np.argmax(self.forward(x), axis=1)


class SGD:
    def __init__(self, net: MultiLayerPerceptron, lr: float):
        self._net = net
        self._lr = lr

    def zero_grad(self) -> None:
        for layer in self._net.linear_layers:
            layer.w_grad = np.zeros_like(layer.w_grad)
            layer.b_grad = np.zeros_like(layer.b_grad)

    def step(self) -> None:
        for layer in self._net.linear_layers:
            layer.w -= self._lr * layer.w_grad
            layer.b -= self._lr * layer.b_grad


def batchify(*x: Tensor, batch_size: int = 8) -> List[List[Tensor]]:
    return [
        [xx[start : start + batch_size] for xx in x]
        for start in np.arange(0, x[0].shape[0], batch_size)
    ]


def error(logits: Tensor, y_true: Tensor) -> float:
    return np.mean(np.argmax(logits, axis=1) != y_true)


def _train(
    num_classes: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    hyperparameters: Dict[str, Any],
) -> Tuple[Any, Tuple[float, float]]:
    net = MultiLayerPerceptron(
        input_dim=x_train.shape[1],
        output_dim=num_classes,
        hidden_sizes=hyperparameters["hidden_sizes"],
    )
    optim = SGD(net, hyperparameters["lr"])
    batch_size = hyperparameters["batch_size"]
    for _ in range(50):
        for x, y in batchify(x_train, y_train, batch_size=batch_size):
            optim.zero_grad()
            pred = net.forward(x)
            grad = cross_entropy_loss_grad(pred, y)
            net.backward(grad)
            optim.step()
    train_error = error(net.forward(x_train), y_train)
    valid_error = error(net.forward(x_valid), y_valid)
    return net, (train_error, valid_error)


def train_nn(
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
    model, (train_err, valid_err) = _train(
        num_classes, x_train, y_train, x_valid, y_valid, params
    )
    return (model, (train_err, valid_err)) if return_model else (train_err, valid_err)
