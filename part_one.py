from typing import Dict, Any, Tuple, Callable, List, Optional

import multiprocessing
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import product_dict, limit_cpu, read_data, train_test_split, KFold
from perceptron import train_perceptron, OneVsAllPerceptron, OneVsOnePerceptron
from neuralnet import train_nn
from svm import train_svm


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
    run_task_four: bool = False,
    output_path: Optional[str] = None,
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

    if run_task_four:
        worst = np.argsort(
            np.max(model.pred(x_test, output_logits=True), axis=0)
            * (model.pred(x_test) != y_test)
        )[-5:]

        x_worst = x_test[worst]
        y_worst = y_test[worst]
        pred_worst = model.pred(x_test)[worst]

        _, ax = plt.subplots(1, 5, figsize=(14, 5))
        for i in range(5):
            ax[i].imshow(x_worst[4 - i].reshape(16, 16))
            ax[i].set_title(f"gt: {y_worst[4-i]}, pred: {pred_worst[4-i]}")

        plt.savefig(output_path)

    return best_params, test_err, cm


def param_grid(args: Namespace) -> List[Dict[str, List[Any]]]:
    kernel_options: Dict[str, List] = {
        "polynomial": list(range(1, 8)),
        "gaussian": [0.1, 0.5, 1, 2],
    }

    if args.algorithm == "perceptron":

        perceptron_type = {
            "one-vs-all": OneVsAllPerceptron,
            "one-vs-one": OneVsOnePerceptron,
        }[args.multiclass]

        kernel_params = kernel_options[args.kernel]
        grid = [
            {
                "kernel_param": kernel_param,
                "kernel": args.kernel,
                "perceptron_type": perceptron_type,
            }
            for kernel_param in kernel_params
        ]
    elif args.algorithm == "svm":
        kernel_params = kernel_options[args.kernel]
        grid = [
            {"kernel_param": kernel_param, "kernel": args.kernel,}
            for kernel_param in kernel_params
        ]
    else:
        grid = list(
            product_dict(hidden_sizes=[[], [128]], batch_size=[8, 16], lr=[1e-1, 1e-2])
        )

    return grid


def task_one(args: Namespace) -> None:
    x, y, num_classes = read_data(args.data)
    if args.debug:
        x, y = x[:100], y[:100]

    train_func: Dict[str, Callable] = {
        "perceptron": train_perceptron,
        "svm": train_svm,
        "nn": train_nn,
    }
    grid = param_grid(args)

    errors = []
    stds = []
    for params in tqdm(grid):
        with multiprocessing.Pool(None, limit_cpu) as pool:
            results: List[Tuple[float, float]] = pool.starmap(
                train_func[args.algorithm],
                [
                    [x, y, num_classes, params, random_state]
                    for random_state in np.random.randint(int(1e9), size=20)
                ],
            )
        errors.append(np.mean(np.array(results), axis=0))
        stds.append(np.std(np.array(results), axis=0))

    rows = []
    for i, params in enumerate(grid):
        rows.append(
            {
                "param": str(params)
                if args.algorithm == "nn"
                else params["kernel_param"],
                "train error": f"{errors[i][0]:.3f}±{stds[i][0]:.3f}",
                "test error": f"{errors[i][1]:.3f}±{stds[i][1]:.3f}",
            }
        )
    df = pd.DataFrame(rows)
    print(df)
    if args.output:
        df.to_csv(args.output, index=False)


def task_two_three(args: Namespace) -> None:
    x, y, num_classes = read_data(args.data)
    if args.debug:
        x, y = x[:100], y[:100]

    train_func: Dict[str, Callable] = {
        "perceptron": train_perceptron,
        "svm": train_svm,
        "nn": train_nn,
    }
    grid = param_grid(args)

    with multiprocessing.Pool(None, limit_cpu) as pool:
        results = pool.starmap(
            run_kfold,
            [
                [train_func[args.algorithm], x, y, num_classes, grid, random_state]
                for random_state in np.random.randint(int(1e9), size=20)
            ],
        )

    params, err, cm = zip(*results)

    mean_cm = np.mean(np.array(cm), axis=0)
    std_cm = np.std(np.array(cm), axis=0)
    cm = []
    for i in range(num_classes):
        loc_cm = []
        for j in range(num_classes):
            loc_cm.append(f"{mean_cm[i, j]:.3f}±{std_cm[i, j]:.3f}")
        cm.append(loc_cm)

    if args.algorithm == "nn":
        params = [str(p) for p in params]
        param_report = max(set(params), key=params.count)
    else:
        params = [p["kernel_param"] for p in params]
        param_report = f"{np.mean(params)}±{np.std(params):.1f}"

    cm.append(
        [f"param = {param_report}", f"error = {np.mean(err):.3f}±{np.std(err):.3f}",]
    )
    df = pd.DataFrame(cm)

    print(f"param = {param_report}")
    print(f"error = {np.mean(err):.3f}±{np.std(err):.3f}")
    print(df)

    if args.output:
        df.to_csv(args.output, index=False)


def task_four(args: Namespace) -> None:
    x, y, num_classes = read_data(args.data)
    if args.debug:
        x, y = x[:100], y[:100]

    train_func: Dict[str, Callable] = {
        "perceptron": train_perceptron,
        "svm": train_svm,
        "nn": train_nn,
    }
    grid = param_grid(args)

    run_kfold(
        train_func[args.algorithm],
        x,
        y,
        num_classes,
        grid,
        random_state=0,
        run_task_four=True,
        output_path=args.output,
    )


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "task",
        choices=["task_one", "task_two_three", "task_four"],
        help="Assignment task.",
    )
    parser.add_argument("-d", "--data", help="Path to the dataset file.", required=True)
    parser.add_argument(
        "--debug", help="Will use a very small part of the dataset", action="store_true"
    )
    parser.add_argument("-o", "--output", help="Output path.")

    subparsers = parser.add_subparsers(
        dest="algorithm", help="Algorithm to use.", required=True
    )

    perc_parser = subparsers.add_parser("perceptron")
    perc_parser.add_argument(
        "multiclass", choices=["one-vs-all", "one-vs-one"], help="Multiclass method.",
    )
    perc_parser.add_argument(
        "-k",
        "--kernel",
        choices=["polynomial", "gaussian"],
        help="Kernel to use.",
        required=True,
    )

    svm_parser = subparsers.add_parser("svm")
    svm_parser.add_argument(
        "-k",
        "--kernel",
        choices=["polynomial", "gaussian"],
        help="Kernel to use.",
        required=True,
    )

    subparsers.add_parser("nn")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "task_one":
        task_one(args)
    elif args.task == "task_two_three":
        task_two_three(args)
    elif args.task == "task_four":
        task_four(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    np.random.seed(0)
    main()
