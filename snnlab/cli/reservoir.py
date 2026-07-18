from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from snnlab.architectures.reservoir import build_reservoir_network
from snnlab.configs.reservoir import ReservoirConfig
from snnlab.core.encoders import GaussianPopulationEncoder, PoissonEncoder
from snnlab.core.events import ConsoleObserver
from snnlab.core.metrics import classification_accuracy
from snnlab.data import load_iris_dataset, load_mnist_dataset
from snnlab.experiments.base import create_sample_schedule
from snnlab.experiments.reservoir_runner import ReservoirRunner, ReservoirRunnerState
from snnlab.i18n import Translator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Stage-1 CPU reservoir experiment")
    parser.add_argument("--dataset", choices=("iris", "mnist"), default="iris")
    parser.add_argument("--train-samples", type=int, default=100)
    parser.add_argument("--test-samples", type=int, default=30)
    parser.add_argument("--locale", choices=("en", "ru"), default="en")
    parser.add_argument("--method", default="semi_euler")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    translator = Translator(args.locale)

    if args.dataset == "iris":
        x_train, x_test, y_train, y_test = load_iris_dataset(seed=52)
        encoder = GaussianPopulationEncoder(n_features=4, n_per_feature=8, max_rate_hz=100.0)
        n_res = 150
        select_k = 40
        input_scale = 10.0
        recurrent_scale = 4.0
    else:
        x_train, y_train, x_test, y_test = load_mnist_dataset()
        encoder = PoissonEncoder(n_features=784, max_rate_hz=100.0)
        n_res = 800
        select_k = 400
        input_scale = 2.0
        recurrent_scale = 1.0

    cfg = ReservoirConfig(
        seed=52,
        n_input=encoder.output_size,
        n_reservoir=n_res,
        input_scale=input_scale,
        recurrent_scale=recurrent_scale,
        numerical_method=args.method,
        select_k=select_k,
    )
    network = build_reservoir_network(cfg)
    schedule = create_sample_schedule(
        n_items=len(x_train),
        n_samples=min(args.train_samples, len(x_train)),
        seed=cfg.seed,
    )
    state = ReservoirRunnerState(network=network, train_sample_indices=schedule)
    runner = ReservoirRunner(
        x_train=x_train,
        y_train=y_train,
        cfg=cfg,
        encoder=encoder,
        state=state,
        run_dir=Path("runs") / f"reservoir_{args.dataset}",
        observer=ConsoleObserver(translator, sample_every=10),
        checkpoint_every=50,
    )
    runner.collect_features()
    runner.fit_readout()

    test_count = min(args.test_samples, len(x_test))
    test_indices = np.random.default_rng(53).permutation(len(x_test))[:test_count]
    predictions = runner.predict(x_test[test_indices])
    print(f"accuracy={classification_accuracy(y_test[test_indices], predictions):.4f}")


if __name__ == "__main__":
    main()
