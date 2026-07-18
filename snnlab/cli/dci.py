from __future__ import annotations

import argparse
from pathlib import Path

from snnlab.architectures.dci import build_dci_model
from snnlab.configs.dci import (
    DCIConfig,
    DCIHomeostasisConfig,
    DCIPresentationConfig,
    DCISTDPConfig,
    make_dci_dynamics,
)
from snnlab.core.events import ConsoleObserver
from snnlab.data import load_mnist_dataset
from snnlab.experiments.base import create_sample_schedule
from snnlab.experiments.dci_trainer import DCITrainer, DCITrainerState
from snnlab.i18n import Translator


def build_parser() -> argparse.ArgumentParser:
    """
    Builds CLI arguments for the Stage-1 DCI experiment.

    Создаёт CLI-аргументы для Stage-1 DCI-эксперимента.
    """
    parser = argparse.ArgumentParser(description="Run the Stage-1 CPU DCI experiment")
    parser.add_argument("--train-samples", type=int, default=10)
    parser.add_argument("--locale", choices=("en", "ru"), default="en")
    parser.add_argument("--exc-method", default="explicit_euler")
    parser.add_argument("--inh-method", default="explicit_euler")
    parser.add_argument("--train-pool-size", type=int, default=1000)
    parser.add_argument("--test-pool-size", type=int, default=200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    translator = Translator(args.locale)

    cfg = DCIConfig(
        seed=52,
        exc_numerical_method=args.exc_method,
        inh_numerical_method=args.inh_method,
    )

    # EN: Reproduce the current notebook's two-stage sampling protocol:
    #     1) build a shuffled 1000/200 MNIST pool with seed 52;
    #     2) shuffle the training schedule inside that pool with seed 52.
    # RU: Воспроизводим текущий двухэтапный протокол выборки из блокнота:
    #     1) строим shuffled pool MNIST 1000/200 с seed 52;
    #     2) перемешиваем расписание обучения внутри pool снова с seed 52.
    x_train, y_train, _, _ = load_mnist_dataset(
        train_limit=args.train_pool_size,
        test_limit=args.test_pool_size,
        subset_seed=cfg.seed,
    )

    dynamics = make_dci_dynamics(
        cfg,
        target_total_inhibition=20.0,
        weight_exc_inh=0.3,
        input_gain=0.60,
    )
    model = build_dci_model(
        cfg=cfg,
        dynamics=dynamics,
        presentation_cfg=DCIPresentationConfig(),
        stdp_cfg=DCISTDPConfig(eta=0.00003),
        homeostasis_cfg=DCIHomeostasisConfig(
            target_spikes_per_sample=0.25,
            learning_rate=0.005,
            max_current=8.0,
        ),
    )
    schedule = create_sample_schedule(
        n_items=len(x_train),
        n_samples=min(args.train_samples, len(x_train)),
        seed=cfg.seed,
    )
    state = DCITrainerState(model=model, sample_indices=schedule)
    trainer = DCITrainer(
        x_train=x_train,
        y_train=y_train,
        state=state,
        run_dir=Path("runs") / "dci_mnist",
        observer=ConsoleObserver(translator, sample_every=1),
        checkpoint_every=5,
    )
    trainer.train()


if __name__ == "__main__":
    main()
