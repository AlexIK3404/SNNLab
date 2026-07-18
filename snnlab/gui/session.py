from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from snnlab.architectures.dci import build_dci_model
from snnlab.architectures.reservoir import build_reservoir_network
from snnlab.configs.dci import (
    DCIConfig,
    DCIHomeostasisConfig,
    DCIPresentationConfig,
    DCISTDPConfig,
    make_dci_dynamics,
)
from snnlab.configs.reservoir import ReservoirConfig
from snnlab.core.encoders import GaussianPopulationEncoder, PoissonEncoder
from snnlab.core.events import EventObserver, FrameworkEvent
from snnlab.data import load_iris_dataset, load_mnist_dataset
from snnlab.evaluation import (
    DCIEvaluationConfig,
    evaluate_dci_classifier,
    evaluate_reservoir_classifier,
)
from snnlab.experiments.base import create_epoch_schedule
from snnlab.experiments.dci_trainer import DCITrainer, DCITrainerState
from snnlab.experiments.reservoir_runner import ReservoirRunner, ReservoirRunnerState
from snnlab.runtime.checkpoint import CheckpointManager
from snnlab.runtime.control import RunControl
from snnlab.runtime.experiment_io import (
    build_data_protocol,
    build_user_configuration,
    save_json,
    update_data_protocol_evaluation,
    update_data_protocol_reservoir_evaluation,
    write_run_manifest,
)
from snnlab.runtime.model_snapshot import ModelSnapshotManager


@dataclass(slots=True)
class GuiExperimentSession:
    """
    Owns one GUI-visible experiment together with its datasets and runtime object.

    Хранит один GUI-эксперимент вместе с датасетами и runtime-объектом.
    """

    kind: str
    engine: Any
    control: RunControl
    run_dir: Path
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    data_spec: dict[str, Any] | None = None
    last_evaluation: Any | None = None
    last_evaluation_data: Any | None = None
    last_evaluation_position: int = -1

    def run(self) -> Any:
        """
        Runs or continues the architecture-specific training stage.

        Запускает или продолжает архитектурно-зависимый этап обучения.
        """
        # EN: Any training change invalidates cached evaluation responses.
        # RU: Любое изменение обучения инвалидирует кэш evaluation-откликов.
        self.last_evaluation = None
        self.last_evaluation_data = None
        self.last_evaluation_position = -1
        if self.kind == "dci":
            return self.engine.train()
        if self.kind == "reservoir":
            return self.engine.collect_features()
        raise ValueError(f"Unknown session kind: {self.kind!r}")

    def fit_readout(self) -> Any:
        """
        Fits or refits the reservoir readout on all collected features.

        Обучает или переобучает reservoir-readout на всех собранных признаках.
        """
        if self.kind != "reservoir":
            raise RuntimeError("Readout fitting is available only for reservoir sessions")
        return self.engine.fit_readout(allow_partial=True)

    def evaluate(self, params: dict[str, Any]) -> Any:
        """
        Evaluates the current usable model without mutating training state.

        Оценивает текущую используемую модель без изменения training state.
        """
        if self.x_test is None or self.y_test is None:
            raise RuntimeError("This session has no test dataset")

        def progress(position: int, total: int, phase: str) -> None:
            observer = getattr(self.engine, "observer", None)
            if observer is not None:
                observer.handle(
                    FrameworkEvent.create(
                        "evaluation_progress",
                        "gui_session",
                        {
                            "position": int(position),
                            "total": int(total),
                            "phase": str(phase),
                        },
                    )
                )

        if self.kind == "dci":
            config = DCIEvaluationConfig(
                assignment_samples=int(params.get("assignment_samples", 500)),
                test_samples=int(params.get("test_samples", 200)),
                seed=int(params.get("seed", 52)),
                assignment_policy=str(params.get("assignment_policy", "full_train_pool")),
                homeostasis_mode=str(params.get("homeostasis_mode", "frozen")),
                network_state_mode=str(params.get("network_state_mode", "fresh_continuous")),
                prediction_rule=str(params.get("prediction_rule", "mean_response")),
                min_best_response=float(params.get("min_best_response", 0.0)),
                min_absolute_margin=float(params.get("min_absolute_margin", 0.0)),
                min_relative_margin=float(params.get("min_relative_margin", 0.0)),
                top_k_per_class=int(params.get("top_k_per_class", 0)),
            )
            cached_data = None
            if (
                self.last_evaluation_data is not None
                and self.last_evaluation_position == self.position
                and self.last_evaluation_data.response_signature == config.response_signature()
            ):
                cached_data = self.last_evaluation_data

            # EN: Assignment data policy is explicit and reproducible. The
            #     test set is never used for label assignment or decoder tuning.
            # RU: Политика assignment-данных задаётся явно и воспроизводимо.
            #     Test-набор никогда не используется для назначения меток.
            training_schedule = np.asarray(self.engine.state.sample_indices, dtype=np.int64)
            unique_training = np.unique(training_schedule)
            if config.assignment_policy == "training_subset":
                assignment_source_indices = unique_training
            elif config.assignment_policy == "exclude_training":
                all_indices = np.arange(len(self.x_train), dtype=np.int64)
                assignment_source_indices = np.setdiff1d(
                    all_indices,
                    unique_training,
                    assume_unique=False,
                )
                if assignment_source_indices.size == 0:
                    raise RuntimeError(
                        "Assignment policy 'exclude_training' has no samples left in the train pool"
                    )
            elif config.assignment_policy == "full_train_pool":
                assignment_source_indices = np.arange(len(self.x_train), dtype=np.int64)
            else:
                raise ValueError(f"Unsupported assignment_policy: {config.assignment_policy!r}")

            result = evaluate_dci_classifier(
                self.engine.state.model,
                self.x_train[assignment_source_indices],
                self.y_train[assignment_source_indices],
                self.x_test,
                self.y_test,
                config=config,
                progress_callback=progress,
                response_data=cached_data,
            )
            self.last_evaluation_data = result.response_data
            self.last_evaluation_position = self.position

            # EN: Store original pool indices, not positions inside a temporary
            #     assignment subset, in the public data protocol.
            # RU: В публичном протоколе сохраняем исходные индексы pool, а не
            #     позиции внутри временной assignment-подвыборки.
            local_assignment = result.response_data.assignment_responses.sample_indices
            original_assignment = assignment_source_indices[local_assignment]
            update_data_protocol_evaluation(
                run_dir=self.run_dir,
                assignment_policy=config.assignment_policy,
                assignment_indices=original_assignment,
                test_indices=result.response_data.test_responses.sample_indices,
                training_indices=training_schedule,
            )
            save_json(
                self.run_dir / "metrics.json",
                {
                    "architecture": "dci",
                    "training_position": self.position,
                    "accuracy": float(result.accuracy),
                    "assigned_fraction": float(result.assigned_fraction),
                    "accepted_fraction_assignment": float(result.accepted_fraction_assignment),
                    "accepted_fraction_test": float(result.accepted_fraction_test),
                    "assignment_policy": str(config.assignment_policy),
                    "prediction_rule": str(config.prediction_rule),
                    "assignment_samples": int(len(original_assignment)),
                    "test_samples": int(len(result.y_true)),
                    "assigned_neurons": int(result.diagnostics.assigned_count),
                    "unassigned_neurons": int(result.diagnostics.unassigned_count),
                    "silent_neurons": int(result.diagnostics.silent_count),
                    "median_relative_margin": float(result.diagnostics.median_relative_margin),
                    "per_class_accuracy": result.diagnostics.per_class_accuracy,
                    "confusion_matrix": result.confusion_matrix,
                },
            )
        elif self.kind == "reservoir":
            if self.engine.state.readout_model is None:
                if not self.engine.state.features:
                    raise RuntimeError("Reservoir has no collected features for readout fitting")
                self.engine.fit_readout(allow_partial=True)
            result = evaluate_reservoir_classifier(
                self.engine,
                self.x_test,
                self.y_test,
                n_samples=int(params.get("test_samples", 200)),
                seed=int(params.get("seed", 52)) + 30_000,
                progress_callback=progress,
            )
            reservoir_test_indices = np.arange(len(result.y_true), dtype=np.int64)
            update_data_protocol_reservoir_evaluation(
                run_dir=self.run_dir,
                test_indices=reservoir_test_indices,
            )
            save_json(
                self.run_dir / "metrics.json",
                {
                    "architecture": "reservoir",
                    "training_position": self.position,
                    "accuracy": float(result.accuracy),
                    "test_samples": int(len(result.y_true)),
                    "confusion_matrix": result.confusion_matrix,
                },
            )
        else:
            raise ValueError(f"Unknown session kind: {self.kind!r}")

        self.last_evaluation = result
        observer = getattr(self.engine, "observer", None)
        if observer is not None:
            observer.handle(
                FrameworkEvent.create(
                    "evaluation_end",
                    "gui_session",
                    {"accuracy": float(result.accuracy), "kind": self.kind},
                )
            )
        return result

    def save_model_snapshot(self) -> Any:
        """
        Saves an inference-oriented model snapshot distinct from a checkpoint.

        Сохраняет model snapshot для inference отдельно от checkpoint.
        """
        manager = ModelSnapshotManager(self.run_dir)
        if self.kind == "dci":
            assignment = None
            if self.last_evaluation is not None and hasattr(self.last_evaluation, "assignment"):
                assignment = self.last_evaluation.assignment
            info = manager.save(
                kind="dci",
                training_position=self.position,
                payload={
                    "model": self.engine.state.model,
                    "data_spec": dict(self.data_spec or {}),
                    "label_assignment": assignment,
                },
            )
        elif self.kind == "reservoir":
            info = manager.save(
                kind="reservoir",
                training_position=self.position,
                payload={
                    "cfg": self.engine.cfg,
                    "network": self.engine.state.network,
                    "encoder": self.engine.encoder,
                    "readout_model": self.engine.state.readout_model,
                    "data_spec": dict(self.data_spec or {}),
                },
            )
        else:
            raise ValueError(f"Unknown session kind: {self.kind!r}")

        observer = getattr(self.engine, "observer", None)
        if observer is not None:
            observer.handle(
                FrameworkEvent.create(
                    "model_snapshot_saved",
                    "gui_session",
                    {"path": str(info.path), "position": self.position},
                )
            )
        return info

    def extend_training(
        self,
        *,
        additional_samples: int,
        seed: int,
        allow_repeats: bool,
    ) -> None:
        """
        Extends the current schedule without altering processed samples.

        Расширяет текущее расписание без изменения уже обработанных sample.
        """
        self.engine.extend_training(
            additional_samples=additional_samples,
            seed=seed,
            allow_repeats=allow_repeats,
        )
        self.last_evaluation = None
        self.last_evaluation_data = None
        self.last_evaluation_position = -1

    @property
    def position(self) -> int:
        return int(self.engine.state.position)

    @property
    def total_samples(self) -> int:
        return int(self.engine.total_samples)

    def training_sample(self, sample_index: int) -> np.ndarray:
        return np.asarray(self.x_train[int(sample_index)])


def _unique_run_dir(root: str | Path, kind: str) -> Path:
    """Returns a collision-free run directory even for rapid repeated starts.

    Возвращает уникальную папку даже при нескольких быстрых запусках подряд.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path(root) / f"{timestamp}_{kind}"


def build_new_session(
    params: dict[str, Any],
    *,
    observer: EventObserver,
    control: RunControl,
) -> GuiExperimentSession:
    """
    Builds a new DCI or Reservoir session from GUI parameter values.

    Создаёт новую DCI- или Reservoir-session из значений параметров GUI.
    """
    kind = str(params["architecture"])
    if kind == "dci":
        return _build_new_dci_session(params, observer=observer, control=control)
    if kind == "reservoir":
        return _build_new_reservoir_session(params, observer=observer, control=control)
    raise ValueError(f"Unsupported architecture: {kind!r}")


def _build_new_dci_session(
    params: dict[str, Any],
    *,
    observer: EventObserver,
    control: RunControl,
) -> GuiExperimentSession:
    seed = int(params["seed"])
    train_pool_size = int(params["train_pool_size"])
    test_pool_size = int(params["test_pool_size"])

    x_train, y_train, x_test, y_test = load_mnist_dataset(
        train_limit=train_pool_size,
        test_limit=test_pool_size,
        subset_seed=seed,
    )

    n_exc = int(params["n_exc"])
    cfg = DCIConfig(
        seed=seed,
        n_exc=n_exc,
        n_inh=n_exc,
        dt_ms=float(params["dt_ms"]),
        stimulus_ms=float(params["stimulus_ms"]),
        rest_ms=float(params["rest_ms"]),
        exc_a=float(params["exc_a"]),
        exc_b=float(params["exc_b"]),
        exc_c=float(params["exc_c"]),
        exc_d=float(params["exc_d"]),
        inh_a=float(params["inh_a"]),
        inh_b=float(params["inh_b"]),
        inh_c=float(params["inh_c"]),
        inh_d=float(params["inh_d"]),
        exc_numerical_method=str(params["exc_method"]),
        inh_numerical_method=str(params["inh_method"]),
    )
    dynamics = make_dci_dynamics(
        cfg,
        target_total_inhibition=float(params["target_total_inhibition"]),
        weight_exc_inh=float(params["weight_exc_inh"]),
        input_gain=float(params["input_gain"]),
    )
    model = build_dci_model(
        cfg=cfg,
        dynamics=dynamics,
        presentation_cfg=DCIPresentationConfig(
            base_max_rate_hz=float(params["base_max_rate_hz"]),
            rate_increment_hz=float(params["rate_increment_hz"]),
            min_exc_spikes=int(params["min_exc_spikes"]),
            max_attempts=int(params["max_attempts"]),
        ),
        stdp_cfg=DCISTDPConfig(
            tau_pre_ms=float(params["tau_pre_ms"]),
            eta=float(params["eta"]),
            x_target=float(params["x_target"]),
            mu=float(params["mu"]),
        ),
        homeostasis_cfg=DCIHomeostasisConfig(
            target_spikes_per_sample=float(params["target_spikes_per_sample"]),
            learning_rate=float(params["homeo_learning_rate"]),
            max_current=float(params["homeo_max_current"]),
        ),
    )
    samples_per_epoch = min(int(params["samples_per_epoch"]), len(x_train))
    n_epochs = int(params["epochs"])
    schedule = create_epoch_schedule(
        n_items=len(x_train),
        samples_per_epoch=samples_per_epoch,
        n_epochs=n_epochs,
        seed=seed,
    )
    state = DCITrainerState(model=model, sample_indices=schedule)
    run_dir = _unique_run_dir(params.get("runs_root", "runs/gui"), "dci")
    data_spec = {
        "dataset": "mnist",
        "train_limit": train_pool_size,
        "test_limit": test_pool_size,
        "train_pool_size": int(len(x_train)),
        "test_pool_size": int(len(x_test)),
        "subset_seed": seed,
        "samples_per_epoch": samples_per_epoch,
        "epochs": n_epochs,
        "checkpoint_every": int(params["checkpoint_every"]),
        "live_frame_every_steps": int(params["live_frame_every_steps"]),
    }
    trainer = DCITrainer(
        x_train=x_train,
        y_train=y_train,
        state=state,
        run_dir=run_dir,
        observer=observer,
        control=control,
        checkpoint_every=int(params["checkpoint_every"]),
        emit_spike_logs=False,
        emit_live_frames=True,
        live_frame_every_steps=int(params["live_frame_every_steps"]),
        emit_learning_snapshots=True,
        learning_snapshot_every_samples=10,
        data_spec=data_spec,
    )
    user_configuration = build_user_configuration(
        architecture="dci",
        task=str(params.get("task", "classification")),
        backend=str(params.get("backend", "python_cpu")),
        locale=str(params.get("locale", "ru")),
        parameters={key: value for key, value in params.items() if key != "runs_root"},
        evaluation=dict(params.get("evaluation_defaults") or {}),
    )
    write_run_manifest(
        run_dir=run_dir,
        user_configuration=user_configuration,
        data_protocol=build_data_protocol(
            kind="dci",
            data_spec=data_spec,
            schedule=schedule,
        ),
    )
    return GuiExperimentSession(
        kind="dci",
        engine=trainer,
        control=control,
        run_dir=run_dir,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        data_spec=data_spec,
    )


def _build_new_reservoir_session(
    params: dict[str, Any],
    *,
    observer: EventObserver,
    control: RunControl,
) -> GuiExperimentSession:
    seed = int(params["seed"])
    dataset = str(params["dataset"])

    if dataset == "iris":
        x_train, x_test, y_train, y_test = load_iris_dataset(seed=seed)
        encoder = GaussianPopulationEncoder(
            n_features=4,
            n_per_feature=int(params["iris_neurons_per_feature"]),
            max_rate_hz=float(params["max_rate_hz"]),
        )
    elif dataset == "mnist":
        x_train, y_train, x_test, y_test = load_mnist_dataset(
            train_limit=int(params["train_pool_size"]),
            test_limit=int(params["test_pool_size"]),
            subset_seed=seed,
        )
        encoder = PoissonEncoder(
            n_features=784,
            max_rate_hz=float(params["max_rate_hz"]),
        )
    else:
        raise ValueError(f"Unsupported reservoir dataset: {dataset!r}")

    cfg = ReservoirConfig(
        seed=seed,
        n_input=encoder.output_size,
        n_reservoir=int(params["n_reservoir"]),
        dt_ms=float(params["dt_ms"]),
        simulation_ms=float(params["simulation_ms"]),
        max_rate_hz=float(params["max_rate_hz"]),
        input_density=float(params["input_density"]),
        recurrent_density=float(params["recurrent_density"]),
        excitatory_ratio=float(params["excitatory_ratio"]),
        input_scale=float(params["input_scale"]),
        recurrent_scale=float(params["recurrent_scale"]),
        bias_current=float(params["bias_current"]),
        tau_syn_ms=float(params["tau_syn_ms"]),
        neuron_a=float(params["neuron_a"]),
        neuron_b=float(params["neuron_b"]),
        neuron_c=float(params["neuron_c"]),
        neuron_d=float(params["neuron_d"]),
        numerical_method=str(params["numerical_method"]),
        readout=str(params["readout"]),
        use_feature_selection=bool(params["use_feature_selection"]),
        select_k=int(params["select_k"]),
    )
    network = build_reservoir_network(cfg)
    samples_per_epoch = min(int(params["samples_per_epoch"]), len(x_train))
    n_epochs = int(params["epochs"])
    schedule = create_epoch_schedule(
        n_items=len(x_train),
        samples_per_epoch=samples_per_epoch,
        n_epochs=n_epochs,
        seed=seed,
    )
    state = ReservoirRunnerState(network=network, train_sample_indices=schedule)
    run_dir = _unique_run_dir(params.get("runs_root", "runs/gui"), "reservoir")
    data_spec = {
        "dataset": dataset,
        "seed": seed,
        "subset_seed": seed,
        "train_limit": int(params["train_pool_size"]),
        "test_limit": int(params["test_pool_size"]),
        "train_pool_size": int(len(x_train)),
        "test_pool_size": int(len(x_test)),
        "iris_neurons_per_feature": int(params["iris_neurons_per_feature"]),
        "max_rate_hz": float(params["max_rate_hz"]),
        "samples_per_epoch": samples_per_epoch,
        "epochs": n_epochs,
        "checkpoint_every": int(params["checkpoint_every"]),
        "live_frame_every_steps": int(params["live_frame_every_steps"]),
    }
    runner = ReservoirRunner(
        x_train=x_train,
        y_train=y_train,
        cfg=cfg,
        encoder=encoder,
        state=state,
        run_dir=run_dir,
        observer=observer,
        control=control,
        checkpoint_every=int(params["checkpoint_every"]),
        # EN: The reservoir raster is rendered from the complete current-sample
        #     spike log. The array is emitted and discarded, not accumulated in
        #     the session, so memory use stays bounded.
        # RU: Raster резервуара строится по полному spike log текущего sample.
        #     Массив отправляется в GUI и сразу освобождается, а не копится в session.
        emit_spike_logs=True,
        emit_live_frames=True,
        live_frame_every_steps=int(params["live_frame_every_steps"]),
        data_spec=data_spec,
    )
    user_configuration = build_user_configuration(
        architecture="reservoir",
        task=str(params.get("task", "classification")),
        backend=str(params.get("backend", "python_cpu")),
        locale=str(params.get("locale", "ru")),
        parameters={key: value for key, value in params.items() if key != "runs_root"},
        evaluation=dict(params.get("evaluation_defaults") or {}),
    )
    write_run_manifest(
        run_dir=run_dir,
        user_configuration=user_configuration,
        data_protocol=build_data_protocol(
            kind="reservoir",
            data_spec=data_spec,
            schedule=schedule,
        ),
    )
    return GuiExperimentSession(
        kind="reservoir",
        engine=runner,
        control=control,
        run_dir=run_dir,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        data_spec=data_spec,
    )


def load_session_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    observer: EventObserver,
    control: RunControl,
) -> GuiExperimentSession:
    """
    Loads a checkpoint and reconstructs the dataset/encoder needed for continuation.

    Загружает checkpoint и восстанавливает датасет/encoder для продолжения.
    """
    checkpoint_path = Path(checkpoint_path)
    manager = CheckpointManager(checkpoint_path.parent.parent)
    payload = manager.load(checkpoint_path)
    kind = str(payload.get("kind", ""))
    data_spec = dict(payload.get("data_spec") or {})

    if kind == "dci":
        state = payload["state"]
        seed = int(state.model.cfg.seed)
        if not data_spec:
            data_spec = {
                "dataset": "mnist",
                "train_limit": 1000,
                "test_limit": 200,
                "subset_seed": seed,
            }
        x_train, y_train, x_test, y_test = load_mnist_dataset(
            train_limit=data_spec.get("train_limit"),
            test_limit=data_spec.get("test_limit"),
            subset_seed=data_spec.get("subset_seed", seed),
        )
        trainer = DCITrainer(
            x_train=x_train,
            y_train=y_train,
            state=state,
            run_dir=checkpoint_path.parent.parent,
            observer=observer,
            control=control,
            checkpoint_every=int(data_spec.get("checkpoint_every", 50)),
            emit_live_frames=True,
            live_frame_every_steps=int(data_spec.get("live_frame_every_steps", 50)),
            emit_learning_snapshots=True,
            learning_snapshot_every_samples=10,
            data_spec=data_spec,
        )
        return GuiExperimentSession(
            kind="dci",
            engine=trainer,
            control=control,
            run_dir=checkpoint_path.parent.parent,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            data_spec=data_spec,
        )

    if kind == "reservoir":
        cfg = payload["cfg"]
        state = payload["state"]
        dataset = str(data_spec.get("dataset") or ("mnist" if cfg.n_input == 784 else "iris"))
        if dataset == "mnist":
            x_train, y_train, x_test, y_test = load_mnist_dataset(
                train_limit=data_spec.get("train_limit"),
                test_limit=data_spec.get("test_limit"),
                subset_seed=data_spec.get("seed", cfg.seed),
            )
            encoder = PoissonEncoder(
                n_features=784,
                max_rate_hz=float(data_spec.get("max_rate_hz", cfg.max_rate_hz)),
            )
        else:
            x_train, x_test, y_train, y_test = load_iris_dataset(
                seed=int(data_spec.get("seed", cfg.seed))
            )
            encoder = GaussianPopulationEncoder(
                n_features=4,
                n_per_feature=int(
                    data_spec.get("iris_neurons_per_feature", max(1, cfg.n_input // 4))
                ),
                max_rate_hz=float(data_spec.get("max_rate_hz", cfg.max_rate_hz)),
            )
        runner = ReservoirRunner(
            x_train=x_train,
            y_train=y_train,
            cfg=cfg,
            encoder=encoder,
            state=state,
            run_dir=checkpoint_path.parent.parent,
            observer=observer,
            control=control,
            checkpoint_every=int(data_spec.get("checkpoint_every", 100)),
            emit_spike_logs=True,
            emit_live_frames=True,
            live_frame_every_steps=int(data_spec.get("live_frame_every_steps", 10)),
            data_spec=data_spec,
        )
        return GuiExperimentSession(
            kind="reservoir",
            engine=runner,
            control=control,
            run_dir=checkpoint_path.parent.parent,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            data_spec=data_spec,
        )

    raise ValueError(f"Unsupported checkpoint kind: {kind!r}")


def session_parameter_snapshot(session: GuiExperimentSession) -> dict[str, Any]:
    """
    Returns GUI parameter values represented by an existing session.

    Возвращает значения GUI-параметров, представленные существующей session.
    """
    if session.kind == "dci":
        trainer = session.engine
        model = trainer.state.model
        cfg = model.cfg
        dyn = model.dynamics
        p = model.presentation_cfg
        stdp = model.stdp_cfg
        homeo = model.homeostasis_cfg
        spec = session.data_spec or {}
        return {
            "seed": cfg.seed,
            "samples_per_epoch": int(spec.get("samples_per_epoch", session.total_samples)),
            "epochs": int(spec.get("epochs", 1)),
            "checkpoint_every": trainer.checkpoint_every,
            "live_frame_every_steps": trainer.live_frame_every_steps,
            "train_pool_size": spec.get("train_limit", len(session.x_train)),
            "test_pool_size": spec.get(
                "test_limit", len(session.x_test) if session.x_test is not None else 200
            ),
            "n_exc": cfg.n_exc,
            "dt_ms": cfg.dt_ms,
            "stimulus_ms": cfg.stimulus_ms,
            "rest_ms": cfg.rest_ms,
            "exc_method": cfg.exc_numerical_method,
            "inh_method": cfg.inh_numerical_method,
            "input_gain": dyn.input_gain,
            "weight_exc_inh": dyn.weight_exc_inh,
            "target_total_inhibition": dyn.weight_inh_exc * (cfg.n_exc - 1),
            "base_max_rate_hz": p.base_max_rate_hz,
            "rate_increment_hz": p.rate_increment_hz,
            "min_exc_spikes": p.min_exc_spikes,
            "max_attempts": p.max_attempts,
            "tau_pre_ms": stdp.tau_pre_ms,
            "eta": stdp.eta,
            "x_target": stdp.x_target,
            "mu": stdp.mu,
            "target_spikes_per_sample": homeo.target_spikes_per_sample,
            "homeo_learning_rate": homeo.learning_rate,
            "homeo_max_current": homeo.max_current,
            "exc_a": cfg.exc_a,
            "exc_b": cfg.exc_b,
            "exc_c": cfg.exc_c,
            "exc_d": cfg.exc_d,
            "inh_a": cfg.inh_a,
            "inh_b": cfg.inh_b,
            "inh_c": cfg.inh_c,
            "inh_d": cfg.inh_d,
        }

    runner = session.engine
    cfg = runner.cfg
    spec = session.data_spec or {}
    return {
        "seed": cfg.seed,
        "samples_per_epoch": int(spec.get("samples_per_epoch", session.total_samples)),
        "epochs": int(spec.get("epochs", 1)),
        "checkpoint_every": runner.checkpoint_every,
        "live_frame_every_steps": runner.live_frame_every_steps,
        "train_pool_size": spec.get("train_limit", len(session.x_train)),
        "test_pool_size": spec.get(
            "test_limit", len(session.x_test) if session.x_test is not None else 200
        ),
        "dataset": spec.get("dataset", "mnist" if cfg.n_input == 784 else "iris"),
        "iris_neurons_per_feature": spec.get("iris_neurons_per_feature", max(1, cfg.n_input // 4)),
        "n_reservoir": cfg.n_reservoir,
        "dt_ms": cfg.dt_ms,
        "simulation_ms": cfg.simulation_ms,
        "max_rate_hz": cfg.max_rate_hz,
        "numerical_method": cfg.numerical_method,
        "input_density": cfg.input_density,
        "recurrent_density": cfg.recurrent_density,
        "excitatory_ratio": cfg.excitatory_ratio,
        "input_scale": cfg.input_scale,
        "recurrent_scale": cfg.recurrent_scale,
        "bias_current": cfg.bias_current,
        "tau_syn_ms": cfg.tau_syn_ms,
        "readout": cfg.readout,
        "use_feature_selection": cfg.use_feature_selection,
        "select_k": cfg.select_k,
        "neuron_a": cfg.neuron_a,
        "neuron_b": cfg.neuron_b,
        "neuron_c": cfg.neuron_c,
        "neuron_d": cfg.neuron_d,
    }
