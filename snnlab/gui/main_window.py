from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QEvent, QSettings, Qt, QThread, QTimer
from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QToolBar,
    QToolButton,
)

from snnlab.core.events import FrameworkEvent
from snnlab.gui.event_bridge import QtEventObserver
from snnlab.gui.session import (
    GuiExperimentSession,
    build_new_session,
    load_session_from_checkpoint,
    session_parameter_snapshot,
)
from snnlab.gui.widgets.evaluation_view import EvaluationView
from snnlab.gui.widgets.live_view import LiveExperimentView
from snnlab.gui.widgets.metrics_view import MetricsView
from snnlab.gui.widgets.network_view import NetworkView
from snnlab.gui.widgets.parameter_panel import ParameterPanel
from snnlab.gui.widgets.training_view import TrainingView
from snnlab.gui.workers import ExperimentWorker
from snnlab.i18n import Translator
from snnlab.runtime.control import RunControl
from snnlab.runtime.experiment_io import (
    append_run_log,
    build_user_configuration,
    diagnostic_report,
    load_yaml,
    save_json,
    save_yaml,
    validate_user_configuration,
)


class MainWindow(QMainWindow):
    """
    Main desktop window for interactive SNN experiments.

    Главное desktop-окно для интерактивных SNN-экспериментов.
    """

    def __init__(self) -> None:
        super().__init__()
        self.translator = Translator("ru")
        self.current_session: GuiExperimentSession | None = None
        self.current_control: RunControl | None = None
        self._thread: QThread | None = None
        self._worker: ExperimentWorker | None = None
        self._status_key = "gui.status.idle"
        self._last_input_index: int | None = None
        self._close_when_finished = False
        self._current_operation: str | None = None
        self._last_traceback: str | None = None
        self._settings = QSettings("SNNLab", "SNNLab")
        self._layout_refresh_timer = QTimer(self)
        self._layout_refresh_timer.setSingleShot(True)
        self._layout_refresh_timer.timeout.connect(self._refresh_current_tab_layout)
        self._layout_refresh_delayed_timer = QTimer(self)
        self._layout_refresh_delayed_timer.setSingleShot(True)
        self._layout_refresh_delayed_timer.timeout.connect(self._refresh_current_tab_layout)

        self.event_bridge = QtEventObserver()
        self.event_bridge.event_received.connect(self._handle_framework_event)

        self._build_toolbar()
        self._build_parameter_dock()
        self._build_central_tabs()
        self._build_status_bar()
        self._connect_actions()

        self.parameter_panel.set_architecture("dci")
        self._set_run_state("idle")
        self.retranslate_ui()
        self.fit_readout_button.setVisible(False)
        self._apply_default_window_geometry()
        self.restore_layout_state()
        self._schedule_visible_tab_refresh()

    def _build_toolbar(self) -> None:
        """Builds two compact toolbar rows that can overflow safely.

        A single giant QWidget inside one toolbar forces the main window to
        inherit the sum of every control's minimum width. On smaller displays
        that makes the normal window wider than the screen. Separate Qt
        toolbars keep selector and action controls compact and let Qt move
        excess widgets into the native overflow menu.

        Создаёт две компактные строки toolbar с безопасным overflow.

        Один огромный QWidget внутри toolbar заставляет главное окно наследовать
        сумму минимальных ширин всех элементов. На небольших экранах обычное
        окно из-за этого становилось шире монитора. Раздельные Qt-toolbar
        позволяют Qt переносить лишние элементы в стандартное overflow-меню.
        """
        self.selector_toolbar = QToolBar()
        self.selector_toolbar.setObjectName("selector_toolbar")
        self.selector_toolbar.setMovable(False)
        self.action_toolbar = QToolBar()
        self.action_toolbar.setObjectName("action_toolbar")
        self.action_toolbar.setMovable(False)
        self.addToolBar(self.selector_toolbar)
        self.addToolBarBreak()
        self.addToolBar(self.action_toolbar)

        self.architecture_label = QLabel()
        self.architecture_combo = QComboBox()
        self.task_label = QLabel()
        self.task_combo = QComboBox()
        self.backend_label = QLabel()
        self.backend_combo = QComboBox()
        self.language_label = QLabel()
        self.language_combo = QComboBox()
        self.language_combo.addItem("RU", "ru")
        self.language_combo.addItem("EN", "en")
        self.help_mode_checkbox = QCheckBox()

        selector_groups = (
            (self.architecture_label, self.architecture_combo),
            (self.task_label, self.task_combo),
            (self.backend_label, self.backend_combo),
            (self.language_label, self.language_combo),
        )
        for group_index, (label, editor) in enumerate(selector_groups):
            if group_index:
                self.selector_toolbar.addSeparator()
            self.selector_toolbar.addWidget(label)
            self.selector_toolbar.addWidget(editor)
        self.selector_toolbar.addSeparator()
        self.selector_toolbar.addWidget(self.help_mode_checkbox)

        self.start_button = QPushButton()
        self.pause_button = QPushButton()
        self.resume_button = QPushButton()
        self.stop_button = QPushButton()
        self.continue_button = QPushButton()
        self.load_button = QPushButton()
        self.fit_readout_button = QPushButton()
        self.evaluate_button = QPushButton()
        self.save_model_button = QPushButton()
        self.reset_layout_button = QPushButton()
        self.config_button = QToolButton()
        self.config_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.config_menu = QMenu(self.config_button)
        self.export_config_action = QAction(self)
        self.import_config_action = QAction(self)
        self.copy_config_action = QAction(self)
        self.config_menu.addAction(self.export_config_action)
        self.config_menu.addAction(self.import_config_action)
        self.config_menu.addAction(self.copy_config_action)
        self.config_menu.addSeparator()
        self.copy_diagnostics_action = QAction(self)
        self.save_diagnostics_action = QAction(self)
        self.config_menu.addAction(self.copy_diagnostics_action)
        self.config_menu.addAction(self.save_diagnostics_action)
        self.config_button.setMenu(self.config_menu)

        for index, widget in enumerate(
            (
                self.start_button,
                self.pause_button,
                self.resume_button,
                self.stop_button,
                self.continue_button,
                self.load_button,
                self.fit_readout_button,
                self.evaluate_button,
                self.save_model_button,
                self.reset_layout_button,
                self.config_button,
            )
        ):
            if index in (4, 6, 7, 9):
                self.action_toolbar.addSeparator()
            self.action_toolbar.addWidget(widget)

    def _build_parameter_dock(self) -> None:
        self.parameter_panel = ParameterPanel(self.translator)
        self.parameter_dock = QDockWidget()
        self.parameter_dock.setObjectName("parameter_dock")
        self.parameter_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.parameter_dock.setWidget(self.parameter_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.parameter_dock)

    def _build_central_tabs(self) -> None:
        self.tabs = QTabWidget()
        self.live_view = LiveExperimentView()
        self.network_view = NetworkView()
        self.training_view = TrainingView()
        self.evaluation_view = EvaluationView()
        self.metrics_view = MetricsView()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)

        self.tabs.addTab(self.live_view, "")
        self.tabs.addTab(self.network_view, "")
        self.tabs.addTab(self.training_view, "")
        self.tabs.addTab(self.evaluation_view, "")
        self.tabs.addTab(self.metrics_view, "")
        self.tabs.addTab(self.log_view, "")
        self.setCentralWidget(self.tabs)

    def _build_status_bar(self) -> None:
        self.status_label = QLabel()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setMaximumWidth(300)
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress)

    def _connect_actions(self) -> None:
        self.architecture_combo.currentIndexChanged.connect(self._architecture_changed)
        self.language_combo.currentIndexChanged.connect(self._language_changed)
        self.help_mode_checkbox.toggled.connect(self._set_help_mode)
        self.start_button.clicked.connect(self.start_new_run)
        self.pause_button.clicked.connect(self.pause_run)
        self.resume_button.clicked.connect(self.resume_run)
        self.stop_button.clicked.connect(self.stop_run)
        self.continue_button.clicked.connect(self.continue_training)
        self.load_button.clicked.connect(self.load_checkpoint)
        self.fit_readout_button.clicked.connect(self.fit_readout)
        self.evaluate_button.clicked.connect(self.evaluate_model)
        self.save_model_button.clicked.connect(self.save_model_snapshot)
        self.reset_layout_button.clicked.connect(self.reset_layout)
        self.export_config_action.triggered.connect(self.export_configuration)
        self.import_config_action.triggered.connect(self.import_configuration)
        self.copy_config_action.triggered.connect(self.copy_configuration)
        self.copy_diagnostics_action.triggered.connect(self.copy_diagnostic_report)
        self.save_diagnostics_action.triggered.connect(self.save_diagnostic_report)
        self.tabs.currentChanged.connect(lambda *_: self._schedule_visible_tab_refresh())

    def _set_help_mode(self, enabled: bool) -> None:
        """Applies learning/help mode to all GUI panels.

        Включает/выключает режим изучения во всех панелях интерфейса.
        """
        value = bool(enabled)
        self.parameter_panel.set_help_mode(value)
        for view in (self.live_view, self.training_view, self.evaluation_view):
            if hasattr(view, "set_help_mode"):
                view.set_help_mode(value)

    def retranslate_ui(self) -> None:
        self.setWindowTitle(self.translator.tr("gui.window_title"))
        self.architecture_label.setText(self.translator.tr("gui.architecture"))
        self.task_label.setText(self.translator.tr("gui.task"))
        self.backend_label.setText(self.translator.tr("gui.backend"))
        self.language_label.setText(self.translator.tr("app.language"))
        self.help_mode_checkbox.setText(self.translator.tr("gui.help_mode"))

        self.start_button.setText(self.translator.tr("gui.buttons.start"))
        self.pause_button.setText(self.translator.tr("gui.buttons.pause"))
        self.resume_button.setText(self.translator.tr("gui.buttons.resume_paused"))
        self.stop_button.setText(self.translator.tr("gui.buttons.stop"))
        self._refresh_continue_button_text()
        self.load_button.setText(self.translator.tr("gui.buttons.load_checkpoint"))
        self.fit_readout_button.setText(self.translator.tr("gui.buttons.fit_readout"))
        self.evaluate_button.setText(self.translator.tr("gui.buttons.evaluate"))
        self.save_model_button.setText(self.translator.tr("gui.buttons.save_model"))
        self.reset_layout_button.setText(self.translator.tr("gui.buttons.reset_layout"))
        self.config_button.setText(self.translator.tr("gui.buttons.config_menu"))
        self.export_config_action.setText(self.translator.tr("gui.buttons.export_config"))
        self.import_config_action.setText(self.translator.tr("gui.buttons.import_config"))
        self.copy_config_action.setText(self.translator.tr("gui.buttons.copy_config"))
        self.copy_diagnostics_action.setText(self.translator.tr("gui.buttons.copy_diagnostics"))
        self.save_diagnostics_action.setText(self.translator.tr("gui.buttons.save_diagnostics"))

        self._repopulate_combo(
            self.architecture_combo,
            (
                ("dci", "components.architectures.dci"),
                ("reservoir", "components.architectures.reservoir"),
            ),
        )
        self._repopulate_combo(
            self.task_combo, (("classification", "components.tasks.classification"),)
        )
        self._repopulate_combo(
            self.backend_combo, (("python_cpu", "components.backends.python_cpu"),)
        )

        self.parameter_dock.setWindowTitle(self.translator.tr("gui.parameters"))
        self.tabs.setTabText(0, self.translator.tr("gui.tabs.live"))
        self.tabs.setTabText(1, self.translator.tr("gui.tabs.network"))
        self._refresh_training_tab_text()
        self.tabs.setTabText(3, self.translator.tr("gui.tabs.evaluation"))
        self.tabs.setTabText(4, self.translator.tr("gui.tabs.metrics"))
        self.tabs.setTabText(5, self.translator.tr("gui.tabs.logs"))
        self.parameter_panel.retranslate()
        self.live_view.retranslate(self.translator)
        self.metrics_view.retranslate(self.translator)
        self.network_view.retranslate(self.translator)
        self.training_view.retranslate(self.translator)
        self.evaluation_view.retranslate(self.translator)
        self._set_help_mode(self.help_mode_checkbox.isChecked())
        self._refresh_status_text()

    def start_new_run(self) -> None:
        if self._worker_is_active():
            return

        params = self.parameter_panel.values()
        params["runs_root"] = "runs/gui"
        params["task"] = str(self.task_combo.currentData() or "classification")
        params["backend"] = str(self.backend_combo.currentData() or "python_cpu")
        params["locale"] = str(self.language_combo.currentData() or "ru")
        params["evaluation_defaults"] = self.evaluation_view.parameters()
        control = RunControl()
        self.current_control = control
        self.current_session = None
        self._last_input_index = None
        self.live_view.reset()
        self.live_view.set_architecture(str(params["architecture"]))
        self.training_view.reset()
        self.training_view.set_architecture(str(params["architecture"]))
        self._refresh_training_tab_text()
        self.metrics_view.reset()
        self.evaluation_view.reset()
        self.evaluation_view.set_architecture(str(params["architecture"]))
        self.log_view.clear()
        self.progress.setValue(0)

        def factory() -> GuiExperimentSession:
            return build_new_session(
                params,
                observer=self.event_bridge,
                control=control,
            )

        self._set_run_state("running")
        self._start_worker(session_factory=factory, operation="run")

    def pause_run(self) -> None:
        if self.current_control is not None:
            self.current_control.request_pause()
            self._append_log(self.translator.tr("gui.log.pause_requested"))

    def resume_run(self) -> None:
        if self.current_control is not None:
            self.current_control.resume()

    def stop_run(self) -> None:
        if self.current_control is not None:
            self.current_control.request_stop()
            self._append_log(self.translator.tr("gui.log.stop_requested"))

    def _apply_runtime_settings_to_loaded_session(self) -> None:
        """Applies safe runtime-only GUI settings before continuation.

        Применяет безопасные runtime-настройки GUI перед продолжением.
        """
        if self.current_session is None:
            return
        params = self.parameter_panel.values()
        engine = self.current_session.engine
        if hasattr(engine, "checkpoint_every"):
            engine.checkpoint_every = max(0, int(params["checkpoint_every"]))
        if hasattr(engine, "live_frame_every_steps"):
            engine.live_frame_every_steps = max(1, int(params["live_frame_every_steps"]))
        if self.current_session.data_spec is not None:
            self.current_session.data_spec["checkpoint_every"] = int(params["checkpoint_every"])
            self.current_session.data_spec["live_frame_every_steps"] = int(
                params["live_frame_every_steps"]
            )
        if hasattr(engine, "data_spec"):
            engine.data_spec = dict(self.current_session.data_spec or {})

    def continue_training(self) -> None:
        """Continues an unfinished schedule or extends a completed run.

        Продолжает незавершённое расписание либо расширяет завершённый запуск.
        """
        if self.current_session is None or self._worker_is_active():
            return

        # EN: A checkpoint may already contain an unfinished schedule. In that
        #     case, run the remaining samples exactly as stored instead of
        #     appending a new schedule. This preserves the original sample
        #     order, RNG trajectory, epoch structure, and total sample count.
        # RU: Checkpoint может содержать незавершённое расписание. В этом
        #     случае выполняем оставшиеся sample точно в сохранённом порядке,
        #     не добавляя новое расписание. Так сохраняются исходный порядок,
        #     RNG-траектория, структура эпох и общее число sample.
        if self.current_session.position < self.current_session.total_samples:
            self._apply_runtime_settings_to_loaded_session()
            self.current_session.control.reset()
            self.current_control = self.current_session.control
            self._set_run_state("running")
            self._start_worker(session=self.current_session, operation="run")
            return

        # EN: Only a fully completed schedule should ask how many new samples
        #     must be appended. This is continued training, not checkpoint resume.
        # RU: Только после полного завершения расписания спрашиваем, сколько
        #     новых sample добавить. Это дообучение, а не resume checkpoint-а.
        additional, ok = QInputDialog.getInt(
            self,
            self.translator.tr("gui.dialogs.continue_title"),
            self.translator.tr("gui.dialogs.additional_samples"),
            100,
            1,
            1_000_000,
        )
        if not ok:
            return

        repeat_answer = QMessageBox.question(
            self,
            self.translator.tr("gui.dialogs.repeats_title"),
            self.translator.tr("gui.dialogs.allow_repeats"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        allow_repeats = repeat_answer == QMessageBox.StandardButton.Yes
        seed = 52 + self.current_session.total_samples

        self._apply_runtime_settings_to_loaded_session()
        self.current_session.control.reset()
        self.current_session.extend_training(
            additional_samples=additional,
            seed=seed,
            allow_repeats=allow_repeats,
        )
        self.current_control = self.current_session.control
        self._set_run_state("running")
        self._start_worker(session=self.current_session, operation="run")

    def load_checkpoint(self) -> None:
        if self._worker_is_active():
            return
        answer = QMessageBox.warning(
            self,
            self.translator.tr("gui.dialogs.untrusted_checkpoint_title"),
            self.translator.tr("gui.dialogs.untrusted_checkpoint_text"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.translator.tr("gui.dialogs.load_checkpoint"),
            str(Path("runs").resolve()),
            "Checkpoint (*.pkl)",
        )
        if not path:
            return

        control = RunControl()
        self.current_control = control

        def factory() -> GuiExperimentSession:
            return load_session_from_checkpoint(
                path,
                observer=self.event_bridge,
                control=control,
            )

        self._set_run_state("loading")
        self._start_worker(session_factory=factory, operation="load")

    def fit_readout(self) -> None:
        if (
            self.current_session is None
            or self.current_session.kind != "reservoir"
            or self._worker_is_active()
        ):
            return
        self._set_run_state("running")
        self._start_worker(session=self.current_session, operation="fit_readout")

    def evaluate_model(self) -> None:
        """Runs architecture-specific evaluation in the worker thread.

        Запускает архитектурно-зависимую evaluation в worker-потоке.
        """
        if self.current_session is None or self._worker_is_active():
            return
        self.progress.setValue(0)
        self._set_run_state("evaluating")
        self._start_worker(
            session=self.current_session,
            operation="evaluate",
            operation_args=self.evaluation_view.parameters(),
        )

    def save_model_snapshot(self) -> None:
        """Saves a model snapshot without training schedule/history.

        Сохраняет model snapshot без training schedule/history.
        """
        if self.current_session is None or self._worker_is_active():
            return
        self._set_run_state("saving_model")
        self._start_worker(session=self.current_session, operation="save_model")

    def _start_worker(
        self,
        *,
        session_factory=None,
        session: GuiExperimentSession | None = None,
        operation: str,
        operation_args: dict[str, Any] | None = None,
    ) -> None:
        thread = QThread(self)
        worker = ExperimentWorker(
            session_factory=session_factory,
            session=session,
            operation=operation,
            operation_args=operation_args,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.session_ready.connect(self._on_session_ready)
        worker.result_ready.connect(self._on_worker_result)
        worker.failed.connect(self._on_worker_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)

        self._thread = thread
        self._worker = worker
        self._current_operation = operation
        thread.start()
        self._update_button_states()

    def _on_session_ready(self, session: GuiExperimentSession) -> None:
        self.current_session = session
        self.current_control = session.control
        self._set_combo_data(self.architecture_combo, session.kind)
        self.live_view.update_from_session(session)
        self.network_view.set_architecture(session.kind)
        self.training_view.update_from_session(session)
        self._refresh_training_tab_text()
        self.evaluation_view.set_architecture(session.kind)
        self.parameter_panel.set_values(session_parameter_snapshot(session))
        self.progress.setMaximum(max(1, session.total_samples))
        self.progress.setValue(session.position)
        self.fit_readout_button.setVisible(session.kind == "reservoir")
        self._append_log(
            self.translator.tr(
                "gui.log.session_ready",
                kind=session.kind,
                run_dir=str(session.run_dir),
            )
        )

    def _on_worker_result(self, result: Any) -> None:
        operation = self._current_operation
        if operation == "evaluate":
            self.evaluation_view.display_result(result)
            self.tabs.setCurrentWidget(self.evaluation_view)
            self._set_run_state("evaluated")
        elif operation == "fit_readout":
            if self.current_session is not None:
                self.training_view.update_from_session(self.current_session)
            self._set_run_state("readout_fitted")
        elif operation == "save_model":
            self._set_run_state("model_saved")
        elif self._status_key == "gui.status.loading":
            self._set_run_state("loaded")

    def _on_worker_failed(self, traceback_text: str) -> None:
        self._last_traceback = str(traceback_text)
        self._append_log(traceback_text)
        self._set_run_state("error")
        QMessageBox.critical(
            self,
            self.translator.tr("gui.dialogs.error_title"),
            self.translator.tr("gui.dialogs.worker_failed"),
        )

    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._current_operation = None
        if self._status_key == "gui.status.running":
            self._set_run_state("idle")
        self._update_button_states()
        if self._close_when_finished:
            self._close_when_finished = False
            self.close()

    def _handle_framework_event(self, event: FrameworkEvent) -> None:
        payload = event.payload
        if event.name != "simulation_frame":
            formatted = self._format_event(event)
            if formatted:
                self._append_log(formatted)

        if event.name == "simulation_frame":
            sample_index = payload.get("sample_index")
            if sample_index is not None and self.current_session is not None:
                sample_index = int(sample_index)
                if sample_index != self._last_input_index:
                    self.live_view.set_input_sample(
                        self.current_session.training_sample(sample_index)
                    )
                    self._last_input_index = sample_index
            self.live_view.handle_simulation_frame(payload)
            return

        if event.name == "sample_end":
            self.live_view.handle_sample_end(payload)
            self.training_view.handle_sample_end(payload)
            self.metrics_view.append_sample(payload)
            position = int(payload.get("position", 0))
            total = int(payload.get("total_samples", max(1, position)))
            self.progress.setMaximum(max(1, total))
            self.progress.setValue(position)
            return

        if event.name == "learning_snapshot":
            self.training_view.handle_learning_snapshot(payload)
            return

        if event.name == "evaluation_progress":
            position = int(payload.get("position", 0))
            total = max(1, int(payload.get("total", 1)))
            self.progress.setMaximum(total)
            self.progress.setValue(position)
            return

        if event.name == "paused":
            self._set_run_state("paused")
        elif event.name == "resumed":
            self._set_run_state("running")
        elif event.name == "stopped":
            self._set_run_state("stopped")
        elif event.name == "run_end":
            self._set_run_state("completed")
            if self.current_session is not None:
                self.training_view.update_from_session(self.current_session)
        elif event.name == "run_start":
            self._set_run_state("running")

    def _architecture_changed(self) -> None:
        architecture = self.architecture_combo.currentData()
        if architecture:
            self.parameter_panel.set_architecture(str(architecture))
            self.live_view.set_architecture(str(architecture))
            self.network_view.set_architecture(str(architecture))
            self.training_view.set_architecture(str(architecture))
            self._refresh_training_tab_text()
            self.evaluation_view.set_architecture(str(architecture))
            self.network_view.retranslate(self.translator)
            self.training_view.retranslate(self.translator)
            self.fit_readout_button.setVisible(str(architecture) == "reservoir")

    def _refresh_training_tab_text(self) -> None:
        """Uses an architecture-specific title for the learning workspace.

        Использует архитектурно-зависимую подпись рабочей области обучения.
        """
        architecture = self.architecture_combo.currentData()
        if architecture is None and self.current_session is not None:
            architecture = self.current_session.kind
        key = (
            "gui.tabs.training_reservoir"
            if str(architecture) == "reservoir"
            else "gui.tabs.training"
        )
        if self.tabs.count() > 2:
            self.tabs.setTabText(2, self.translator.tr(key))

    def _language_changed(self) -> None:
        locale = self.language_combo.currentData()
        if locale:
            self.translator.set_locale(str(locale))
            self.retranslate_ui()

    def _set_run_state(self, state: str) -> None:
        self._status_key = f"gui.status.{state}"
        self._refresh_status_text()
        self._update_button_states()

    def _refresh_status_text(self) -> None:
        self.status_label.setText(self.translator.tr(self._status_key))

    def _refresh_continue_button_text(self) -> None:
        """Updates the action label according to the loaded schedule state.

        Обновляет подпись действия с учётом состояния загруженного расписания.
        """
        session = self.current_session
        if session is not None and session.position < session.total_samples:
            key = "gui.buttons.continue_checkpoint"
        else:
            key = "gui.buttons.extend_training"
        self.continue_button.setText(self.translator.tr(key))

    def _update_button_states(self) -> None:
        self._refresh_continue_button_text()
        active = self._worker_is_active()
        paused = self._status_key == "gui.status.paused"
        self.start_button.setEnabled(not active)
        self.load_button.setEnabled(not active)
        self.pause_button.setEnabled(active and not paused)
        self.resume_button.setEnabled(active and paused)
        self.stop_button.setEnabled(active)
        self.continue_button.setEnabled(not active and self.current_session is not None)
        self.fit_readout_button.setEnabled(
            not active
            and self.current_session is not None
            and self.current_session.kind == "reservoir"
        )
        self.evaluate_button.setEnabled(not active and self.current_session is not None)
        self.save_model_button.setEnabled(not active and self.current_session is not None)
        # EN: Parameter edits apply only when constructing a new session.
        #     Disabling them during execution prevents the false impression that
        #     a running model changes in-place.
        # RU: Изменения параметров применяются только при создании новой session.
        #     Блокировка во время запуска не создаёт ложного впечатления, что
        #     текущая модель меняется на лету.
        self.parameter_panel.setEnabled(not active)
        self.architecture_combo.setEnabled(not active)
        self.task_combo.setEnabled(not active)
        self.backend_combo.setEnabled(not active)

    def _worker_is_active(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def _repopulate_combo(self, combo: QComboBox, items: tuple[tuple[str, str], ...]) -> None:
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for value, label_key in items:
            combo.addItem(self.translator.tr(label_key), value)
        index = combo.findData(current)
        combo.setCurrentIndex(max(0, index))
        combo.blockSignals(False)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: Any) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _append_log(self, text: str) -> None:
        self.log_view.appendPlainText(text)
        if self.current_session is not None:
            try:
                append_run_log(self.current_session.run_dir, text)
            except OSError:
                # EN: Logging must never crash a simulation.
                # RU: Ошибка записи лога не должна останавливать симуляцию.
                pass

    def _current_configuration_payload(self) -> dict[str, Any]:
        """Returns the complete portable GUI configuration.

        Возвращает полную переносимую конфигурацию GUI.
        """
        return build_user_configuration(
            architecture=str(self.architecture_combo.currentData() or "dci"),
            task=str(self.task_combo.currentData() or "classification"),
            backend=str(self.backend_combo.currentData() or "python_cpu"),
            locale=str(self.language_combo.currentData() or "ru"),
            parameters=self.parameter_panel.values(),
            evaluation=self.evaluation_view.parameters(),
        )

    def export_configuration(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.translator.tr("gui.dialogs.export_config"),
            str(Path("experiment.yaml").resolve()),
            "YAML (*.yaml *.yml)",
        )
        if not path:
            return
        target = save_yaml(path, self._current_configuration_payload())
        self._append_log(self.translator.tr("gui.log.config_exported", path=str(target)))
        QMessageBox.information(
            self,
            self.translator.tr("gui.dialogs.export_config"),
            self.translator.tr("gui.dialogs.config_exported", path=str(target)),
        )

    def import_configuration(self) -> None:
        if self._worker_is_active():
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.translator.tr("gui.dialogs.import_config"),
            str(Path.cwd()),
            "YAML (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            payload = validate_user_configuration(load_yaml(path))
            self._set_combo_data(self.architecture_combo, payload["architecture"])
            self._architecture_changed()
            self._set_combo_data(self.task_combo, payload.get("task"))
            self._set_combo_data(self.backend_combo, payload.get("backend"))
            self.parameter_panel.set_values(dict(payload.get("parameters") or {}))
            self.evaluation_view.set_parameters(dict(payload.get("evaluation") or {}))
            locale = payload.get("locale")
            if locale in {"ru", "en"}:
                self._set_combo_data(self.language_combo, locale)
                self._language_changed()
        except Exception as exc:  # noqa: BLE001 - GUI boundary reports validation errors.
            QMessageBox.critical(self, self.translator.tr("gui.dialogs.error_title"), str(exc))
            return
        self._append_log(self.translator.tr("gui.log.config_imported", path=path))
        QMessageBox.information(
            self,
            self.translator.tr("gui.dialogs.import_config"),
            self.translator.tr("gui.dialogs.config_imported"),
        )

    def copy_configuration(self) -> None:
        import yaml

        text = yaml.safe_dump(
            self._current_configuration_payload(), allow_unicode=True, sort_keys=False
        )
        QApplication.clipboard().setText(text)
        QMessageBox.information(
            self,
            self.translator.tr("gui.buttons.copy_config"),
            self.translator.tr("gui.dialogs.config_copied"),
        )

    def _diagnostic_payload(self) -> dict[str, Any]:
        session = self.current_session
        return diagnostic_report(
            current_configuration=self._current_configuration_payload(),
            run_dir=session.run_dir if session is not None else None,
            status=self._status_key,
            position=session.position if session is not None else None,
            total_samples=session.total_samples if session is not None else None,
            traceback_text=self._last_traceback,
        )

    def copy_diagnostic_report(self) -> None:
        import json

        text = json.dumps(self._diagnostic_payload(), ensure_ascii=False, indent=2)
        QApplication.clipboard().setText(text)
        QMessageBox.information(
            self,
            self.translator.tr("gui.buttons.copy_diagnostics"),
            self.translator.tr("gui.dialogs.diagnostics_copied"),
        )

    def save_diagnostic_report(self) -> None:
        default_dir = (
            self.current_session.run_dir if self.current_session is not None else Path.cwd()
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.translator.tr("gui.dialogs.save_diagnostics"),
            str(Path(default_dir) / "diagnostic_report.json"),
            "JSON (*.json)",
        )
        if not path:
            return
        target = save_json(path, self._diagnostic_payload())
        self._append_log(self.translator.tr("gui.log.diagnostics_saved", path=str(target)))
        QMessageBox.information(
            self,
            self.translator.tr("gui.dialogs.save_diagnostics"),
            self.translator.tr("gui.dialogs.diagnostics_saved", path=str(target)),
        )

    def _format_event(self, event: FrameworkEvent) -> str:
        p = event.payload
        if event.name == "sample_end":
            return self.translator.tr(
                "run.sample",
                position=p.get("position", 0),
                total=p.get("total_samples", "?"),
                label=p.get("label", "-"),
                spikes=p.get("exc_spikes", p.get("spikes", 0)),
                active=p.get("active_exc", p.get("active", 0)),
            )
        if event.name == "checkpoint_saved":
            return self.translator.tr("run.checkpoint_saved", path=p.get("path", ""))
        if event.name == "run_start":
            return self.translator.tr("run.start", architecture=p.get("architecture", event.source))
        if event.name == "run_end":
            return self.translator.tr("run.end", position=p.get("position", 0))
        if event.name == "paused":
            return self.translator.tr("run.paused", position=p.get("position", 0))
        if event.name == "resumed":
            return self.translator.tr("run.resumed")
        if event.name == "stopped":
            return self.translator.tr("run.stopped", position=p.get("position", 0))
        if event.name == "evaluation_end":
            return self.translator.tr(
                "gui.log.evaluation_end", accuracy=f"{100.0 * float(p.get('accuracy', 0.0)):.2f}"
            )
        if event.name == "model_snapshot_saved":
            return self.translator.tr("gui.log.model_saved", path=p.get("path", ""))
        if event.name == "stage_start":
            return self.translator.tr("stage.start", stage=p.get("stage", ""))
        if event.name == "stage_end":
            return self.translator.tr("stage.end", stage=p.get("stage", ""))
        if event.name == "simulation_frame":
            return ""
        return f"[{event.name}] {event.source}"

    def _apply_default_window_geometry(self) -> None:
        """Chooses a sane first-run size from the current screen.

        Выбирает разумный размер первого запуска по доступной области экрана.
        """
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1200, 800)
            return
        available = screen.availableGeometry()
        width = min(1500, max(960, int(available.width() * 0.92)))
        height = min(900, max(700, int(available.height() * 0.90)))
        width = min(width, available.width())
        height = min(height, available.height())
        self.resize(width, height)
        self.move(
            available.x() + max(0, (available.width() - width) // 2),
            available.y() + max(0, (available.height() - height) // 2),
        )

    def _normalize_window_geometry(self) -> None:
        """Clamps restored normal-window geometry to the active screen.

        Qt may restore geometry saved on another display or with another DPI.
        Maximized/full-screen windows are left alone; only normal windows are
        moved and shrunk back into the available desktop area.

        Ограничивает восстановленную геометрию обычного окна текущим экраном.

        Qt может восстановить координаты с другого монитора или DPI.
        Полноэкранное/развёрнутое состояние не трогаем; обычное окно при
        необходимости возвращаем внутрь доступной области рабочего стола.
        """
        if self.isMaximized() or self.isFullScreen():
            return
        frame = self.frameGeometry()
        screen = QGuiApplication.screenAt(frame.center()) or QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        width = min(max(900, self.width()), available.width())
        height = min(max(650, self.height()), available.height())
        x = min(max(frame.x(), available.left()), available.right() - width + 1)
        y = min(max(frame.y(), available.top()), available.bottom() - height + 1)
        self.setGeometry(x, y, width, height)

    def _splitters(self) -> tuple[tuple[str, Any], ...]:
        """Returns user-resizable splitters whose state should persist.

        Возвращает пользовательские splitter-ы, состояние которых сохраняется.
        """
        pairs: list[tuple[str, Any]] = []
        for prefix, view, names in (
            ("live", self.live_view, ("main_splitter", "top_splitter", "bottom_splitter")),
            (
                "evaluation",
                self.evaluation_view,
                (
                    "main_splitter",
                    "overview_splitter",
                    "diagnostics_splitter",
                    "diagnostics_top",
                    "diagnostics_bottom",
                ),
            ),
        ):
            for name in names:
                splitter = getattr(view, name, None)
                if splitter is not None:
                    pairs.append((f"{prefix}/{name}", splitter))
        return tuple(pairs)

    def _refresh_current_tab_layout(self) -> None:
        """Forces the newly shown tab to consume its actual viewport size.

        Hidden pyqtgraph widgets can retain the geometry they had before the
        main window was maximized. Activating the visible layout and refreshing
        splitter/viewport geometry avoids the need to toggle window state.

        Заставляет открытую вкладку использовать фактический размер viewport.

        Скрытые pyqtgraph-виджеты иногда сохраняют геометрию, которая была до
        разворачивания окна. Активация layout и обновление splitter/viewport
        убирают необходимость переключать оконный и полноэкранный режимы.
        """
        if not hasattr(self, "tabs"):
            return
        widget = self.tabs.currentWidget()
        if widget is None:
            return
        layout = widget.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        refresh = getattr(widget, "refresh_layout", None)
        if callable(refresh):
            refresh()
        widget.updateGeometry()
        widget.update()

    def _schedule_visible_tab_refresh(self) -> None:
        """Schedules two post-layout refresh passes for Qt/pyqtgraph.

        Планирует два прохода обновления после перерасчёта Qt-layout.
        """
        self._layout_refresh_timer.start(0)
        self._layout_refresh_delayed_timer.start(60)

    def reset_layout(self) -> None:
        """Restores the default dock and splitter layout.

        Восстанавливает стандартную раскладку dock-панелей и splitter-ов.
        """
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.parameter_dock)
        self.parameter_dock.show()
        self.resizeDocks([self.parameter_dock], [380], Qt.Orientation.Horizontal)
        if hasattr(self.live_view, "main_splitter"):
            self.live_view.main_splitter.setSizes([520, 380])
            self.live_view.top_splitter.setSizes([350, 900])
            self.live_view.bottom_splitter.setSizes([450, 800])
        if hasattr(self.evaluation_view, "main_splitter"):
            self.evaluation_view.main_splitter.setSizes([240, 760])
        if hasattr(self.evaluation_view, "overview_splitter"):
            self.evaluation_view.overview_splitter.setSizes([700, 900])
        if hasattr(self.evaluation_view, "diagnostics_splitter"):
            self.evaluation_view.diagnostics_splitter.setSizes([420, 420])
        if hasattr(self.evaluation_view, "diagnostics_top"):
            self.evaluation_view.diagnostics_top.setSizes([1, 1])
        if hasattr(self.evaluation_view, "diagnostics_bottom"):
            self.evaluation_view.diagnostics_bottom.setSizes([1, 1])
        self._schedule_visible_tab_refresh()

    def save_layout_state(self) -> None:
        """Stores Qt geometry, dock state, and user splitter positions.

        Сохраняет геометрию Qt, dock-state и позиции пользовательских splitter-ов.
        """
        self._settings.setValue("main/geometry", self.saveGeometry())
        self._settings.setValue("main/window_state", self.saveState())
        for key, splitter in self._splitters():
            self._settings.setValue(f"splitters/{key}", splitter.saveState())

    def restore_layout_state(self) -> None:
        """Restores and validates saved window/splitter geometry.

        Восстанавливает и проверяет сохранённую геометрию окна и splitter-ов.
        """
        geometry = self._settings.value("main/geometry")
        state = self._settings.value("main/window_state")
        if geometry is not None:
            self.restoreGeometry(geometry)
        if state is not None:
            self.restoreState(state)
        for key, splitter in self._splitters():
            saved = self._settings.value(f"splitters/{key}")
            if saved is not None:
                splitter.restoreState(saved)
        QTimer.singleShot(0, self._normalize_window_geometry)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        self._schedule_visible_tab_refresh()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().showEvent(event)
        self._schedule_visible_tab_refresh()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._schedule_visible_tab_refresh()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self._worker_is_active() and self.current_control is not None:
            answer = QMessageBox.question(
                self,
                self.translator.tr("gui.dialogs.exit_title"),
                self.translator.tr("gui.dialogs.exit_running"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.current_control.request_stop()
            # EN: Let the worker leave at its next safe boundary before closing
            #     the Qt application; destroying a running QThread is unsafe.
            # RU: Даём worker выйти на ближайшей безопасной границе до закрытия
            #     Qt-приложения; уничтожать работающий QThread небезопасно.
            self._close_when_finished = True
            event.ignore()
            return
        self.save_layout_state()
        event.accept()
