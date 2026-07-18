# SNNLab 1.0

[English](README.md) | [Русский](README_RU.md)

SNNLab is a Python research framework and desktop environment for experiments with spiking neural networks. It currently includes two working architectures, real-time visualizations, reproducible experiment configuration, checkpoint continuation, and an RU/EN interface.

![DCI evaluation](docs/assets/dci_evaluation.png)

## What is included

- **DCI / Diehl–Cook-like SNN** with Izhikevich excitatory and inhibitory populations, Input→E STDP, lateral inhibition, target-rate homeostasis, receptive fields, and SNN-native classification.
- **Reservoir / LSM-like SNN** with fixed recurrent weights, selectable numerical integration methods, live spike visualization, and a readout for reservoir-computing experiments.
- Seven numerical integration methods for Izhikevich neurons.
- Live input image, spike raster, population-state view, activity history, receptive fields, evaluation diagnostics, and interactive values on plots.
- Pause, stop, checkpoint loading, exact continuation, and further training.
- RU/EN interface and a study mode with contextual explanations of parameters.
- YAML experiment configuration, run logs, metrics, and reproducibility metadata.

## Requirements

- 64-bit Python **3.11 or 3.12** is recommended.
- Windows 10/11 is the primary tested platform. Linux should also work with a compatible Qt environment.
- Internet access is required the first time MNIST is downloaded.

## Installation options

### Option 1 — install the ready-made wheel

This is the simplest option for colleagues who only want to run SNNLab. The wheel contains the packaged SNNLab code; `pip` installs it into a Python environment and creates the `snnlab-gui` command.

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the wheel with GUI and MNIST support from the folder containing the file:

```powershell
python -m pip install "./snnlab-1.0.0-py3-none-any.whl[gui,mnist]"
snnlab-gui
```

The wheel is not a standalone `.exe`: Python is still required. The GUI and TensorFlow dependencies are downloaded by `pip`.

### Option 2 — install from the Git repository for normal use

Use this when the source code has been cloned or downloaded as a ZIP, but you do not plan to edit the package itself:

```powershell
cd SNNLab
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ".[gui,mnist]"
snnlab-gui
```

This installs a fixed copy of the current source into the environment.

### Option 3 — editable installation for development

Use this when modifying SNNLab. Changes in the source directory become available without rebuilding the wheel:

```powershell
cd SNNLab
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[gui,mnist,dev]"
python -m pytest
snnlab-gui
```

### Option 4 — minimal core installation

For code-only use without the desktop GUI or MNIST loader:

```powershell
python -m pip install .
```

or from the wheel:

```powershell
python -m pip install ./snnlab-1.0.0-py3-none-any.whl
```

This installs NumPy, scikit-learn, PyYAML, and the SNNLab core. Add the `gui` and `mnist` extras when those features are needed.

## Quick start

Start the desktop interface:

```powershell
snnlab-gui
```

Useful command-line entry points:

```powershell
snnlab-dci --help
snnlab-reservoir --help
snnlab-dci-regression --help
```

Example experiment configurations are stored in `examples/` and can be imported from the GUI configuration menu.

## Checkpoints and continued training

SNNLab checkpoints preserve model weights, dynamic state, training position, sample schedule, and random-number-generator state. A trusted checkpoint can be loaded in the GUI and used for evaluation or further training.

Checkpoint files use Python pickle internally. Do not open checkpoints from untrusted sources.

## Experiment outputs

GUI runs are stored under `runs/gui/`. Depending on the experiment, the run directory contains configuration, logs, metrics, checkpoints, model snapshots, and data-protocol metadata.

Do not commit generated `runs/`, `.venv/`, `build/`, or `dist/` directories to Git.

## Current scope

- Python CPU backend.
- DCI classification is currently demonstrated on MNIST.
- Reservoir experiments support architecture-specific analysis and readout evaluation.
- Additional neuron models, arbitrary visual network construction, C++, CUDA, and remote execution are planned extensions.

## Project structure

```text
SNNLab/
├── snnlab/          # framework and GUI source code
├── tests/           # automated tests
├── examples/        # example experiment configurations
├── docs/assets/     # images used by the README
├── README.md
├── README_RU.md
├── LICENSE
└── pyproject.toml
```

## License

SNNLab is distributed under the MIT License. See [LICENSE](LICENSE).
