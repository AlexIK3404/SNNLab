# CLI

Command-line entry points for running SNNLab without the desktop interface.

Командные точки входа для запуска SNNLab без desktop-интерфейса.

## Commands

- `snnlab-dci` — DCI experiments.
- `snnlab-reservoir` — Reservoir experiments.
- `snnlab-dci-regression` — regression/reference validation utility.

CLI modules should stay thin: parse arguments, build configuration, call framework APIs, and print/save results. Simulation logic belongs in `architectures/` and orchestration in `experiments/`.