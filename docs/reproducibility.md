# Reproducibility / Воспроизводимость

For every public result, archive:

- `experiment.yaml`;
- `environment.json`;
- `data_protocol.json`;
- the evaluated checkpoint/model snapshot;
- evaluation parameters and metrics;
- the Git commit hash;
- all seeds.

The schedule hash detects accidental changes to ordering or subset selection. `data_protocol.json` stores full indices because a seed alone is not enough when dataset versions, shuffling implementations, or pool construction change.

For publication-quality comparisons, run multiple seeds and keep the test protocol fixed. Use validation data for decoder thresholds and hyperparameter selection.
