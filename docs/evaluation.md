# Evaluation protocol / Протокол оценки

## DCI

Evaluation has two stages:

1. collect E-neuron responses for label assignment;
2. collect responses for the held-out test set and apply an SNN-native decoder.

Assignment policies:

- `full_train_pool` — baseline-compatible, may overlap STDP samples;
- `exclude_training` — assignment candidates exclude every sample in the training schedule;
- `training_subset` — assignment uses only samples previously shown to STDP.

The exact original pool indices and overlap count are written to `data_protocol.json`.

`balanced_topk` normalizes each neuron's response by its own best assignment response and weights votes by relative selectivity. It is still an SNN-native decoder: no external classifier is fitted.

## Reservoir

Reservoir evaluation uses the fitted external readout by design. Evaluation uses an isolated RNG stream and does not alter the continuation trajectory.
