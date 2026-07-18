# Reservoir parameter guide / Параметры Reservoir

Reservoir weights are fixed; the trainable object is the external readout. This architecture is intentionally separate from DCI.

- `input_scale`: input weight magnitude. Too high can saturate neuron-time occupancy.
- `recurrent_scale`: recurrent drive. Too high can produce self-sustained activity.
- `input_density`: fraction of input connections.
- `recurrent_density`: fraction of recurrent connections.
- `excitatory_ratio`: sign distribution of outgoing recurrent weights.
- `select_k`: number of spike-count features retained before readout fitting.

MNIST and Iris use different safe presets because their encoders have very different input dimensionality.

Веса Reservoir фиксированы; обучается внешний readout. Для MNIST и Iris используются разные безопасные presets из-за разной размерности encoder.
