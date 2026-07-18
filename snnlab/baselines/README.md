# Frozen baselines

This directory stores human- and machine-readable reference configurations for results that should remain reproducible while the framework evolves.

Этот каталог хранит человеко- и машиночитаемые эталонные конфигурации результатов, которые должны оставаться воспроизводимыми при развитии фреймворка.

## `dci_mnist_v0`

The first frozen end-to-end DCI/STDP MNIST development baseline:

- 400 E + 400 I neurons;
- 1500 unique training images repeated over 5 epochs;
- 7500 total presentations;
- 55.4% balanced SNN-voting accuracy on a 500-sample single-seed test run;
- 13 silent E neurons after training.

This is a development reference, not a publication benchmark. The YAML file records the exact parameter set and the evaluation limitations.

Первый зафиксированный end-to-end DCI/STDP baseline для MNIST:

- 400 E + 400 I нейронов;
- 1500 уникальных обучающих изображений, 5 эпох;
- 7500 предъявлений;
- 55.4% accuracy сбалансированного SNN-голосования на 500 test sample и одном seed;
- 13 молчащих E-нейронов после обучения.

Это эталон разработки, а не публикационный benchmark. YAML фиксирует параметры и ограничения evaluation.
