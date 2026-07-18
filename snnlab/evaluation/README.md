# Evaluation

Architecture-specific evaluation pipelines and decoders.

Архитектурно-зависимые процедуры оценки и декодирования результатов.

## Current responsibilities

- DCI label assignment and SNN-native voting/decoding;
- class-response and selectivity diagnostics;
- confusion matrices and per-class metrics;
- Reservoir readout evaluation.

Evaluation must not mutate the training trajectory unless a protocol explicitly says so. Use isolated copies/RNG streams for inference-style evaluation.

Evaluation не должен незаметно менять траекторию обучения. Для inference-подобной оценки используются изолированные состояния и RNG-потоки.