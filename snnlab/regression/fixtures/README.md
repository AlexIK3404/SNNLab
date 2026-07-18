# DCI notebook regression fixture / Эталон DCI из блокнота

`dci_notebook_clean_v1.npz` was generated from the current clean target-rate DCI
implementation in the exported research notebook script `snn_castom (1).py`.

`dci_notebook_clean_v1.npz` сгенерирован из текущей clean target-rate DCI-реализации
в экспортированном исследовательском скрипте блокнота `snn_castom (1).py`.

The fixture freezes / Fixture фиксирует:

- initial Input -> E weights / начальные веса Input -> E;
- initial E/I membrane states / начальные состояния мембран E/I;
- sample schedule / порядок sample;
- per-sample scalar metrics / метрики каждого sample;
- E/I spike counts / counts E/I;
- complete final-attempt spike logs / полные spike logs финальных попыток;
- final weights / итоговые веса;
- final E/I continuous state / итоговое непрерывное состояние E/I;
- final homeostatic current / итоговый гомеостатический ток;
- final Poisson RNG state / итоговое состояние Poisson RNG.

The fixture is intentionally small (`8 E + 8 I`, six samples) so the full
regression test stays fast while exercising retries, rejected presentations,
STDP, rest dynamics, lateral inhibition, homeostasis, and state continuation.

Fixture намеренно мал (`8 E + 8 I`, шесть sample), чтобы полная regression-проверка
оставалась быстрой, но задействовала retries, rejected-предъявления, STDP,
динамику отдыха, латеральное торможение, гомеостаз и продолжение состояния.
