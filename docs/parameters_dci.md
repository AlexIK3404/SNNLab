# DCI parameter guide / Параметры DCI

| Parameter | Meaning | Typical failure when too low | Typical failure when too high |
|---|---|---|---|
| `input_gain` | Scales Input→E conductance | silent samples, many retries | broad synchronous activity |
| `weight_exc_inh` | One-to-one E→I recruitment | weak inhibition recruitment | excessive paired inhibition |
| `target_total_inhibition` | Total lateral I→E inhibition | weak competition | almost no accepted activity |
| `eta` | Input→E STDP learning rate | receptive fields remain noisy | unstable/rapid weight collapse |
| `x_target` | Target pre-trace in STDP rule | excessive potentiation | excessive depression |
| `mu` | Weight dependence exponent | abrupt boundary attraction | very slow weight differentiation |
| `target_spikes_per_sample` | Per-neuron homeostasis target | strong suppressive current | weak competition between neurons |
| `homeo_learning_rate` | Homeostatic adaptation speed | leaders persist too long | oscillation or population shutdown |

The GUI study mode contains longer RU/EN explanations for every parameter, including Izhikevich `a`, `b`, `c`, and `d`.

Режим изучения GUI содержит более подробные RU/EN-описания всех параметров, включая `a`, `b`, `c`, `d` модели Ижикевича.
