# Single-neuron numerical-method ablation

The **Single neuron** workspace isolates
the numerical solver from connections, plasticity, encoders, readouts, and classification so
that a method can be characterized before network-level effects obscure the cause of a result.

Open the side menu with the `☰` button and select **Single neuron**. A quick trace runs in
the GUI thread; the complete study runs in a worker thread and can be cancelled safely.

## Measurements

The complete study records:

- membrane-potential and recovery-variable traces;
- spike count, firing rate, mean ISI, and ISI coefficient of variation after burn-in;
- f-I curves for every selected method;
- spike-period error over a configurable `dt` grid;
- long-horizon spike-time drift against a fine reference;
- local-truncation-error and spike-period estimates of convergence order;
- divergence flags and measured Python compute time per grid step;
- a sweep of the complex-composition parameter `alpha`;
- a paired control run with grid-aligned reset for every method.

Every complete GUI run is written below `runs/gui/single_neuron/` as:

- `study.json` — configuration and all scalar/curve results;
- `environment.json` — SNNLab, Git, Python, platform, and package versions;
- `trace.npz` — full time, voltage, recovery, and spike-time arrays.

Generated run directories are intentionally ignored by Git.

## Reset handling

An Izhikevich spike is a state event, not an ordinary sample on the time grid. With **grid
reset**, the spike is timestamped and reset at the end of the step. This adds an `O(dt)` timing
error independently of the formal order of the smooth ODE integrator.

With **event-corrected reset**, SNNLab brackets the threshold crossing inside the step, refines
its time using the selected integrator, applies `(v, u) <- (c, u + d)` at that event, and
integrates the remainder of the timestep. The complete study always uses this mode for its
primary results and runs grid reset separately as a control. The reset choice in the single-run
panel affects only the displayed trace.

## Reference and order estimates

The spike-period reference uses event-corrected midpoint runs at `h` and `h/2`, followed by
second-order Richardson extrapolation. The period itself is the least-squares slope of spike
time against spike index after burn-in, which is less sensitive to one interval than using only
the last two spikes.

For intrinsically bursting and chattering presets this quantity is the average inter-spike
period, not a separately identified burst period. Burst-level statistics should be added before
using those presets to make claims about burst timing.

Local order is estimated away from a reset discontinuity. A heavily substepped RK4 solution is
used for one smooth step, and the slope of local error against `h` is reduced by one to report
the corresponding global order. This is the primary order test; a spike-period fit over a finite
`dt` interval may be distorted by cancellation or a non-asymptotic point.

For a composition of two explicit-Euler substeps,

```text
psi_h = phi_(c2 h) composed with phi_(c1 h),
```

second order requires both `c1 + c2 = 1` and `c1*c2 = 1/2`. No real pair satisfies these
conditions. The conjugate choice `c1,2 = 1/2 +/- i*alpha` satisfies them only at `alpha = 1/2`.
SNNLab therefore treats `alpha = 0.5` as the theoretical second-order point and verifies it
numerically instead of assigning the same order to every complex composition.

## What was not imported from the MATLAB controller prototype

The measured f-I curve is useful here. Gain-scheduled IMC-PI control, zero-order hold, actuator
delay, PWM, sensor loss/noise, disturbance rejection, closed-loop frequency response, and
real-time deadlines answer a different research question: control of a neuron's firing rate.
They are not part of the numerical-method ablation and are deliberately not mixed into its
metrics.

The current workspace also leaves the square-root SNIC onset fit out of the primary analysis.
The supplied MATLAB figure reports a strong whole-range fit but a very weak onset-only fit, so
using that fit as evidence for onset behavior would be unjustified without a denser near-rheobase
protocol and uncertainty analysis.

## Planned experiment sections

The same side menu is the extension point for the remaining experiment sections:

- single neuron — implemented;
- fixed neuron population and fixed network dynamics;
- plasticity and weight evolution with the network held fixed;
- end-to-end training and task metrics across repeated seeds.

Network accuracy should be interpreted only after the single-neuron and fixed-network error,
stability, activity, and cost measurements are available.
