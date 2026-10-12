# Single-neuron numerical-method ablation

The **Single neuron** workspace isolates
the numerical solver from connections, plasticity, encoders, readouts, and classification so
that a method can be characterized before network-level effects obscure the cause of a result.

Open the side menu with the `☰` button and select **Single neuron**. The run controls remain in
the top toolbar, so the parameter form never has to be scrolled to start or stop work. **One
trajectory** runs the method selected in the single-run form. **Compare methods** runs only
the selected outputs in a worker thread and can be cancelled safely. The GUI defaults to
grid-aligned reset for the planned ablation; the event-corrected mode remains selectable.
Current, observation-window duration, initial transient, and step size are shared
conditions. Total simulated time is transient plus observation window; the former can
be longer than the latter. The v(t), u(t), and phase plots show the observation window
only, with its time axis rebased to zero. Saved raw trace arrays retain the absolute
simulation time for reproducibility. The single-run and comparison forms have separate
reset and composition settings.
Trajectories, ISI
convergence, f–I, order versus α, and the summary are independently selectable. The
dt list appears for trajectories, convergence, summary, or α; the current range only for f–I;
and the α list only when its plot is selected. The single-run form shows `s` only for
real Euler composition and `α` only for complex Euler composition. The method
comparison has its own coefficient fields, shown only when that composition is
among the selected methods and a method-comparison output is requested.

The comparison form also exposes run count, base seed, Gaussian current-noise amplitude,
and its fixed update interval. Each method sees the same input stream for the same seed.
One neuron with constant current is deterministic, so multiple seeds give identical
results when the current noise is zero. Firing rate is the number of post-burn-in spikes
divided by the observation-window duration; mean ISI is reported separately and is
undefined with fewer than two spikes. The f–I plot shows mean rate with ±one sample
standard deviation, and the Trials plot keeps every method/seed result. Smooth
convergence, summary, and α order always use constant current, even when noisy
f–I and Trials are selected in the same run. The saved metadata records this.
Trace and phase plots use one seed; the comparison run count applies only to f–I and Trials.
In a study, phase portraits use every selected comparison method and `dt` from the list,
with the comparison `s` or `α`. An individual run uses its method and coefficients.

Learning mode adds contextual `?` buttons beside every parameter and result tab. Explanatory
paragraphs stay out of the normal workspace. The compact View menu selects both axes
or just X/Y, mouse panning or rectangle zoom, all-data fit, and expanded view.
When the graph has focus, `P` pans, `Z` enables rectangle zoom, `X`/`Y`/`B` choose
zoom axes, `+`/`-` scale, `0` fits all data, `F` expands and `Ctrl+S` exports.
The mouse wheel and right drag also zoom; hover shows curve coordinates.
Plots can be saved as PNG or SVG.
In the summary tab, learning mode makes each column heading clickable to explain its
measurement and interpretation; the reference line has its own `?` button. Cell tooltips
carry the same metric descriptions.

## Measurements

Depending on the selected outputs, the study records:

- membrane-potential and recovery-variable traces for every selected method at each listed `dt`;
- the ISI of each successive spike pair minus the reference period, plotted by interval
  number at a selectable `dt`; this is derived from the saved trajectories without new runs;
- phase portraits of every selected method at every listed `dt`, with shared
  constant-current nullclines and selectors for step and visible methods;
- per-seed firing rate under paired noisy inputs for a variability comparison;
- spike count, firing rate, mean ISI, and ISI coefficient of variation after burn-in;
- f-I curves for every selected method;
- spike-period error over a configurable `dt` grid;
- long-horizon spike-time drift against a fine reference;
- spike counts at the finest `dt` alongside the reference count; a mismatch makes
  index-paired drift undefined;
- local-truncation-error and spike-period estimates of convergence order;
- divergence flags and measured simulation time per grid step, including threshold
  handling and event search (the time from one run is exploratory, not a repeated benchmark);
- a sweep of the complex-composition parameter `alpha`;
- a paired control run with the *other* reset mode for every method only if the summary
  is requested. With grid-aligned reset selected, the control uses event-corrected reset.

Every GUI run, including **One trajectory**, is written below `runs/gui/single_neuron/` as:

- `study.json` — configuration, each output's effective reset/noise protocol, input axes,
  source-file SHA-256 fingerprints, and computed scalar/curve results;
- `environment.json` — SNNLab, Git, Python, platform, and package versions;
- `trace.npz` — time, voltage, recovery, and spike-time arrays when a trajectory
  or phase portrait was requested; otherwise an empty archive.

The **Trajectories** tab computes every selected method at every listed `dt`. A compact
dropdown changes which `dt` is displayed without recalculation; another menu toggles
individual method curves. The per-spike ISI plot has its own step and method selectors.
When convergence is included, its refined reference period is the ISI baseline; otherwise
the first method's median ISI at the finest step is an approximate display baseline.
The `dt` field under **Single run** controls the individual trajectory and
phase portrait. f–I and stochastic trials use their own comparison step. With zero current noise, only one f–I
run per condition is needed; the repeated-trials plot requires positive noise.
The convergence plot includes `O(dt)` and
`O(dt²)` guide lines;
these are visual slope references, not fitted results.

**Implicit Euler near a spike:** The `implicit_euler` identifier names the
backward-Euler formula. A direct call to the core stepper computes exactly one
fixed-length step and returns a failed result if its continuous branch has no
real root. The single-neuron grid-reset study and DCI/reservoir populations
use the same shared adaptive grid step with step doubling and local-error
control. The event-reset study uses the same step-doubling kernel and error
controller but locates the spike and continues after reset inside the interval.

The smooth backward-Euler equations reduce to a quadratic equation in voltage.
The old eight-iteration Newton solver could return
an unconverged state when that equation had no real root. The new solver checks
the discriminant and selects the branch continuous with the state as `dt→0`.
Each candidate inner step is calculated once at full size and twice at half size.
The difference estimates local error in both state variables; the two-half-step
solution is accepted only when each error is within its absolute plus relative
tolerance. A missing real root also rejects the candidate. `rtol`, `atol(v)` and
`atol(u)` are visible only when implicit Euler is selected in their respective
single-run or comparison controls; the two modes keep separate tolerances.
Smaller tolerances improve smooth-step accuracy at greater cost; they do not
eliminate grid-reset timing error. Near threshold the recovery state is evaluated
at the crossing rather than at an overshot endpoint.
The outer spike schedule still follows the chosen grid/event reset mode.
DCI and reservoir populations use these same error tolerances and grid-aligned
reset. Refractory periods and changing synaptic currents still affect network
trajectories independently of the integrator.
The run archive records the number
of implicit quadratic solves (three per candidate step), and the summary cost cell
reports their average per outer grid step in its tooltip. Threshold bisection
work is additional. This adaptive method is no longer a fixed-step
backward Euler trajectory at every point; observed convergence order and cost
must be interpreted with that distinction.
If an interval cannot be integrated at the minimum internal step, the run is
marked diverged; its failed trial state is never written to the trajectory.

Generated run directories are intentionally ignored by Git.

**Open result…** in the Single neuron toolbar reads `study.json` and `trace.npz` from an
existing run folder and reconstructs selected plots without simulation. The loader accepts
current multi-dt studies as well as saved single-dt studies (format 4 or 5); format-4 total
duration is converted to the current observation-window definition in the form.

## Reset handling

An Izhikevich spike is a state event, not an ordinary sample on the time grid. With **grid
reset**, the spike is timestamped and reset at the end of the step. This adds an `O(dt)` timing
error independently of the formal order of the smooth ODE integrator.

With **event-corrected reset**, SNNLab brackets the threshold crossing inside the step, refines
its time using the selected integrator, applies `(v, u) <- (c, u + d)` at that event, and
integrates the remainder of the timestep. The selected reset mode applies to trajectories,
f–I, stochastic trials, α, and the primary convergence result. The optional summary also
calculates a control with the other reset mode. Independently of this choice, the error
reference is event-corrected, so grid-reset error includes threshold-time quantization.

## Reference and order estimates

The spike-period reference uses event-corrected midpoint runs at `2h`, `h`, and `h/2`,
followed by second-order Richardson extrapolation of both adjacent pairs. Their difference
is saved as a reference-refinement diagnostic; a large discrepancy calls for a finer
reference before interpreting a method ranking. The period itself is the least-squares slope of spike
time against spike index after burn-in, which is less sensitive to one interval than using only
the last two spikes.

Drift uses only spikes in the observation window and is left undefined if their counts
differ or the first retained spikes cannot be plausibly matched; joining spikes by index
in such cases would produce a misleading number. The summary's spike counts also use
only this window. Compute time per step covers the full simulated run, including warm-up.

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

The two semi-explicit methods also support both update orders: `semi_euler` and
`semi_midpoint` compute the voltage stage before recovery (`v→u`), while
`semi_euler_uv` and `semi_midpoint_uv` reverse it (`u→v`). These are different
numerical schemes, not merely a display-order switch. Both midpoint variants
remain second order for smooth trajectories. The older method IDs remain intact
for existing saved experiments.

The preset list additionally offers phasic spiking, phasic bursting, and spike-frequency
adaptation parameter sets. The form suggests a starting current for these presets; their
reported firing pattern depends on the stimulus and initial state. Phasic responses may
have too few post-transient spikes for an ISI convergence estimate.

## Fast-spiking follow-up

A user-observed large separation between methods with the fast-spiking preset at
`dt = 0.1 ms`, `I = 12`, and grid-aligned reset is a
research question, not yet an established accuracy result. Record the exact a,b,c,d,
current, threshold, initial state, duration, burn-in, dt, reset mode, and method list.
Then compare the same setup at successively smaller dt with grid-aligned reset,
use event-corrected reset as the control, and inspect spike count, matched spike times, ISI,
divergence, and cost. A shrinking difference with dt points toward discretization;
a difference dominated by grid reset points toward event timing. Also check for
different spike counts before interpreting a spike-index drift statistic. Repeat on
independent noisy input streams only after the deterministic behavior is understood.
Saved PNG/SVG exports from each plot make this comparison reviewable.

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
