# Seeded atmosphere → phase snapshot → Fresnel → receiver power

```sh
python examples/phase_screen_fresnel_backend.py --backend numpy
CUDA_VISIBLE_DEVICES=0 python examples/phase_screen_fresnel_backend.py --backend cupy
python examples/phase_screen_fresnel_backend.py --backend array_api_strict --sizes 64
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_random_streams.py tests/test_phase_screen_backend.py
```

Use `--precision complex64`, `--seed 12`, `--sizes 256 512 1024`, `--frames 4`,
`--dt 0.01`, `--repeats 20`, and `--output report.json` as needed. The JSON report
records software, commit, tracked changes, seed, physical parameters, powers,
CPU reference error, synchronized timings and memory. No plotting is required.

## Physical and computational boundaries

Synthetic SI parameters: wavelength 1 µm, Gaussian waist 1 mm, input power 1 W,
20 mm square grid, Fried parameter 3 mm at that wavelength, outer scale 20 mm,
wind (0.01, -0.005) m/s, screen-to-receiver distance 0.1 m, receiver radius 1 mm.
The integrated Cn² is derived with `Cn_squared_from_fried_parameter`; its units
are m^(1/3), distinct from local Cn² in m^(-2/3). Phase is in radians, intensity
in W/m² and integrated power in W. These are illustrative parameters, not a
measured atmospheric profile.

`FiniteAtmosphericLayer` supplies seeded von Karman phase screens using its
existing multiscale spectral model. Generation and frozen-flow evolution remain
on NumPy. Each frame's phase is explicitly transferred to a `NewStyleField`.
`PhaseApodizer` then applies that resident snapshot to a fresh input wavefront;
Fresnel propagation and power reduction use the selected backend. This is a
time series of independent optical evaluations through **correlated** atmospheric
frames, not repeated passage of one beam through the same screen.

An atmospheric layer can also act directly on a NewStyleField wavefront through
`layer(wavefront)` / `layer.backward(wavefront)`. This transfers the current CPU
phase on every call. For repeated evaluations of the same atmospheric state,
use the explicit resident snapshot as in this example. Updating the underlying
resident phase array is reflected on the next PhaseApodizer application.
The phase is cast to the wavefront's complex precision before exponentiation,
including for strict Array API backends.

Resident snapshot execution has no screen upload or CPU sample generation.
This milestone does **not** claim GPU-native atmospheric evolution. The finite
spectral representation has finite bandwidth and eventually repeats under
translation; do not extrapolate this small-window example to long observations.
The demonstration uses absolute `evolve_until(t)` and fresh layer instances for
replay. Finite-layer reset/time/cache semantics are a separate follow-up audit.

Phase multiplication alone preserves power and its adjoint undoes the phase.
After propagation, finite-window cropping can lose scattered light. The receiver
integral uses 8×8 fractional aperture coverage and already weighted pixel power.
The Fresnel sampled-transfer kernel uses oversampling 1 and padding 2, as in the
[Gaussian milestone](gaussian_fresnel_backend.md).

## Random streams and validation

CuPy RNG construction is repaired independently of this host-screen pipeline.
CuPy 14.2's modern Generator lacks some required methods and cannot be copied
using the supported state API. HCIPy instead uses native `RandomState` per-call
substreams, seeded by a copyable host generator (one uint64 seed per call).
Samples are generated on the GPU, without modifying global CuPy state. Copies
reproduce future calls, including after mixed distributions. Call shapes/order
and software/device must match; neither NumPy-identical draws nor invariance to
splitting a draw is promised. Each call has generator setup overhead. Native
CuPy restrictions, including weighted choice without replacement, remain.

Tests verify normal moments, independent-stream and lag-one correlations with
six-standard-error thresholds, exact same-backend seed/copy replay, and device
residency. CPU/GPU optical comparisons use the **same explicit host phase**;
they make no assumption about matching NumPy/CuPy random draws.

Phase tests cover complex64/128 and NumPy/CuPy/strict, scalar phases, updates,
phase-only power, adjoints, unchanged input, frozen-flow replay, wavelength
scaling, evolving optical output and repeated resident execution. A separate
64-realization ensemble compares the finite-screen phase structure function at
three separations with the continuous von Karman model. Independent SeedSequence
children are used; correlated time frames are not counted as independent samples.
The bound is six estimated standard errors plus an 8% finite-grid spectral
truncation allowance. This tests the spectrum through its real-space structure
function, not just a single screen image.

## Timing interpretation

The report separates context initialization, host setup, input upload, first
Fresnel call, per-frame host evolution, phase upload, propagation/reductions,
scalar download, warm-up, repeated resident snapshot time, and final field
download. GPU work is synchronized. End-to-end includes the time series, three
warm-ups and all repeated evaluations; CPU reference validation and serialization
are excluded. The first frame may compile phase/reduction kernels.
Resident timing includes aperture, screen-only and output total-power reductions.

Memory is process-lifetime peak RSS plus current GPU pool used/reserved bytes
and free device memory. Pool numbers are not peak GPU allocation measurements.
Use separate processes for independent cold-start cases. Do not compare these
timings directly to the Gaussian benchmark: this example performs extra phase
operations and power reductions, and separately measures host atmosphere work.
