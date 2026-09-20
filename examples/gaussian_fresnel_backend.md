# Gaussian beam → Fresnel propagation → aperture power

Run from the repository root with the HCIPy development environment:

```sh
python examples/gaussian_fresnel_backend.py --backend numpy
CUDA_VISIBLE_DEVICES=0 python examples/gaussian_fresnel_backend.py --backend cupy
python examples/gaussian_fresnel_backend.py --backend array_api_strict --sizes 128
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_gaussian_backend.py
```

Optional arguments: `--precision complex64`, `--sizes 256 512 1024`,
`--padding 2`, `--repeats 20`, `--output results.json`.
NumPy is required; CuPy and array_api_strict are optional. On Mazda the verified
installation is `python -m pip install 'cupy-cuda12x[ctk]'` (CuPy 14.2.0,
environment-local CUDA 12.9 wheels, driver 595.71.05). System CUDA is not required.

## Physical model and interfaces

This synthetic scalar, monochromatic example has a 1 mm waist (1/e field radius),
1 µm vacuum wavelength, 1 W input power, 0.1 m propagation distance, 20 mm square
computational window and 1 mm circular receiver radius. All lengths are metres;
intensity is W/m² and pixel/aperture powers are W. HCIPy arrays do not enforce
units; these are the explicit normalization conventions of this example.

`GaussianBeam(..., z=0)` generates the waist using existing HCIPy interfaces.
An explicit `NewStyleField(xp.asarray(data), cpu_grid)` opts into backend arrays
without changing global Field configuration. `FresnelPropagator` operates on the
Wavefront. The aperture measurement is `xp.sum(output.power.data * aperture)`:
`power` already includes grid integration weights. Fractional aperture coverage
is computed with 8×8 subpixel samples on the CPU before transfer. This is
intensity-weighted quadrature, not coherent transmission by a fractional mask.

CPU boundaries: source/aperture sampling, grid metadata, transfer-function
generation and first-use propagator setup. The scalar FourierFilter kernel is
uploaded once per backend/device/precision change. Repeated scalar propagation,
intensity and regular-grid power reductions stay on the device. Only reporting
copies the output and scalar measurements to the host. Nonuniform host grid
weights may require conversion during power evaluation. Full GPU grids,
polarization, other optical elements and matrix-valued filters are not covered
by this milestone. Each propagator owns mutable scratch storage.

## Validation and sampling

The independent paraxial reference uses
`z_R = π w₀²/λ`, `q = 1 + i z/z_R`, and
`E(r,z) = sqrt(2P/(πw₀²)) exp(ikz) exp(-r²/(w₀²q))/q`.
The encircled power is `P [1 - exp(-2a²/w(z)²)]`.
This matches Fresnel's positive longitudinal-phase convention. Existing
`GaussianBeam.evaluate()` uses the opposite convention away from the waist;
its intensity remains a valid independent reference.

Tests use relative complex-field L2 tolerances of 2e-12 (complex128) and 2e-6
(complex64), where the latter allows FFT roundoff. Separate carrier and envelope
exponentials avoid losing envelope phase when adding it to a large `kz`.
Near-zero Gaussian tails are assessed by a global norm rather than relative
per-pixel errors. Aperture error is dominated by pixel quadrature: tests refine
64→128→256 pixels and padding 1→2→4. At 256 pixels the aperture error is below
0.001 W for the 1 W test beam. CPU/GPU agreement, unchanged inputs, retained
outputs, backward propagation, repeated cached execution and chained steps are
also tested, including strict Array API execution with CPU setup.

`num_oversampling=1` samples the analytic transfer function at Fourier grid
centres. It conserves power on a sufficiently large window in the tested direct
transfer branch. HCIPy's default oversampling of 2 averages the transfer function
over cells and can attenuate power. Tests cover this default and both Fresnel
setup branches, with a 0.4% intensity discretization tolerance. Cropped output
power need not be conserved when light leaves the computational window.
Padding does not replace resolving the waist or enlarging the physical window.

## Benchmark interpretation

JSON reports hardware/software, configuration, git commit/dirty state, parameters,
precision, numerical error and memory. Timings separate CUDA context creation,
host setup, source/aperture upload, first call (kernel setup/upload and FFT plan),
three warm-up calls, steady-state samples, and final output download. Each steady
sample propagates the **same input** and measures receiver power; chained physical
steps are a separate correctness test. GPU timings synchronize the current stream.
End-to-end is the sum of these measured stages, including warm-up and all repeats;
it excludes imports, validation reference generation and JSON serialization.

Memory reports process-lifetime peak RSS and, on CuPy, current pool used/reserved
bytes plus device free memory after the loop. These are not per-call peak GPU
allocation measurements; FFT plans/context allocations can lie outside the pool.
CPU timing uses HCIPy's default FFT dispatch/thread policy, recorded in metadata.
For independent cold measurements, launch each case in a fresh process.

Application-specific geometry, atmospheric profiles, source coherence, target
statistics, receiver response and timing requirements must be specified separately.
The next milestone is seeded phase-screen application and statistical validation
of evolving atmosphere, building on HCIPy's existing atmosphere classes.
