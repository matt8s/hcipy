# Resident periodic spectral noise and frozen-flow propagation

```sh
python examples/spectral_noise_backend.py --backend numpy
CUDA_VISIBLE_DEVICES=0 python examples/spectral_noise_backend.py --backend cupy
python examples/spectral_noise_backend.py --backend array_api_strict --sizes 64
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_spectral_noise_backend.py
```

Options include `--precision float32`, `--oversample 2`, `--seed 12`,
`--sizes 256 512 1024`, `--repeats 20`, and `--output report.json`.

## API and device boundaries

The legacy constructor and NumPy seed interface are retained:

```python
factory = hp.SpectralNoiseFactoryFFT(psd, cpu_grid, oversample=2)
noise = factory.make_random(seed=12)
```

To opt into native backend arrays:

```python
factory = hp.SpectralNoiseFactoryFFT(psd, cpu_grid, oversample=2,
                                    xp=cp, dtype=cp.float32)
noise = factory.make_random(seed=12)
phase = noise()                         # NewStyleField on the GPU
noise.shift([0.001, -0.002])            # In-place, metres, coordinate order x/y
phase_at_new_position = noise()
saved = noise.copy()                   # Independent coefficients and FFT scratch
```

`dtype` is the real output precision; coefficients and FFTs use its corresponding
complex precision. Default explicit-backend precision is float64. The factory
evaluates the PSD and constructs its grids/Fourier geometry on the CPU, then
uploads amplitudes and frequency axes once. Construct a new factory if grid,
PSD parameters, dtype or device changes. Keep the selected current device fixed
for sampling; mismatched backend/device generators are rejected.

Fresh coefficient sampling, phase translation and inverse FFT synthesis use the
selected backend. CuPy's existing HCIPy RNG uses two independently seeded native
RandomState calls per realization, with small host-side seed bookkeeping.
There are no CPU sample arrays in the CuPy path. `array_api_strict` uses the
existing NumPy RNG fallback and therefore only validates portable array execution.

For a stream of independent realizations, pass a persistent generator from
`hcipy._math.random.make_random_generator(xp, seed)`. Passing the same integer
seed repeatedly replays the same realization. Generator copies replay subsequent
draws; `noise.copy()` snapshots an already sampled spectral realization. There is
no new reset method: replay a seed or restore a saved realization. GPU/CPU
streams need not match, and call-sequence/software/device reproducibility limits
of the backend RNG still apply.

`shifted(displacement)` deep-copies before shifting. Repeated frozen-flow evolution
should use in-place `shift()` when a snapshot is unnecessary, avoiding repeated
copies of the factory and FFT caches. Returned phase Fields retain their CPU
grid metadata and their array data on the backend device.

## Fourier conventions and physical model

Frequency-grid coordinates k are angular spatial frequency in rad/m. The PSD
callable receives that grid but returns W(k), expressed per cycles/m frequency
area, consistent with HCIPy's `power_spectral_density_von_karman` convention.
For dimension d and frequency-cell weight Δk^d, coefficients are sampled as

`C(k) = sqrt(W(k) (2π)^d / Δk^d) (a + i b)`, with independent unit normal a,b.

Synthesis is the real part of HCIPy's inverse Fourier quadrature:
`φ(x) = Re Σ C(k) exp(ik·x) Δk^d/(2π)^d`.
Consequently the ensemble mean-square phase is `Σ W(k) Δk^d/(2π)^d`, with no
additional fitted factor of two. Coefficients are not constrained to Hermitian
symmetry; taking the real part supplies the real-valued process.

Translation multiplies coefficients by `exp(-ik·s)`, producing `φ(x-s)`.
Coordinate order is x/y/...; array axes are reversed, with x varying fastest.
`factory.period` now reports `2π/Δk` per coordinate, including actual FFT padding.
This fixes the legacy period's missing oversampling and reversed rectangular
dimensions, and fixes legacy translation's invalid broadcast reduction.

The example uses a periodic von Karman phase screen with r0=3 mm, L0=20 mm
at wavelength 1 µm. A 1 W Gaussian with 1 mm waist crosses the screen and
propagates 0.1 m to a 1 mm-radius receiver on a 20 mm square grid. Phase is radians,
intensity W/m² and aperture power W. Frozen-flow velocity is (0.01,-0.005) m/s,
step 0.01 s. Each step evaluates a fresh beam through the evolved screen, rather
than repeatedly passing a beam through it. These are synthetic SI parameters.

With oversample=2 the screen period is 40 mm. Finite frequency spacing suppresses
unrepresented low frequencies; the Nyquist cutoff omits high frequencies. No
inner-scale model is added. This periodic representation is **not** a replacement
for FiniteAtmosphericLayer's multiscale FFT/MFT model. Equal seeds across grids or
backends do not create a matched realization or a grid-convergence sequence.

## Validation and timing

Tests compare shared coefficients against direct inverse Fourier quadrature on
rectangular grids, including fractional translations and padding 1/2. Integer
pixel shifts are checked against array rolls; whole-period shifts and twenty
small shifts versus one large shift verify periodicity and temporal composition.
Float64 bounds are 3e-12 relative L2; float32 uses 3e-6 for direct comparisons and
5e-6 for accumulated translation roundoff. Copies and seeded streams are tested
independently, and GPU tests reject unintended host-array uploads during sampling,
translation and synthesis after setup.

An ensemble of 64 independent screens per backend checks absolute mean-square
phase and periodogram power in three frequency bands against the sampled PSD.
Acceptance uses six estimated standard errors plus a 1% numerical allowance.
This verifies discrete spectral normalization and shape, not convergence to an
unbounded atmosphere. Applications still need window/bandwidth and ensemble
convergence studies for their actual path and receiver parameters.

The example reports synchronized first calls, repeated coefficient sampling,
translation, synthesis, and phase/Fresnel/power computation separately. Factory
setup includes CPU PSD/FFT geometry plus cached uploads; source upload and final
coefficient/phase/field downloads are reported separately. Three warm-up steps use
zero displacement; the measured steps actually evolve the coefficients. Independent
fresh-sampling timing follows the frozen-flow sequence and does not replace it.
End-to-end includes all of these stages, excluding reference validation and JSON
serialization. CPU reference comparison uses downloaded final coefficients, not
matching seeds. GPU pools are current used/reserved bytes, not peak GPU memory;
RSS is process-lifetime peak usage.
