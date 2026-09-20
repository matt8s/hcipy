# Finite-screen MFT residency audit

`mft_residency_audit.py` profiles the existing two-band finite-screen algorithm
and an isolated CuPy inverse-MFT matrix-product prototype. It is a measurement
tool, not a GPU implementation of the finite atmosphere.

## Reproduce

```sh
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python examples/mft_residency_audit.py --cupy --sizes 256 512 \
    --precision complex128 --repeats 5 --output mft_audit128.json
```

Repeat with `--precision complex64`. Omit `--cupy` for a CPU-only audit.
NumPy/SciPy/Numba/NumExpr CPU thread limits are explicit; this is a one-thread
baseline, not a tuned comparison. GPU operations synchronize before and after
each sample. Three MFT warm-up calls precede the five steady samples. Factory
setup, fresh sampling, matrix construction, translation, synthesis, actual
shift-plus-synthesis frames, uploads, first GPU product and download are reported
separately. Fresh-sampling samples include the first draw. Full CPU sampling and
synthesis use the existing complex128 implementation in both precision runs;
only the isolated MFT and its shared coefficients change precision.

The synthetic case uses a 1 m square, r0=0.15 m at the phase spectrum's reference
wavelength, L0=10 m, inner scale=0.01 m, oversampling 8 and seed 42. High-frequency
FFT and low-frequency MFT coefficients retain their existing partition. Matrix
upload includes host conjugation and any first-use allocation overhead. The
pre-upload synchronization is not a measurement of CUDA context initialization.
Case wall time includes the audit's reference checks; it excludes imports and
metadata collection and is not an optical simulation's end-to-end time.

## Numerical boundary

The forward convention is exp(-i k·x) with spatial quadrature weights; the
inverse convention is exp(+i k·x) with frequency weights divided by (2π)^2.
The prototype evaluates `M1.conj().T @ coefficients @ M2.conj().T`, multiplied
by the scalar inverse weight. It applies only to the regular two-dimensional
grids used here. Five spatial samples are independently checked with direct
positive-sign Fourier quadrature. GPU output is checked against the entire CPU
MFT output using exactly shared coefficients. These are deterministic numerical
checks, not validation of ensemble turbulence statistics or backend RNG streams.

Current `MatrixFourierTransform.backward(NewStyleField(cupy_array, grid))`
raises `NotImplementedError` at the SciPy BLAS conversion boundary. The script
records that result explicitly. CPU matrices and intermediates, NumPy dot/BLAS,
and legacy Field construction prevent general resident execution. GPU prototype
products allocate intermediates and outputs; no reusable device scratch or cache
policy has been implemented. Pool used/reserved bytes are reported separately
from free device memory and process peak RSS; none is a peak-device measurement.

## Implementation direction

Keep CPU grid/matrix setup and the legacy BLAS path. A narrow NewStyleField path
can cache uploaded matrices and weights by namespace name, device and precision,
using the existing backend dispatch. Preserve `precompute_matrices` and
`allocate_intermediate` semantics, output ownership, deepcopy and tensor behavior.
Validate forward/backward signs, nonuniform weights, 1D/2D rectangular and shifted
grids, both precisions, device/precision switches and strict portable operations.
Only then extend multiscale sampling and finite-layer evolution, preserving both
bands, realization replay and parameter invalidation.

Upstream [Array API issue #335](https://github.com/ehpor/hcipy/issues/335) and
[Fourier batching PR #358](https://github.com/ehpor/hcipy/pull/358) overlap this
work. PR #358 changes the same MFT methods, replaces explicit BLAS selection
with `get_blas_funcs`, and moves tensor handling inside transforms. Coordinate
the residency proposal with that work before choosing the final patch layout.
