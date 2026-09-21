# Finite-screen MFT residency audit

`mft_residency_audit.py` profiles the existing two-band finite-screen algorithm
and the resident CuPy inverse MFT. It is a measurement tool, not yet a GPU
implementation of the complete finite atmosphere.

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
shift-plus-synthesis frames, uploads, first GPU MFT and download are reported
separately. Fresh-sampling samples include the first draw. Full CPU sampling and
synthesis use the existing complex128 implementation in both precision runs;
only the isolated MFT and its shared coefficients change precision.
Check GPU utilization and active processes before benchmarking. Numerical checks
remain useful under contention, but timings from a concurrently loaded GPU must
not be reported as resident-throughput evidence.

The synthetic case uses a 1 m square, r0=0.15 m at the phase spectrum's reference
wavelength, L0=10 m, inner scale=0.01 m, oversampling 8 and seed 42. High-frequency
FFT and low-frequency MFT coefficients retain their existing partition. Matrix
upload includes any first-use allocation overhead. The pre-upload synchronization
is not a measurement of CUDA context initialization.
Case wall time includes the audit's reference checks; it excludes imports and
metadata collection and is not an optical simulation's end-to-end time.

## Numerical boundary

The forward convention is exp(-i k·x) with spatial quadrature weights; the
inverse convention is exp(+i k·x) with frequency weights divided by (2π)^2.
The inverse path is algebraically equivalent to
`M1.conj().T @ coefficients @ M2.conj().T`, multiplied by the scalar inverse
weight. Five spatial samples are independently checked with direct
positive-sign Fourier quadrature. GPU output is checked against the entire CPU
MFT output using exactly shared coefficients. These are deterministic numerical
checks, not validation of ensemble turbulence statistics or backend RNG streams.

`MatrixFourierTransform.backward(NewStyleField(cupy_array, grid))` now uses
resident matrix products after CPU matrix setup and cached uploads. Portable
matrix multiplication controls intermediate allocation; no reusable device
scratch is promised. Pool used/reserved bytes are reported separately from free
device memory and process peak RSS; none is a peak-device measurement.

## Implementation direction

CPU grid/matrix setup and the legacy BLAS path are retained. The NewStyleField
path caches uploaded matrices and weights by namespace name, device and precision
using existing backend dispatch. `allocate_intermediate` remains a legacy BLAS
option because portable matmul owns backend intermediate allocation. Production
tests cover signs, nonuniform weights, 1D/2D rectangular and shifted grids,
precisions, devices, strict operations, ownership, deepcopy and tensor fields.
Multiscale sampling and finite-layer evolution remain CPU-only and must preserve
both spectral bands, realization replay and parameter invalidation when extended.

Upstream [Array API issue #335](https://github.com/ehpor/hcipy/issues/335) and
[Fourier batching PR #358](https://github.com/ehpor/hcipy/pull/358) overlap this
work. PR #358 changes the same MFT methods, replaces explicit BLAS selection
with `get_blas_funcs`, and moves tensor handling inside transforms. Coordinate
the residency proposal with that work before choosing the final patch layout.

## Validation status

Direct-quadrature, inverse, Parseval, cache, ownership, deepcopy, tensor,
precision, backend and two-device tests pass for NumPy, CuPy and
`array_api_strict`. The full default suite retains its pre-existing CuPy math
failures; no new failing test IDs were introduced by the resident MFT path.

Clean-tree accuracy runs at 256², 512² and 1024² used exactly shared low-band
coefficients. CuPy/legacy-CPU relative L2 differences were at most 3.40e-15 for
complex128 and 1.06e-6 for complex64. Independent direct-quadrature relative
errors were at most 1.26e-13 and 4.97e-7, respectively.

No timing table is reported from these runs. An unrelated workload resumed on
both GPUs during measurement and produced bimodal steady-state samples, so the
timings do not establish uncontended throughput or speedup. Re-run the commands
above on an idle device before making a performance claim.
