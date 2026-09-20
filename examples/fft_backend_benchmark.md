# Mazda resident FFT measurements

Measured 2026-09-19 at clean implementation commit
`d4a66fbb72b782c67ec135a8b0525c4a6caa11e0`.

```sh
python examples/fft_backend.py --backend numpy --output fft_numpy128.json
CUDA_VISIBLE_DEVICES=0 python examples/fft_backend.py --backend cupy --output fft_cupy128.json
python examples/fft_backend.py --backend numpy --precision complex64 --output fft_numpy64.json
CUDA_VISIBLE_DEVICES=0 python examples/fft_backend.py --backend cupy --precision complex64 --output fft_cupy64.json
```

## Configuration

Mazda: RHEL 9, Linux 5.14.0-687.39.1.el9_8.x86_64; two Xeon Gold 6326 CPUs
(32 physical / 64 logical cores), GPU 0 A100-PCIE-40GB. Driver 595.71.05,
CUDA runtime 12.9; Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1, CuPy 14.2.0,
array_api_strict 2.6.1. CPU dispatch uses SciPy FFT and HCIPy's default 64-worker
policy for these sizes; no MKL/pyFFTW. No concurrent benchmark processes launched.

Each command evaluates grids 256²/512²/1024² sequentially, padding 2,
full Fourier field of view, emulated FFT shifts, 3 warm-up pairs and 20 measured
forward/backward pairs. Input is `exp(-r²/w²)`, w=1 mm, on a 20 mm square grid.
Both timed CPU and GPU paths use explicit NewStyleField with the requested
precision. Each GPU measurement synchronizes the current stream before/after
execution. See [boundary and normalization details](fft_backend.md).

## Execution times

Median milliseconds for one transform, plus complete-run wall time:

| Precision | Grid | CPU forward ms | GPU forward ms | CPU backward ms | GPU backward ms | CPU total s | GPU total s |
|---|---:|---:|---:|---:|---:|---:|---:|
| complex128 | 256² | 3.483 | 0.157 | 3.680 | 0.175 | 0.274 | 0.293 |
| complex128 | 512² | 8.878 | 0.231 | 10.550 | 0.218 | 0.667 | 0.147 |
| complex128 | 1024² | 46.701 | 0.615 | 65.933 | 0.774 | 3.558 | 0.522 |
| complex64 | 256² | 3.720 | 0.175 | 3.350 | 0.192 | 0.235 | 0.360 |
| complex64 | 512² | 13.414 | 0.254 | 10.965 | 0.273 | 0.743 | 0.131 |
| complex64 | 1024² | 58.922 | 0.414 | 55.305 | 0.475 | 3.496 | 0.534 |

Totals include setup, first calls, warm-up, all repeats and both output downloads.
Context/pool state persists across sizes; first-size totals include cold context
initialization. CPU threading is untuned. Precision rankings at smaller grids
are noisy; these are single-run measurements, not hardware-optimal speedup claims.
At 1024² complex128 GPU forward ranged 0.612–0.620 ms and backward 0.771–0.778 ms;
CPU ranges were 43.661–51.716 ms and 63.608–86.759 ms respectively.

Component timings in milliseconds:

| Precision | Grid | CPU setup | GPU setup | GPU upload | GPU first forward | GPU first backward | GPU output download |
|---|---:|---:|---:|---:|---:|---:|---:|
| complex128 | 256² | 63.860 | 39.800 | 2.029 | 65.780 | 4.262 | 5.187 |
| complex128 | 512² | 109.026 | 105.737 | 1.674 | 15.342 | 0.454 | 12.753 |
| complex128 | 1024² | 406.852 | 395.394 | 4.193 | 37.118 | 0.850 | 51.834 |
| complex64 | 256² | 46.163 | 58.733 | 2.716 | 94.646 | 4.519 | 1.209 |
| complex64 | 512² | 95.377 | 93.231 | 0.967 | 18.298 | 0.753 | 4.734 |
| complex64 | 1024² | 383.881 | 412.703 | 3.023 | 73.288 | 0.882 | 22.724 |

Initial GPU context stage: 167.546 ms (complex128), 188.781 ms (complex64).
GPU warm-up times for three pairs: 1.518/1.331/4.052 ms (complex128),
1.868/1.989/2.953 ms (complex64). First forward includes transfer of cached
shift factors and initial FFT planning/compilation. Download includes the padded
Fourier field and the restored spatial field. CPU uploads are no-copy asarray;
CPU first-forward/backward times were 13.291/6.474, 19.838/23.622,
80.178/107.533 ms for complex128. Full component timings are emitted in JSON.

## Accuracy and memory

Complex128 GPU analytic relative field errors: 1.04e-13, 2.09e-13, 4.18e-13.
These agree with the CPU path and reflect the existing shift-factor evaluation.
GPU/legacy-CPU differences are ≤4.77e-16; inverse round-trip errors ≤4.72e-16;
Parseval errors ≤4.45e-16. Complex64 GPU analytic error ≤2.79e-7,
legacy-CPU difference ≤2.80e-7, round-trip error ≤3.89e-7, Parseval error ≤2.64e-7.
Legacy CPU reference execution uses complex128 scratch before casting, while the
timed portable complex64 path computes in complex64 throughout.

GPU pool used/reserved MiB for 256²/512²/1024²:
- complex128: 15/24, 60/104, 240/424.
- complex64: 7.5/15, 30/63, 120/255.

CPU peak RSS bytes: 284,672,000 / 366,723,072 / 714,698,752 (complex128),
274,624,512 / 335,245,312 / 567,980,032 (complex64).
GPU-process peak RSS: 603,934,720 / 669,286,400 / 923,131,904 (complex128),
1,518,706,688 bytes across all complex64 cases. RSS includes process-lifetime
high-water allocations, including initialization/compilation. GPU pools are
current used/reserved allocation counts, not peak device-memory measurements.

## Validation and continuation

All 160 new FFT cases passed on NumPy, CuPy and array_api_strict. Full suite:
**2466 passed, 209 unchanged baseline failures, 617 skipped, 1 xpassed**,
1145.19 s. No new failing test IDs; lint/diff checks passed. Slow tests unrun.

The next task is resident coefficient sampling, translation and synthesis in
SpectralNoiseFactoryFFT, preserving its normalization, seeded copy semantics
and periodic model. Validate spectral statistics before extending the finite
atmosphere's multiscale FFT/MFT representation. The fast resident FFT timing
alone does not establish GPU-native atmospheric evolution performance.
