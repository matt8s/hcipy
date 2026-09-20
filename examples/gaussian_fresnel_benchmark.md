# Mazda Gaussian/Fresnel milestone measurements

Measured 2026-09-19 at implementation commit
`94e51bb946f8363819544ece14c3a26e76d5e5f5`. The only tracked working-tree
difference during measurement was the user's `.gitignore` edit; numerical code
matched this commit. No concurrent CPU/GPU benchmark processes were launched.

## Reproduction and environment

```sh
python examples/gaussian_fresnel_backend.py --backend numpy --output numpy128.json
CUDA_VISIBLE_DEVICES=0 python examples/gaussian_fresnel_backend.py --backend cupy --output cupy128.json
python examples/gaussian_fresnel_backend.py --backend numpy --precision complex64 --output numpy64.json
CUDA_VISIBLE_DEVICES=0 python examples/gaussian_fresnel_backend.py --backend cupy --precision complex64 --output cupy64.json
```

Each command runs grids 256², 512², 1024² sequentially, with padding 2,
3 warm-ups and 20 measured calls per grid. Each call propagates and integrates
receiver power. Current CUDA stream synchronized before/after every sample.
See [model and timing definitions](gaussian_fresnel_backend.md).

- Mazda: RHEL 9, kernel `5.14.0-687.39.1.el9_8.x86_64`, glibc 2.34.
- CPU: two Intel Xeon Gold 6326 @ 2.90 GHz, 32 physical cores / 64 logical CPUs.
- GPU 0: NVIDIA A100-PCIE-40GB, driver 595.71.05, CUDA driver API 13.2,
  runtime 12.9. GPU 1 unused.
- Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1, CuPy 14.2.0,
  array_api_compat 1.15.0, array_api_strict 2.6.1.
- HCIPy defaults, user overrides disabled. MKL and pyFFTW absent; NumPy arrays
  use SciPy FFT with HCIPy's 64-worker policy for these padded sizes. CuPy uses
  native cuFFT. CPU worker-count tuning was not performed.
- Synthetic SI parameters: waist 0.001 m, wavelength 1e-6 m, distance 0.1 m,
  window 0.02 m, aperture radius 0.001 m, input power 1 W. Fourier transfer
  oversampling 1; receiver aperture oversampling 8×8. All grids use the direct
  transfer-function branch.

## Synchronized timings

Steady-state median milliseconds, including aperture power measurement:

| Precision | Grid | CPU ms | GPU ms | CPU/GPU ratio | CPU total s | GPU total s |
|---|---:|---:|---:|---:|---:|---:|
| complex128 | 256² | 6.406 | 0.515 | 12.4 | 0.260 | 0.490 |
| complex128 | 512² | 21.200 | 0.562 | 37.7 | 0.708 | 0.398 |
| complex128 | 1024² | 84.500 | 1.351 | 62.5 | 2.972 | 1.191 |
| complex64 | 256² | 5.601 | 0.843 | 6.6 | 0.223 | 0.695 |
| complex64 | 512² | 18.330 | 0.561 | 32.7 | 0.672 | 0.419 |
| complex64 | 1024² | 82.891 | 0.920 | 90.1 | 2.961 | 1.040 |

Totals include setup, warm-up and 20 calls, not just propagation. Contexts and
memory pools persist between sizes in each process. Thus the first GPU case pays
cold-start costs that later sizes do not. These are single-run measurements with
the default CPU thread policy, not a claim of hardware-optimal speedup. GPU
complex64 at 256² varied from 0.582–0.962 ms, so its precision comparison is noisy.
The 1024² complex128 ranges were CPU 79.912–102.821 ms and GPU 1.341–1.369 ms.

Component timings (milliseconds; first call includes CPU kernel construction,
kernel upload, FFT plan creation and first propagation):

| Precision | Grid | CPU host setup | CPU first call | GPU host setup | GPU input upload | GPU first call | GPU output download |
|---|---:|---:|---:|---:|---:|---:|---:|
| complex128 | 256² | 57.403 | 52.842 | 39.539 | 2.201 | 192.127 | 1.515 |
| complex128 | 512² | 94.864 | 127.536 | 94.146 | 2.645 | 285.382 | 1.846 |
| complex128 | 1024² | 271.697 | 706.391 | 372.441 | 8.789 | 773.972 | 4.739 |
| complex64 | 256² | 49.423 | 46.037 | 35.463 | 2.036 | 206.612 | 1.215 |
| complex64 | 512² | 108.015 | 144.408 | 124.845 | 1.413 | 278.448 | 0.808 |
| complex64 | 1024² | 309.867 | 726.878 | 310.961 | 6.245 | 699.605 | 1.659 |

Initial GPU context stage: 176.175 ms (complex128), 172.915 ms (complex64).
Three-call GPU warm-up: 68.044/2.410/3.876 ms for complex128 and
260.649/2.082/2.712 ms for complex64. First-case warm-ups include reduction/kernel
compilation. CPU input conversion is ≤1.073 ms; CPU reporting ≤4.523 ms.

## Accuracy and memory

| Grid | GPU complex128 field relative L2 error | GPU/CPU relative difference | Aperture error W | GPU pool used/reserved MiB | CPU peak RSS MiB |
|---:|---:|---:|---:|---:|---:|
| 256² | 8.96e-16 | 3.69e-16 | 7.52e-4 | 10.5 / 21.5 | 262.1 |
| 512² | 9.23e-16 | 4.29e-16 | 1.14e-4 | 50.0 / 101.5 | 339.3 |
| 1024² | 1.04e-15 | 4.63e-16 | 2.56e-6 | 208.0 / 421.5 | 668.5 |

Analytical aperture power: 0.8643904702354173 W. All complex128 total powers
agree with 1 W to 4.5e-16 W. Complex64 GPU field error is 2.83e-7–3.81e-7,
GPU/CPU difference ≤3.48e-7, total-power error ≤5.37e-7 W. The complex64
1024² aperture error is 3.02e-6 W. Grid convergence is dominated by aperture
quadrature rather than propagation error.

GPU complex64 pool used/reserved: 5.25/12.75, 25/52.75, 104/244.75 MiB.
GPU-process peak RSS: 1066.8 MiB for the complex128 run; 651.8/735.1/1014.3 MiB
through the complex64 cases. CPU complex64 peak RSS: 263.3/334.1/613.3 MiB.
RSS is process-lifetime high-water memory, not incremental per-case allocation.
Pool figures include retained allocations and omit some context/FFT allocations;
they are not peak GPU usage. GPU free memory after complex128 cases was
41.93/41.85/41.51 GB (decimal).

## Validation status and next work

Before CuPy installation: 2220 passed, 860 skipped, 1 xpassed. After installing
CuPy, unchanged core tests exposed 240 baseline failures. After this milestone,
the full suite gave **2246 passed, 240 same baseline failures, 617 skipped,
1 xpassed**, with no observed regressions. All 23 added Gaussian tests passed.
The standalone strict example also ran successfully (128²; relative field error
7.19e-16). Slow tests remain unrun. Existing GPU RNG initialization and NumPy
conversion failures are the next core compatibility work before seeded evolving
phase-screen validation. CPU setup/first-call cost now dominates these short
GPU runs; profile that boundary before pursuing batching or multi-GPU execution.
