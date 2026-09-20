# Mazda resident periodic-screen milestone

Measured at clean public commit `fedc76525239f814bcb3a3fab449da579a602c21`,
including the implementation in `363bb6e`. These measurements exercise actual
coefficient evolution and fresh coefficient sampling, rather than only applying
a static phase snapshot.

## Reproduce and scope

```sh
python examples/spectral_noise_backend.py --backend numpy --output spectral_numpy64.json
CUDA_VISIBLE_DEVICES=0 python examples/spectral_noise_backend.py --backend cupy --output spectral_cupy64.json
python examples/spectral_noise_backend.py --backend numpy --precision float32 --output spectral_numpy32.json
CUDA_VISIBLE_DEVICES=0 python examples/spectral_noise_backend.py --backend cupy --precision float32 --output spectral_cupy32.json
```

Mazda: RHEL 9, Linux 5.14.0-687.39.1.el9_8.x86_64, two Xeon Gold 6326 CPUs
(32 physical / 64 logical cores), GPU 0 A100-PCIE-40GB, driver 595.71.05,
CUDA runtime 12.9. Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1, CuPy 14.2.0,
array_api_strict 2.6.1. CPU FFT uses SciPy with HCIPy's default 64-worker policy;
CPU threading is not tuned. Benchmark commands ran sequentially.

Each command runs 256²/512²/1024² output grids, spectral oversampling 2 (therefore
four times as many frequency coefficients), three warm-ups and 20 evolved frames.
Twenty fresh independent coefficient draws are then measured separately. All
GPU stages synchronize the current stream. Host seed bookkeeping remains, but
CuPy random arrays, spectral coefficients, shifts and synthesis remain on GPU.

Synthetic parameters: 20 mm spatial window, 40 mm screen period, von Karman
r0=3 mm and L0=20 mm at wavelength 1 µm; Gaussian waist 1 mm, input power 1 W,
0.1 m propagation, receiver radius 1 mm. Seed 12, frozen-flow velocity
(0.01,-0.005) m/s, dt=0.01 s, final time 0.2 s. Fresnel padding 2 and transfer
oversampling 1; aperture quadrature 8×8. See [model/conventions](spectral_noise_backend.md).

This periodic FFT model differs from FiniteAtmosphericLayer's multiscale model.
Its timings are not a replacement-workload comparison with earlier finite-layer
benchmarks or WaveTrain. Equal seeds across backends/grids do not generate matched
realizations. Numerical comparisons below explicitly share final coefficients.

## Synchronized steady-state timings

Median milliseconds; each operation is measured separately with synchronization:

| Precision | Grid | CPU sampling | GPU sampling | CPU shift | GPU shift | CPU synthesis | GPU synthesis |
|---|---:|---:|---:|---:|---:|---:|---:|
| float64 | 256² | 10.721 | 1.060 | 12.488 | 0.205 | 4.096 | 0.239 |
| float64 | 512² | 48.376 | 1.203 | 56.176 | 0.243 | 12.913 | 0.263 |
| float64 | 1024² | 255.943 | 2.230 | 262.243 | 0.598 | 69.514 | 0.821 |
| float32 | 256² | 9.654 | 1.045 | 12.767 | 0.200 | 3.155 | 0.248 |
| float32 | 512² | 44.525 | 1.414 | 54.743 | 0.228 | 11.192 | 0.263 |
| float32 | 1024² | 267.225 | 1.971 | 268.380 | 0.399 | 53.926 | 0.482 |

Sampling draws two full normal arrays and combines them with the cached spectral
amplitude. It does not include a new phase synthesis. Frozen-flow evolution uses
the shift/synthesis stages, without resampling the atmosphere. The optics stage
below includes phase application, Fresnel propagation, aperture power and two
total-power reductions.

| Precision | Grid | CPU optics ms | GPU optics ms | CPU complete run s | GPU complete run s |
|---|---:|---:|---:|---:|---:|
| float64 | 256² | 8.502 | 0.820 | 0.955 | 0.600 |
| float64 | 512² | 30.568 | 0.740 | 3.626 | 0.428 |
| float64 | 1024² | 139.141 | 1.543 | 18.565 | 1.708 |
| float32 | 256² | 9.140 | 0.838 | 0.983 | 0.735 |
| float32 | 512² | 34.032 | 0.895 | 3.733 | 0.512 |
| float32 | 1024² | 148.582 | 1.107 | 18.635 | 1.934 |

Complete runs include setup, first calls, warm-up, evolved frames, independent
draw timing, and final downloads. At 1024² float64, GPU shift ranged
0.595–0.680 ms, synthesis 0.819–0.846 ms, optics 1.535–1.780 ms and sampling
2.211–2.687 ms. Corresponding CPU ranges were 258.263–269.147,
65.687–73.868, 124.985–155.292 and 230.431–319.476 ms. These are single-run,
untuned CPU comparisons; no optimal-performance claim is made.

## Setup, transfers and memory

Float64 GPU component timings in milliseconds:

| Grid | Host optics setup | Factory setup + uploads | Source upload | First sampling | First synthesis | First optics | Final download |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256² | 43.722 | 40.585 | 0.705 | 70.699 | 19.489 | 184.948 | 7.634 |
| 512² | 98.308 | 145.983 | 1.445 | 2.802 | 15.954 | 97.619 | 11.676 |
| 1024² | 261.820 | 754.556 | 4.002 | 3.132 | 35.725 | 473.972 | 60.181 |

Initial GPU context stage: 175.872 ms (float64), 181.477 ms (float32).
Three warm-ups: 7.904/3.785/8.272 ms for float64. Context/compilation/pools persist
between sizes within a process. Factory setup includes host PSD and Fourier-grid
construction plus amplitude/axis uploads; first synthesis includes FFT factors
and first-use execution. Final downloads include coefficients, phase, output
field and scalar measurements. Reference computations are outside the timer.

Float32 factory setup + uploads: 39.565/192.537/871.073 ms; source uploads:
0.508/2.551/5.665 ms; first sampling: 70.963/2.678/3.381 ms; first synthesis:
20.654/14.597/51.605 ms; first optics: 316.136/151.233/572.973 ms; final downloads:
4.970/12.656/30.743 ms. Short-run GPU totals are dominated by CPU setup and first
optical call, rather than repeated spectral synthesis.

GPU pool used/reserved MiB (256²/512²/1024²): float64 approximately
26.5/50, 114/218, 464/890; float32 13.5/27, 58/119, 236/487.
GPU-process peak RSS bytes: 661,786,624 / 810,270,720 / 1,251,307,520 (float64),
1,560,555,520 across float32 cases. CPU peak RSS: 309,760,000 / 508,915,712 /
1,159,159,808 bytes (float64), 292,970,496 / 415,764,480 / 881,934,336 (float32).
RSS is lifetime high-water usage; GPU pool figures are current used/reserved
allocations, not peak device memory. Persistent caches are included.

## Accuracy and validation

GPU/CPU final phase relative errors were ≤5.30e-16 (float64) and ≤2.72e-7
(float32); propagated-field errors ≤7.48e-16 and ≤4.71e-7 respectively.
The CPU reference uses downloaded final coefficients and legacy double-precision
FFT scratch, so float32 differences include FFT working-precision effects.
Phase-only power was 1 W to reported precision; float64 propagated output lost
at most 3.92e-9 W through the finite window. Float32 power error was ≤5.97e-7 W.

Final GPU receiver powers in float64: 0.8613869093722032,
0.8634672959057985, 0.8687517095318757 W. Different grid sizes correspond to
different realizations and are not a deterministic convergence sequence.

All **42 new tests passed**, covering direct quadrature, period, coordinate axes,
temporal composition/reversal, 1D–3D, float32/64, seeded replay, independent copies
and residency. Independent 64-screen ensembles on each backend check absolute
variance and three periodogram bands against the sampled PSD using six SEM plus
1% numerical allowance. Full suite: **2508 passed, 209 unchanged baseline
failures, 617 skipped, 1 xpassed**, 1153.91 s; no new failing test IDs.
Lint/diff checks passed. Slow tests remain unrun.

Next: correct the separate multiscale translation axis-order defect, validate
its low/high-frequency sum against direct Fourier quadrature, then extend and
profile the MFT execution path for finite-layer device residency.
