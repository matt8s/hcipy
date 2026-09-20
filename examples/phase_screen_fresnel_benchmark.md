# Mazda seeded phase-screen milestone evidence

Measured 2026-09-19 at `be78c3d16ce71df8343882d3fcac1504d413a106`, including
the RNG repair `631b43e`. Only the preserved user `.gitignore` differed from
the committed tree during these measurements.

## Reproduce

```sh
python examples/phase_screen_fresnel_backend.py --backend numpy --sizes 256 512 1024 --output phase_numpy128.json
CUDA_VISIBLE_DEVICES=0 python examples/phase_screen_fresnel_backend.py --backend cupy --sizes 256 512 1024 --output phase_cupy128.json
python examples/phase_screen_fresnel_backend.py --backend numpy --sizes 256 512 1024 --precision complex64 --output phase_numpy64.json
CUDA_VISIBLE_DEVICES=0 python examples/phase_screen_fresnel_backend.py --backend cupy --sizes 256 512 1024 --precision complex64 --output phase_cupy64.json
```

Hardware/software matches the [Gaussian benchmark](gaussian_fresnel_benchmark.md):
RHEL 9, two Xeon Gold 6326 CPUs (64 logical CPUs), one A100-PCIE-40GB (GPU 0),
driver 595.71.05, CUDA runtime 12.9, Python 3.14.7, NumPy 2.5.3, SciPy 1.18.1,
CuPy 14.2.0, array_api_strict 2.6.1. HCIPy defaults; SciPy FFT with the default
64-worker policy, no MKL/pyFFTW. No concurrent benchmark processes were launched.

Parameters: 1 µm wavelength, 1 mm waist, 1 W source, 20 mm window, r0=3 mm,
L0=20 mm, integrated Cn²=9.596167163856036e-10 m^(1/3), wind (0.01,-0.005) m/s,
0.1 m propagation and 1 mm receiver radius. Seed 12; four correlated frozen-flow
frames at 0, 0.01, 0.02, 0.03 s. Phase spectral oversampling 2, transfer-function
oversampling 1, padding 2, aperture quadrature 8×8. Three warm-ups and 20 repeated
resident evaluations of the final snapshot. GPU timings synchronize the stream.

## Timings and bottleneck

Resident execution includes phase application, Fresnel propagation, receiver
power and two total-power reductions. End-to-end also includes setup, the four
host-evolved/uploaded frames, warm-up, all repeats and final output download.

| Precision | Grid | CPU resident median ms | GPU resident median ms | CPU end-to-end s | GPU end-to-end s |
|---|---:|---:|---:|---:|---:|
| complex128 | 256² | 7.344 | 0.796 | 0.696 | 0.821 |
| complex128 | 512² | 32.245 | 0.824 | 1.547 | 0.765 |
| complex128 | 1024² | 98.431 | 1.606 | 5.176 | 2.700 |
| complex64 | 256² | 8.463 | 0.830 | 0.710 | 0.831 |
| complex64 | 512² | 32.316 | 0.957 | 1.578 | 0.784 |
| complex64 | 1024² | 122.026 | 1.191 | 5.860 | 2.742 |

These single-run measurements use untuned CPU threading. Small-grid and precision
rankings are noisy. At 1024² complex128 the ranges were 88.449–111.476 ms CPU,
1.594–1.625 ms GPU. Contexts/pools persist across sizes within each process.
Only the first size pays cold GPU context and phase/reduction compilation costs.

Complex128 breakdown (milliseconds; per-frame values below are the final frame):

| Grid | CPU setup | CPU first Fresnel | GPU setup | GPU first Fresnel | GPU input upload | Host evolution in GPU run | Phase upload | GPU final field download |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256² | 88.193 | 44.270 | 61.641 | 81.240 | 2.243 | 36.802 | 0.598 | 0.595 |
| 512² | 189.915 | 153.549 | 172.155 | 128.182 | 2.478 | 107.603 | 2.560 | 1.138 |
| 1024² | 583.188 | 714.418 | 638.764 | 601.156 | 7.232 | 368.244 | 7.704 | 3.717 |

Initial GPU context: 167.351 ms. Three-call resident warm-ups: 2.772/2.852/4.847 ms.
The initial 256² host evolution took 255.526 ms, including first-use setup;
the first phase/propagation/reduction call took 92.718 ms. Final-frame scalar
downloads were 0.108/0.133/0.193 ms. At 1024², host evolution ranged from
241.674 to 395.560 ms. Thus GPU-resident optical execution is fast, but the
**host atmosphere boundary dominates fresh-frame throughput**.

Complex64 GPU setup: 64.926/176.581/674.653 ms; first Fresnel:
90.026/123.160/633.450 ms; input uploads: 2.147/2.017/8.997 ms. Final-frame
host evolution: 39.396/118.532/354.002 ms; phase uploads: 0.409/1.456/4.911 ms;
final field downloads: 0.221/0.910/4.953 ms. Initial context: 170.856 ms.
The JSON output retains every frame/component timing for reruns.

## Numerical checks and memory

Final-frame CPU reference relative field error on GPU was
3.64e-16/4.28e-16/4.75e-16 for complex128, and
2.64e-7/3.78e-7/3.84e-7 for complex64. The reference uses legacy CPU phase
application with a double-precision host screen, so complex64 input is promoted
in the reference path; these errors include phase/FFT precision differences.
CPU complex64 errors against that reference were 1.58e-7–1.93e-7.

Complex128 phase-only power error was ≤2.3e-16 W. Propagated total-power loss
was ≤3.6e-9 W in these cases; finite-window cropping prevents exact conservation.
GPU complex64 output-power error was ≤6.0e-7 W.
Final receiver powers were 0.8586717408341544, 0.8584712698654882 and
0.8552793097057685 W for complex128. Different grid sizes draw different spectral
realizations even with the same seed: these powers are **not a grid-convergence
sequence**. CPU/GPU comparisons at each size use the same explicit host screen.

GPU complex128 pool used/reserved MiB: 11.5/25.5, 54.0/105.5, 224.0/425.5.
GPU complex64: 6.0/15.0, 28.0/63.0, 116.0/255.0. GPU-process peak RSS was
1,456,201,728 bytes in both runs; CPU complex128 peaks were
336,506,880 / 468,496,384 / 997,355,520 bytes. RSS is lifetime high-water usage;
GPU pools include cached allocations and are not peak GPU memory measurements.

## RNG cost

The new copyable CuPy RNG uses a fresh native RandomState per call. With seed 12,
three warm-ups and 20 synchronized `rng.normal(size=n)` calls in float64:

| Samples per call | NumPy median ms | CuPy median ms |
|---:|---:|---:|
| 1 | 0.00154 | 0.39639 |
| 65,536 | 0.98050 | 0.39928 |
| 1,048,576 | 16.64783 | 0.52237 |

This quantifies the ~0.4 ms native-generator setup boundary. The existing
atmospheric example uses NumPy screen generation, so these RNG times are a
separate microbenchmark, not part of the phase-screen pipeline timing.
To reproduce the microbenchmark's timed operation, create
`rng = make_random_generator(xp, seed=12)` from `hcipy._math.random`, warm up
three times, then time 20 calls using `perf_counter()` and synchronize the current
CuPy stream before/after each call. Do not include output copies in that timing.

## Validation and next task

Full suite: **2296 passed, 209 remaining baseline failures, 617 skipped,
1 xpassed**, 1139.14 s. This resolves 31 previous RNG failures and adds 19 passing
stream/phase tests. No new failing test IDs. Lint and diff checks passed; slow
tests unrun. Strict standalone phase example passed at 64².

Next fix finite-layer state semantics: a reproduced `evolve_until(0.1)` left
`layer.t == 0`; reset after displacement did not restore the initial screen;
changing Cn² by a factor of four left cached phase unchanged. These existing
defects are separate from resident snapshot execution and must be addressed
before expanding atmosphere state management or GPU-native evolution.
