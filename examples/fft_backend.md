# Resident FastFourierTransform execution

```sh
python examples/fft_backend.py --backend numpy
CUDA_VISIBLE_DEVICES=0 python examples/fft_backend.py --backend cupy
python examples/fft_backend.py --backend array_api_strict --sizes 64
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/test_fft_backend.py
```

Optional flags: `--precision complex64`, `--padding 2`, `--repeats 20`,
`--sizes 256 512 1024`, `--no-emulate-fftshifts`, `--output report.json`.

## Supported execution and boundaries

Construct `FastFourierTransform` with a CPU grid as before. Pass an explicit
`NewStyleField(xp.asarray(values), grid)` to opt into backend-native execution.
The grid, padding/cropping geometry, and phase factors are constructed on the
CPU. Scratch storage and phase factors are cached on the input backend/device
in the corresponding complex precision; repeated calls reuse them. Changing
backend/device/precision replaces this single cache. Legacy Field execution
retains its existing NumPy implementation and working arrays.

Float32/complex64 inputs use complex64 arithmetic; float64/complex128 inputs
use complex128. The legacy NumPy implementation uses an internal complex128
buffer even for complex64 inputs, so small precision-dependent differences are
expected. Real input is cast to the complex working precision. Tensor components
are transformed individually and stacked with the backend namespace, preserving
tensor shape without CPU transfers. This is not a new batched FFT implementation.

Both emulated FFT shifts and explicit backend fftshift/ifftshift are supported,
along with input/output offsets, padding, Fourier cropping and one to three
dimensions in the tests. Returned Fields retain their corresponding CPU grids;
array values stay on the selected device. Inputs and earlier outputs are not
overwritten by subsequent transforms. A warmed transform can be deep-copied;
the copy owns independent scratch arrays. Transform instances are not safe for
concurrent use because their scratch space is mutable.

## Scientific validation

The continuous convention is `F(k) = ∫ f(x) exp(-ik·x) dx`. Forward quadrature
includes the input grid weights; inverse quadrature includes output frequency
weights divided by `(2π)^d`. Parseval therefore compares spatial power with
frequency power divided by `(2π)^d`. Cropped frequency domains need not preserve
power or yield an exact round trip.

Tests compare against independent dense quadrature matrices for shifted,
non-square odd/even grids, with and without padding/cropping and shift emulation.
Relative L2 bounds are 2e-12 in complex128 and 2e-6 in complex64. Separate tests
cover 1D/3D round trips, device residency, cache/precision/backend changes,
tensor-component ownership, and deep-copy compatibility. GPU/CPU comparisons use
identical supplied values; no random sequence equivalence is assumed.

The example transforms a deterministic Gaussian `exp(-r²/w²)` with w=1 mm on a
20 mm square grid. The analytic transform is `πw² exp(-w²|k|²/4)`. Its finite
window truncation is negligible here. The script reports analytical error,
legacy CPU difference, inverse round-trip error and Parseval error.

## Timing and next step

The JSON report separates context initialization, CPU setup, input upload,
first forward/backward calls, three warm-up pairs, steady forward/backward
times and final output downloads. All GPU stages synchronize the current stream.
End-to-end includes setup, warm-up and all repeats, excluding imports, reference
validation and JSON serialization. Memory reports lifetime peak process RSS,
current GPU pool used/reserved bytes, and device free bytes; pool sizes are not
peak GPU allocation measurements. CPU runs use the recorded HCIPy default
thread policy; grids within one process share context/pool state.

This enables the FFT stage needed by `SpectralNoiseFactoryFFT`. The next task is
to make that factory's sampling, coefficient translation and synthesis resident,
with seeded copy/reset and spectral statistics tests. The finite atmosphere uses
a multiscale FFT/MFT model; replacing it with a periodic FFT screen would change
its low-frequency statistics and is not part of this milestone.
