"""Gaussian Fourier pair with synchronized resident FFT benchmarks.

CUDA_VISIBLE_DEVICES=0 python examples/fft_backend.py --backend cupy
"""
import argparse
import importlib
import json
import platform
import resource
import subprocess
from time import perf_counter

import numpy as np
import scipy
import hcipy as hp
from hcipy.field import NewStyleField
from hcipy._math.backends import to_numpy


def run(xp, size, precision, repeats, padding, emulate):
    def synchronize():
        if xp.__name__ == 'cupy':
            xp.cuda.get_current_stream().synchronize()

    def timed(operation):
        synchronize()
        start = perf_counter()
        result = operation()
        synchronize()
        return result, perf_counter() - start

    start = perf_counter()
    synchronize()
    context_seconds = perf_counter() - start
    begin = perf_counter()
    waist, extent = 0.001, 0.02
    grid = hp.make_pupil_grid(size, extent)
    host = np.asarray(np.exp(-(grid.x**2 + grid.y**2) / waist**2)).astype(precision)
    fft = hp.FastFourierTransform(grid, q=padding, emulate_fftshifts=emulate)
    setup_seconds = perf_counter() - begin
    data, upload_seconds = timed(lambda: xp.asarray(host))
    field = NewStyleField(data, grid)
    spectrum, first_forward = timed(lambda: fft.forward(field))
    _, first_backward = timed(lambda: fft.backward(spectrum))

    def roundtrip():
        return fft.backward(fft.forward(field))

    def warmup():
        for _ in range(3):
            roundtrip()

    _, warmup_seconds = timed(warmup)
    forward_samples, backward_samples = [], []
    for _ in range(repeats):
        spectrum, elapsed = timed(lambda: fft.forward(field))
        forward_samples.append(elapsed)
        recovered, elapsed = timed(lambda: fft.backward(spectrum))
        backward_samples.append(elapsed)
    (actual, restored), download_seconds = timed(lambda: (to_numpy(spectrum.data), to_numpy(recovered.data)))
    end_to_end_seconds = perf_counter() - begin + context_seconds
    memory = dict(process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if xp.__name__ == 'cupy':
        memory.update(gpu_pool_used_bytes=xp.get_default_memory_pool().used_bytes(),
                      gpu_pool_reserved_bytes=xp.get_default_memory_pool().total_bytes(),
                      gpu_device_free_bytes=xp.cuda.runtime.memGetInfo()[0])
    reference = np.pi * waist**2 * np.exp(-waist**2 * (fft.output_grid.x**2 + fft.output_grid.y**2) / 4)
    cpu = fft.forward(hp.Field(host, grid))
    power_in = np.sum(abs(host)**2) * grid.weights
    power_out = np.sum(abs(actual)**2) * fft.output_grid.weights / (2 * np.pi)**2
    return dict(size=size, precision=precision, padding=padding, emulate_fftshifts=emulate, repeats=repeats,
                parameters=dict(waist_m=waist, extent_m=extent, input_peak_amplitude=1),
                timings_s=dict(context=context_seconds, host_setup=setup_seconds, upload=upload_seconds,
                               first_forward=first_forward, first_backward=first_backward, warmup=warmup_seconds,
                               forward_median=float(np.median(forward_samples)), backward_median=float(np.median(backward_samples)),
                               forward_min=min(forward_samples), forward_max=max(forward_samples),
                               backward_min=min(backward_samples), backward_max=max(backward_samples),
                               download=download_seconds, end_to_end=end_to_end_seconds),
                relative_analytic_error=float(np.linalg.norm(actual - reference) / np.linalg.norm(reference)),
                relative_cpu_difference=float(np.linalg.norm(actual - cpu) / np.linalg.norm(cpu)),
                relative_roundtrip_error=float(np.linalg.norm(restored - host) / np.linalg.norm(host)),
                relative_parseval_error=float(abs(power_out / power_in - 1)), memory=memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['numpy', 'cupy', 'array_api_strict'], default='numpy')
    parser.add_argument('--sizes', nargs='+', type=int, default=[256, 512, 1024])
    parser.add_argument('--precision', choices=['complex64', 'complex128'], default='complex128')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--padding', type=int, default=2)
    parser.add_argument('--emulate-fftshifts', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--output', help='Optional JSON report path.')
    args = parser.parse_args()
    if min(args.sizes) < 2 or args.repeats < 1 or args.padding < 1:
        parser.error('sizes >= 2 and positive repeats/padding required')
    hp.Configuration().reset(enable_user_overrides=False)
    xp = importlib.import_module(args.backend)
    metadata = dict(backend=args.backend, backend_version=xp.__version__, numpy=np.__version__, scipy=scipy.__version__,
                    python=platform.python_version(), platform=platform.platform(), configuration=hp.Configuration().model_dump(),
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    tracked_changes=subprocess.check_output(['git', 'diff', '--name-only'], text=True).splitlines())
    if args.backend == 'cupy':
        metadata.update(gpu=xp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                        cuda_runtime=xp.cuda.runtime.runtimeGetVersion(), cuda_driver=xp.cuda.runtime.driverGetVersion())
    report = dict(metadata=metadata, cases=[run(xp, n, args.precision, args.repeats, args.padding, args.emulate_fftshifts) for n in args.sizes])
    text = json.dumps(report, indent=2, default=str)
    if args.output:
        with open(args.output, 'w') as output:
            output.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
