"""Gaussian beam propagation and synchronized, machine-readable benchmarks.

Run from the repository root, for example:
CUDA_VISIBLE_DEVICES=0 python examples/gaussian_fresnel_backend.py --backend cupy
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


def run(backend='numpy', size=256, repeats=20, precision='complex128', padding=2):
    """Return timings, physical measurements and numerical errors for one case."""
    xp = importlib.import_module(backend)

    def synchronize():
        if backend == 'cupy':
            xp.cuda.get_current_stream().synchronize()

    # Initialize the CUDA context separately; record its cost rather than hiding it.
    start = perf_counter()
    synchronize()
    context_seconds = perf_counter() - start
    start = perf_counter()
    waist, wavelength, distance, extent, radius = 0.001, 1e-6, 0.1, 0.02, 0.001
    grid = hp.make_pupil_grid(size, extent)
    source = hp.GaussianBeam(waist, 0, wavelength)(grid)
    source.total_power = 1  # W; electric field amplitude in sqrt(W)/m.
    host_field = np.asarray(source.electric_field).astype(precision)
    host_aperture = np.asarray(hp.evaluate_supersampled(hp.make_circular_aperture(2 * radius), grid, 8))
    prop = hp.FresnelPropagator(grid, distance, num_oversampling=1, zero_padding=padding)
    setup_seconds = perf_counter() - start

    start = perf_counter()
    data = xp.asarray(host_field)
    aperture = xp.asarray(host_aperture, dtype=getattr(xp, 'float32' if precision == 'complex64' else 'float64'))
    synchronize()
    upload_seconds = perf_counter() - start
    wavefront = hp.Wavefront(NewStyleField(data, grid), wavelength)

    start = perf_counter()
    result = prop(wavefront)
    synchronize()
    first_call_seconds = perf_counter() - start  # CPU kernel setup, kernel upload, FFT planning/execution.

    start = perf_counter()
    for _ in range(3):
        result = prop(wavefront)
        power = xp.sum(result.power.data * aperture)
    synchronize()
    warmup_seconds = perf_counter() - start

    samples = []
    for _ in range(repeats):
        synchronize()
        start = perf_counter()
        result = prop(wavefront)
        power = xp.sum(result.power.data * aperture)
        synchronize()
        samples.append(perf_counter() - start)

    start = perf_counter()
    actual = to_numpy(result.electric_field.data)
    measured_power = float(power)
    total_power = float(result.total_power)
    synchronize()
    download_seconds = perf_counter() - start

    zr = np.pi * waist**2 / wavelength
    q = 1 + 1j * distance / zr
    reference = np.sqrt(2 / (np.pi * waist**2)) / q * np.exp(2j * np.pi / wavelength * distance) * np.exp(
        -(grid.x**2 + grid.y**2) / (waist**2 * q))
    expected_power = 1 - np.exp(-2 * radius**2 / (waist**2 * abs(q)**2))
    memory = {'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if backend == 'cupy':
        memory.update(gpu_pool_used_bytes=xp.get_default_memory_pool().used_bytes(),
                      gpu_pool_reserved_bytes=xp.get_default_memory_pool().total_bytes(),
                      gpu_device_free_bytes=xp.cuda.runtime.memGetInfo()[0])
    # A CPU execution at identical sampling/precision is an independent backend comparison.
    cpu_result = hp.FresnelPropagator(grid, distance, num_oversampling=1, zero_padding=padding)(
        hp.Wavefront(hp.Field(host_field, grid), wavelength))
    return dict(
        backend=backend, size=size, precision=precision, repeats=repeats, zero_padding=padding,
        parameters_si=dict(waist_m=waist, wavelength_m=wavelength, distance_m=distance,
                           grid_extent_m=extent, aperture_radius_m=radius, input_power_W=1),
        sampling=dict(pixels_per_waist=waist / grid.delta[0], transfer_oversampling=1,
                      aperture_oversampling=8, fresnel_branch_threshold_m=extent**2 / (size * wavelength)),
        timings_s=dict(context=context_seconds, host_setup=setup_seconds, input_upload=upload_seconds,
                       first_call=first_call_seconds, warmup=warmup_seconds,
                       steady_median=float(np.median(samples)), steady_min=min(samples),
                       steady_max=max(samples), output_download=download_seconds,
                       end_to_end=context_seconds + setup_seconds + upload_seconds + first_call_seconds
                       + warmup_seconds + sum(samples) + download_seconds),
        aperture_power_W=measured_power, analytic_aperture_power_W=expected_power,
        aperture_absolute_error_W=abs(measured_power - expected_power), total_power_W=total_power,
        relative_field_error=float(np.linalg.norm(actual - reference) / np.linalg.norm(reference)),
        relative_cpu_difference=float(np.linalg.norm(actual - cpu_result.electric_field) / np.linalg.norm(cpu_result.electric_field)),
        memory=memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['numpy', 'cupy', 'array_api_strict'], default='numpy')
    parser.add_argument('--sizes', type=int, nargs='+', default=[256, 512, 1024])
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--precision', choices=['complex64', 'complex128'], default='complex128')
    parser.add_argument('--padding', type=int, default=2)
    parser.add_argument('--output', help='Optional JSON output path.')
    args = parser.parse_args()
    if args.repeats < 1 or min(args.sizes) < 2 or args.padding < 1:
        parser.error('repeats/padding must be positive and grid sizes at least two')
    hp.Configuration().reset(enable_user_overrides=False)
    xp = importlib.import_module(args.backend)
    metadata = dict(platform=platform.platform(), processor=platform.processor(), python=platform.python_version(),
                    numpy=np.__version__, scipy=scipy.__version__, backend_version=xp.__version__,
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    working_tree_dirty=bool(subprocess.check_output(['git', 'diff', '--name-only'], text=True).strip()),
                    configuration=hp.Configuration().model_dump())
    if args.backend == 'cupy':
        metadata.update(gpu=xp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                        cuda_runtime=xp.cuda.runtime.runtimeGetVersion(),
                        cuda_driver=xp.cuda.runtime.driverGetVersion())
    report = dict(metadata=metadata, cases=[run(args.backend, n, args.repeats, args.precision, args.padding) for n in args.sizes])
    text = json.dumps(report, indent=2, default=str)
    if args.output:
        with open(args.output, 'w') as output:
            output.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
