"""Seeded frozen-flow atmosphere, resident phase snapshots and Fresnel propagation.

CUDA_VISIBLE_DEVICES=0 python examples/phase_screen_fresnel_backend.py --backend cupy
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


def run(xp, size, precision, seed, frames, dt, repeats):
    """Measure host evolution and device execution separately, returning SI data."""
    gpu = xp.__name__ == 'cupy'

    def synchronize():
        if gpu:
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
    grid = hp.make_pupil_grid(size, 0.02)
    wavelength, waist, r0, outer_scale, distance = 1e-6, 0.001, 0.003, 0.02, 0.1
    strength = hp.Cn_squared_from_fried_parameter(r0, wavelength)
    layer = hp.FiniteAtmosphericLayer(grid, strength, outer_scale, velocity=[0.01, -0.005], seed=seed)
    source = hp.GaussianBeam(waist, 0, wavelength)(grid)
    source.total_power = 1
    host_field = np.asarray(source.electric_field).astype(precision)
    host_aperture = np.asarray(hp.evaluate_supersampled(hp.make_circular_aperture(0.002), grid, 8))
    prop = hp.FresnelPropagator(grid, distance, num_oversampling=1, zero_padding=2)
    setup_seconds = perf_counter() - begin
    (data, aperture), upload_seconds = timed(lambda: (xp.asarray(host_field), xp.asarray(host_aperture)))
    wavefront = hp.Wavefront(NewStyleField(data, grid), wavelength)
    # Create the propagator kernel/FFT plan separately from per-frame execution.
    _, first_call_seconds = timed(lambda: prop(wavefront))
    frame_results = []
    for time in np.arange(frames) * dt:
        start = perf_counter()
        layer.evolve_until(float(time))
        host_phase = np.asarray(layer.phase_for(wavelength))
        generation_seconds = perf_counter() - start
        phase, transfer_seconds = timed(lambda: xp.asarray(host_phase, dtype=getattr(xp, precision)))
        snapshot = hp.PhaseApodizer(NewStyleField(phase, grid))

        def propagate_and_measure():
            screened = snapshot(wavefront)
            output = prop(screened)
            return output, xp.sum(output.power.data * aperture), screened.total_power, output.total_power

        (output, received, phase_power, output_power), compute_seconds = timed(propagate_and_measure)
        measurements, report_seconds = timed(lambda: tuple(float(v) for v in (received, phase_power, output_power)))
        frame_results.append(dict(time_s=float(time), aperture_power_W=measurements[0],
                                  phase_only_power_W=measurements[1], output_power_W=measurements[2],
                                  host_evolution_s=generation_seconds, phase_upload_s=transfer_seconds,
                                  propagation_measurement_s=compute_seconds, scalar_download_s=report_seconds))
    # Warm up and repeat the last frozen snapshot, without evolving/uploading it.
    def warmup():
        for _ in range(3):
            propagate_and_measure()

    _, warmup_seconds = timed(warmup)
    steady = []
    for _ in range(repeats):
        (output, _, _, _), elapsed = timed(propagate_and_measure)
        steady.append(elapsed)
    actual, download_seconds = timed(lambda: to_numpy(output.electric_field.data))
    end_to_end_seconds = perf_counter() - begin + context_seconds

    memory = dict(process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if gpu:
        memory.update(gpu_pool_used_bytes=xp.get_default_memory_pool().used_bytes(),
                      gpu_pool_reserved_bytes=xp.get_default_memory_pool().total_bytes(),
                      gpu_device_free_bytes=xp.cuda.runtime.memGetInfo()[0])
    reference = hp.FresnelPropagator(grid, distance, num_oversampling=1)(
        hp.PhaseApodizer(hp.Field(host_phase, grid))(hp.Wavefront(hp.Field(host_field, grid), wavelength)))
    relative_error = np.linalg.norm(actual - reference.electric_field) / np.linalg.norm(reference.electric_field)
    return dict(size=size, precision=precision, seed=seed, repeats=repeats, frames=frame_results,
                parameters_si=dict(wavelength_m=wavelength, waist_m=waist, input_power_W=1, grid_extent_m=0.02,
                                   fried_parameter_m=r0, outer_scale_m=outer_scale, integrated_Cn2_m_one_third=strength,
                                   velocity_m_per_s=[0.01, -0.005], distance_m=distance, receiver_radius_m=0.001),
                sampling=dict(phase_spectral_oversampling=2, transfer_oversampling=1, padding=2, aperture_oversampling=8),
                timings_s=dict(context=context_seconds, host_setup=setup_seconds, input_upload=upload_seconds,
                               first_propagation=first_call_seconds, warmup=warmup_seconds,
                               resident_median=float(np.median(steady)), resident_min=min(steady), resident_max=max(steady),
                               final_field_download=download_seconds, end_to_end=end_to_end_seconds),
                relative_cpu_field_error=float(relative_error), memory=memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['numpy', 'cupy', 'array_api_strict'], default='numpy')
    parser.add_argument('--sizes', type=int, nargs='+', default=[256, 512])
    parser.add_argument('--precision', choices=['complex64', 'complex128'], default='complex128')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--frames', type=int, default=4)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--output', help='Optional JSON manifest/results path.')
    args = parser.parse_args()
    if min(args.sizes) < 2 or args.frames < 1 or args.repeats < 1 or not np.isfinite(args.dt) or args.dt < 0:
        parser.error('sizes >= 2, positive frames/repeats, and finite nonnegative dt are required')
    hp.Configuration().reset(enable_user_overrides=False)
    xp = importlib.import_module(args.backend)
    metadata = dict(backend=args.backend, backend_version=xp.__version__, numpy=np.__version__, scipy=scipy.__version__,
                    python=platform.python_version(), platform=platform.platform(),
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    tracked_changes=subprocess.check_output(['git', 'diff', '--name-only'], text=True).splitlines(),
                    configuration=hp.Configuration().model_dump(), screen_generation_backend='numpy')
    if args.backend == 'cupy':
        metadata.update(gpu=xp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                        cuda_runtime=xp.cuda.runtime.runtimeGetVersion(), cuda_driver=xp.cuda.runtime.driverGetVersion())
    report = dict(metadata=metadata, cases=[run(xp, n, args.precision, args.seed, args.frames, args.dt, args.repeats) for n in args.sizes])
    text = json.dumps(report, indent=2, default=str)
    if args.output:
        with open(args.output, 'w') as output:
            output.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
