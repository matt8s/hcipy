"""Resident periodic turbulence sampling, frozen flow and optical propagation.

CUDA_VISIBLE_DEVICES=0 python examples/spectral_noise_backend.py --backend cupy
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
from hcipy._math.random import make_random_generator


def run(xp, size, precision, seed, repeats, oversample):
    def synchronize():
        if xp.__name__ == 'cupy':
            xp.cuda.get_current_stream().synchronize()

    def timed(operation):
        synchronize()
        start = perf_counter()
        value = operation()
        synchronize()
        return value, perf_counter() - start

    start = perf_counter()
    synchronize()
    context_seconds = perf_counter() - start
    begin = perf_counter()
    grid = hp.make_pupil_grid(size, 0.02)
    wavelength, waist, r0, outer_scale, distance, dt = 1e-6, 0.001, 0.003, 0.02, 0.1, 0.01
    source = hp.GaussianBeam(waist, 0, wavelength)(grid)
    source.total_power = 1
    complex_dtype = np.complex64 if precision == 'float32' else np.complex128
    host_field = np.asarray(source.electric_field).astype(complex_dtype)
    host_aperture = np.asarray(hp.evaluate_supersampled(hp.make_circular_aperture(0.002), grid, 8))
    spectrum = hp.power_spectral_density_von_karman(r0, outer_scale)
    prop = hp.FresnelPropagator(grid, distance, num_oversampling=1, zero_padding=2)
    host_setup = perf_counter() - begin
    factory, factory_setup = timed(lambda: hp.SpectralNoiseFactoryFFT(
        spectrum, grid, oversample, xp=xp, dtype=getattr(xp, precision)))
    rng = make_random_generator(xp, seed=seed)
    noise, first_sampling = timed(lambda: factory.make_random(rng))
    phase, first_synthesis = timed(noise)
    (field, aperture), source_upload = timed(lambda: (xp.asarray(host_field), xp.asarray(host_aperture)))
    wavefront = hp.Wavefront(NewStyleField(field, grid), wavelength)

    def propagate(phase):
        screened = hp.PhaseApodizer(phase)(wavefront)
        output = prop(screened)
        return output, xp.sum(output.power.data * aperture), screened.total_power, output.total_power

    _, first_optics = timed(lambda: propagate(phase))

    def warmup():
        for _ in range(3):
            noise.shift([0, 0])
            propagate(noise())

    _, warmup_seconds = timed(warmup)
    shift_times, synthesis_times, optical_times = [], [], []
    # Fresh input beam for each evolving screen; no repeated traversal by one beam.
    for _ in range(repeats):
        _, elapsed = timed(lambda: noise.shift([0.01 * dt, -0.005 * dt]))
        shift_times.append(elapsed)
        phase, elapsed = timed(noise)
        synthesis_times.append(elapsed)
        (output, received, phase_power, total_power), elapsed = timed(lambda: propagate(phase))
        optical_times.append(elapsed)
    # Measure fresh independent coefficient sampling separately from frozen flow.
    sample_times = []
    for _ in range(repeats):
        fresh, elapsed = timed(lambda: factory.make_random(rng))
        sample_times.append(elapsed)
    del fresh
    downloaded, download_seconds = timed(lambda: (to_numpy(noise.C.data), to_numpy(phase.data),
                                                  to_numpy(output.electric_field.data), float(received),
                                                  float(phase_power), float(total_power)))
    end_to_end = perf_counter() - begin + context_seconds
    coefficients, actual_phase, actual_field, received, phase_power, total_power = downloaded
    memory = dict(process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if xp.__name__ == 'cupy':
        memory.update(gpu_pool_used_bytes=xp.get_default_memory_pool().used_bytes(),
                      gpu_pool_reserved_bytes=xp.get_default_memory_pool().total_bytes(),
                      gpu_device_free_bytes=xp.cuda.runtime.memGetInfo()[0])
    # Shared *coefficients*, not equal seeds: independent backend streams differ.
    cpu_factory = hp.SpectralNoiseFactoryFFT(spectrum, grid, oversample)
    reference_phase = hp.SpectralNoiseFFT(cpu_factory, hp.Field(coefficients, cpu_factory.input_grid))()
    reference_field = hp.FresnelPropagator(grid, distance, num_oversampling=1)(
        hp.PhaseApodizer(reference_phase)(hp.Wavefront(hp.Field(host_field, grid), wavelength))).electric_field

    def summary(values):
        return dict(median=float(np.median(values)), minimum=min(values), maximum=max(values))

    return dict(size=size, precision=precision, seed=seed, repeats=repeats, oversample=oversample,
                parameters_si=dict(wavelength_m=wavelength, waist_m=waist, input_power_W=1, extent_m=0.02,
                                   fried_parameter_m=r0, outer_scale_m=outer_scale, propagation_m=distance,
                                   receiver_radius_m=0.001, velocity_m_per_s=[0.01, -0.005], dt_s=dt),
                period_m=factory.period.tolist(), final_time_s=repeats * dt,
                timings_s=dict(context=context_seconds, host_setup=host_setup, factory_setup_and_upload=factory_setup,
                               first_sampling=first_sampling, first_synthesis=first_synthesis, source_upload=source_upload,
                               first_optics=first_optics, warmup=warmup_seconds, shift=summary(shift_times),
                               synthesis=summary(synthesis_times), optics=summary(optical_times),
                               fresh_sampling=summary(sample_times), download=download_seconds, end_to_end=end_to_end),
                aperture_power_W=received, phase_only_power_W=phase_power, output_power_W=total_power,
                relative_cpu_phase_error=float(np.linalg.norm(actual_phase - reference_phase) / np.linalg.norm(reference_phase)),
                relative_cpu_field_error=float(np.linalg.norm(actual_field - reference_field) / np.linalg.norm(reference_field)),
                memory=memory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['numpy', 'cupy', 'array_api_strict'], default='numpy')
    parser.add_argument('--sizes', type=int, nargs='+', default=[256, 512, 1024])
    parser.add_argument('--precision', choices=['float32', 'float64'], default='float64')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--oversample', type=int, default=2)
    parser.add_argument('--output', help='Optional JSON manifest and timing report.')
    args = parser.parse_args()
    if min(args.sizes) < 2 or args.repeats < 1 or args.oversample < 1:
        parser.error('sizes >= 2 and positive repeats/oversample required')
    hp.Configuration().reset(enable_user_overrides=False)
    xp = importlib.import_module(args.backend)
    metadata = dict(backend=args.backend, backend_version=xp.__version__, python=platform.python_version(),
                    platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
                    configuration=hp.Configuration().model_dump(),
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    tracked_changes=subprocess.check_output(['git', 'diff', '--name-only'], text=True).splitlines())
    if args.backend == 'cupy':
        metadata.update(gpu=xp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                        cuda_runtime=xp.cuda.runtime.runtimeGetVersion(), cuda_driver=xp.cuda.runtime.driverGetVersion())
    report = dict(metadata=metadata, cases=[run(xp, n, args.precision, args.seed, args.repeats, args.oversample) for n in args.sizes])
    text = json.dumps(report, indent=2, default=str)
    if args.output:
        with open(args.output, 'w') as output:
            output.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
