"""Profile the finite-screen MFT boundary and a resident matrix-product prototype.

This is an audit, not a backend implementation of MatrixFourierTransform.
Run with CUDA_VISIBLE_DEVICES=0 and explicit CPU thread environment variables.
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import hcipy as hp
from hcipy.field import NewStyleField
from threadpoolctl import threadpool_info


def run(size, precision, repeats, cp):
    """Measure existing CPU stages and optional CuPy inverse-MFT products."""
    begin = perf_counter()
    timings = {}

    def measure(name, operation, count=1, gpu=False):
        samples = []
        for _ in range(count):
            if gpu:
                cp.cuda.get_current_stream().synchronize()
            start = perf_counter()
            result = operation()
            if gpu:
                cp.cuda.get_current_stream().synchronize()
            samples.append(perf_counter() - start)
        timings[name] = dict(samples_s=samples, median_s=float(np.median(samples)))
        return result

    grid = hp.make_pupil_grid(size, 1)
    psd = hp.power_spectral_density_von_karman(0.15, 10, inner_scale=0.01)
    factory = measure('cpu_factory_setup', lambda: hp.SpectralNoiseFactoryMultiscale(psd, grid, 8))
    rng = np.random.default_rng(42)
    noise = measure('cpu_fresh_sampling', lambda: factory.make_random(rng), repeats)
    # Existing sampling is complex128; isolate MFT execution precision explicitly.
    coefficients = noise.C_2.astype(precision)
    mft = factory.fourier_2
    measure('cpu_matrix_setup', lambda: mft._compute_matrices(np.dtype(precision)))
    measure('cpu_first_mft', lambda: mft.backward(coefficients))
    measure('cpu_mft_warmup', lambda: mft.backward(coefficients), 3)
    reference = measure('cpu_mft', lambda: mft.backward(coefficients), repeats)
    measure('cpu_translation', lambda: noise.shift([0.001, -0.0005]), repeats)
    measure('cpu_first_synthesis', noise)
    measure('cpu_synthesis', noise, repeats)

    def evolve():
        noise.shift([0.001, -0.0005])
        return noise()

    measure('cpu_evolving_frame', evolve, repeats)
    # Independent positive-sign inverse quadrature at five spatial points.
    indices = np.linspace(0, grid.size - 1, 5, dtype=int)
    phase = (
        np.asarray(grid.x)[indices, None] * np.asarray(factory.input_grid_2.x)[None, :]
        + np.asarray(grid.y)[indices, None] * np.asarray(factory.input_grid_2.y)[None, :]
    )
    direct = np.exp(1j * phase) @ np.asarray(coefficients) * factory.input_grid_2.weights / (2 * np.pi) ** 2
    quadrature_error = float(np.linalg.norm(np.asarray(reference)[indices] - direct) / np.linalg.norm(direct))
    tolerance = 2e-5 if precision == 'complex64' else 1e-12
    if not np.isfinite(quadrature_error) or quadrature_error > tolerance:
        raise AssertionError(f'CPU quadrature error {quadrature_error} exceeds {tolerance}')

    result = dict(
        size=size,
        precision=precision,
        repeats=repeats,
        timings=timings,
        parameters=dict(extent_m=1, r0_m=0.15, outer_scale_m=10, inner_scale_m=0.01, oversampling=8, seed=42),
        cpu_sampling_and_synthesis_precision='complex128',
        cpu_relative_direct_quadrature_error=quadrature_error,
        cpu_max_absolute_direct_quadrature_error=float(np.max(np.abs(np.asarray(reference)[indices] - direct))),
    )
    if cp is not None:
        measure('gpu_pre_upload_sync', lambda: cp.cuda.get_current_stream().synchronize(), gpu=True)
        # CPU synthesis above may have replaced the matrices with complex128.
        mft._compute_matrices(np.dtype(precision))
        left, right, weights = measure(
            'gpu_matrix_upload', lambda: (cp.asarray(mft.M1.conj().T), cp.asarray(mft.M2.conj().T), cp.asarray(mft.weights_output)), gpu=True
        )
        data = measure('gpu_coefficients_upload', lambda: cp.asarray(np.asarray(coefficients)), gpu=True)

        def prototype():
            return ((left @ data.reshape(mft.shape_output)) @ right).ravel() * weights

        measure('gpu_first_mft_prototype', prototype, gpu=True)
        measure('gpu_mft_prototype_warmup', prototype, 3, gpu=True)
        output = measure('gpu_mft_prototype', prototype, repeats, gpu=True)
        host = measure('gpu_download', lambda: cp.asnumpy(output), gpu=True)
        error = float(np.linalg.norm(host - reference) / np.linalg.norm(reference))
        result['prototype_relative_cpu_error'] = error
        result['prototype_max_absolute_cpu_error'] = float(np.max(np.abs(host - reference)))
        if not np.isfinite(error) or error > tolerance:
            raise AssertionError(f'Prototype error {error} exceeds {tolerance}')
        try:
            mft.backward(NewStyleField(data, factory.input_grid_2))
        except (TypeError, ValueError, NotImplementedError) as exc:
            result['current_cupy_mft'] = dict(exception=type(exc).__name__, message=str(exc))
        else:
            result['current_cupy_mft'] = dict(exception=None)
        result['gpu_memory'] = dict(
            pool_used_bytes=cp.get_default_memory_pool().used_bytes(),
            pool_reserved_bytes=cp.get_default_memory_pool().total_bytes(),
            device_free_bytes=cp.cuda.runtime.memGetInfo()[0],
        )
    result['case_wall_s'] = perf_counter() - begin
    result['process_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return result


def main():
    """Write a JSON audit with raw timing samples and environment provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[256, 512])
    parser.add_argument('--precision', choices=['complex64', 'complex128'], default='complex128')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--cupy', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    if min(args.sizes) < 2 or args.repeats < 1:
        parser.error('sizes >= 2 and repeats >= 1 required')
    cp = None
    if args.cupy:
        import cupy as cp
    hp.Configuration().reset(enable_user_overrides=False)
    metadata = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        platform=platform.platform(),
        configuration=hp.Configuration().model_dump(),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        public_tree_status=subprocess.check_output(['git', 'status', '--short'], text=True).splitlines(),
        thread_environment={
            key: os.environ.get(key)
            for key in [
                'OPENBLAS_NUM_THREADS',
                'OMP_NUM_THREADS',
                'MKL_NUM_THREADS',
                'NUMBA_NUM_THREADS',
                'NUMEXPR_NUM_THREADS',
                'CUDA_VISIBLE_DEVICES',
            ]
        },
    )
    if cp is not None:
        metadata.update(
            cupy=cp.__version__,
            gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            cuda_driver=cp.cuda.runtime.driverGetVersion(),
        )
    cases = [run(n, args.precision, args.repeats, cp) for n in args.sizes]
    metadata['threadpools'] = threadpool_info()
    text = json.dumps(dict(metadata=metadata, cases=cases), indent=2, default=str)
    if args.output:
        with open(args.output, 'w') as stream:
            stream.write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
