"""Periodic spectral noise: normalization, translation and resident execution."""
import numpy as np
import pytest

import hcipy as hp
from hcipy.field import NewStyleField
from hcipy._math.backends import to_numpy
from hcipy._math.random import make_random_generator
from conftest import get_backend


@pytest.fixture(params=['numpy', 'cupy', 'array_api_strict'])
def backend(request):
    return get_backend(request.param)


def psd(grid):
    return hp.power_spectral_density_von_karman(0.01, 0.02)(grid)


@pytest.mark.parametrize('oversample', [1, 2])
def test_legacy_seed_and_period(oversample):
    grid = hp.make_uniform_grid([12, 8], [0.08, 0.06])
    factory = hp.SpectralNoiseFactoryFFT(psd, grid, oversample)
    noise = factory.make_random(12)
    rng = np.random.default_rng(12)
    expected = factory.C * (rng.standard_normal(factory.input_grid.size) + 1j * rng.standard_normal(factory.input_grid.size))
    np.testing.assert_array_equal(noise.C, expected)
    np.testing.assert_allclose(factory.period, np.array([0.08, 0.06]) * oversample)
    original = noise().copy()
    for axis in (0, 1):
        shift = [0, 0]
        shift[axis] = factory.period[axis]
        np.testing.assert_allclose(noise.shifted(shift)(), original, atol=1e-12)


@pytest.mark.parametrize('axis', [0, 1])
def test_legacy_shift_axes(axis):
    grid = hp.make_uniform_grid([12, 8], [0.08, 0.06])
    noise = hp.SpectralNoiseFactoryFFT(psd, grid).make_random(12)
    original = noise().shaped.copy()
    shift = [0, 0]
    shift[axis] = grid.delta[axis]
    shifted = noise.shifted(shift)().shaped
    np.testing.assert_allclose(shifted, np.roll(original, 1, axis=1 - axis), atol=1e-12)
    np.testing.assert_array_equal(noise().shaped, original)


@pytest.mark.parametrize('dtype,tolerance', [('float32', 3e-6), ('float64', 3e-12)])
@pytest.mark.parametrize('oversample', [1, 2])
def test_shared_coefficients_and_translation(backend, dtype, tolerance, oversample):
    grid = hp.make_uniform_grid([12, 8], [0.08, 0.06])
    reference_factory = hp.SpectralNoiseFactoryFFT(psd, grid, oversample)
    factory = hp.SpectralNoiseFactoryFFT(psd, grid, oversample, xp=backend, dtype=getattr(backend, dtype))
    reference = reference_factory.make_random(23)
    complex_dtype = backend.complex64 if dtype == 'float32' else backend.complex128
    coefficients = backend.asarray(np.asarray(reference.C), dtype=complex_dtype)
    noise = hp.SpectralNoiseFFT(factory, NewStyleField(coefficients, factory.input_grid))
    for shift in ([0, 0], [0.0007, -0.0013], [0.002, 0]):
        # Direct Fourier quadrature avoids using the shift implementation as
        # its own reference, and tests signs and axis order on a rectangle.
        k = factory.input_grid
        kernel = np.exp(1j * ((np.asarray(grid.x)[:, None] - shift[0]) * np.asarray(k.x)[None, :]
                              + (np.asarray(grid.y)[:, None] - shift[1]) * np.asarray(k.y)[None, :]))
        expected = np.real(kernel @ np.asarray(reference.C)) * k.weights / (2 * np.pi)**2
        result = noise.shifted(shift)()
        assert result.dtype == getattr(backend, dtype)
        assert np.linalg.norm(to_numpy(result.data) - expected) / np.linalg.norm(expected) < tolerance
    # Copying an already-evaluated realization must own independent coefficients.
    before = to_numpy(noise().data).copy()
    copied = noise.copy()
    copied.shift([grid.delta[0], 0])
    np.testing.assert_array_equal(to_numpy(noise().data), before)
    assert not np.allclose(to_numpy(copied().data), before)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_seeded_stream_replay(backend, dtype):
    grid = hp.make_pupil_grid(16, 0.08)
    factory = hp.SpectralNoiseFactoryFFT(psd, grid, xp=backend, dtype=getattr(backend, dtype))
    first = factory.make_random(17)
    replay = factory.make_random(17)
    assert first().dtype == getattr(backend, dtype)
    np.testing.assert_array_equal(to_numpy(first.C.data), to_numpy(replay.C.data))
    assert not np.allclose(to_numpy(first().data), to_numpy(factory.make_random(18)().data))
    rng = make_random_generator(backend, seed=19)
    factory.make_random(rng)
    clone = rng.copy()
    np.testing.assert_array_equal(to_numpy(factory.make_random(rng).C.data), to_numpy(factory.make_random(clone).C.data))


def test_backend_period_and_device_shift(backend):
    grid = hp.make_uniform_grid([12, 8], [0.08, 0.06])
    factory = hp.SpectralNoiseFactoryFFT(psd, grid, 2, xp=backend)
    noise = factory.make_random(17)
    original = to_numpy(noise().data).copy()
    noise.shift(backend.asarray(factory.period))
    np.testing.assert_allclose(to_numpy(noise().data), original, atol=1e-12)


@pytest.mark.parametrize('dtype,tolerance', [('float32', 5e-6), ('float64', 3e-12)])
def test_shift_composition_and_reversal(backend, dtype, tolerance):
    grid = hp.make_pupil_grid(16, 0.08)
    factory = hp.SpectralNoiseFactoryFFT(psd, grid, xp=backend, dtype=getattr(backend, dtype))
    noise = factory.make_random(17)
    initial = to_numpy(noise().data).copy()
    expected = to_numpy(noise.shifted([0.002, -0.004])().data)
    for _ in range(20):
        noise.shift([0.0001, -0.0002])
    actual = to_numpy(noise().data)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < tolerance
    noise.shift([-0.002, 0.004])
    assert np.linalg.norm(to_numpy(noise().data) - initial) / np.linalg.norm(initial) < tolerance


def test_invalid_backend_options():
    grid = hp.make_pupil_grid(8)
    with pytest.raises(ValueError, match='Specify xp'):
        hp.SpectralNoiseFactoryFFT(psd, grid, dtype=np.float32)
    with pytest.raises(ValueError, match='dtype must be'):
        hp.SpectralNoiseFactoryFFT(psd, grid, xp=np, dtype=np.int32)
    strict = get_backend('array_api_strict')
    factory = hp.SpectralNoiseFactoryFFT(psd, grid, xp=strict)
    with pytest.raises(ValueError, match='factory backend'):
        factory.make_random(make_random_generator(np, seed=12))


@pytest.mark.parametrize('dims', [(9,), (4, 5, 6)])
def test_other_dimensions(backend, dims):
    grid = hp.make_uniform_grid(dims, [0.08] * len(dims))
    factory = hp.SpectralNoiseFactoryFFT(lambda g: hp.Field(np.ones(g.size), g), grid, xp=backend)
    noise = factory.make_random(7)
    original = to_numpy(noise().data).reshape(grid.shape)
    for axis in range(len(dims)):
        shift = [0] * len(dims)
        shift[axis] = grid.delta[axis]
        actual = to_numpy(noise.shifted(shift)().data).reshape(grid.shape)
        np.testing.assert_allclose(actual, np.roll(original, 1, axis=len(dims) - axis - 1), atol=1e-11)


def test_spectral_power_ensemble(backend):
    grid = hp.make_uniform_grid([32, 24], [0.08, 0.06])
    factory = hp.SpectralNoiseFactoryFFT(psd, grid, xp=backend)
    frequency_grid = factory.input_grid
    target = np.asarray(psd(frequency_grid)).reshape(frequency_grid.shape)
    delta_f_area = frequency_grid.weights / (2 * np.pi)**2
    radius = np.hypot(frequency_grid.x, frequency_grid.y).shaped / (2 * np.pi)
    bands = [(radius >= low) & (radius < high) for low, high in [(20, 50), (50, 90), (90, 140)]]
    measurements = []
    for seed in range(64):
        screen = to_numpy(factory.make_random(1000 + seed)().data).reshape(grid.shape)
        # DFT periodogram in the PSD's cycles/m convention; no fitted scale.
        periodogram = abs(np.fft.fftshift(np.fft.fft2(screen)))**2 / (grid.size**2 * delta_f_area)
        measurements.append([np.mean(screen**2)] + [np.mean(periodogram[band]) for band in bands])
    measurements = np.asarray(measurements)
    expected = np.array([np.sum(target) * delta_f_area] + [np.mean(target[band]) for band in bands])
    sem = measurements.std(axis=0, ddof=1) / np.sqrt(len(measurements))
    # Independent screens, six estimated standard errors, and 1% numerical
    # allowance. This checks absolute normalization and spectral shape.
    assert np.all(abs(measurements.mean(axis=0) - expected) < 6 * sem + 0.01 * expected)


def test_sampling_shift_and_synthesis_residency(monkeypatch):
    cp = get_backend('cupy')
    grid = hp.make_pupil_grid(32, 0.08)
    factory = hp.SpectralNoiseFactoryFFT(psd, grid, xp=cp)
    noise = factory.make_random(12)
    noise()  # Warm the FFT cache.
    rng = make_random_generator(cp, seed=13)
    asarray = cp.asarray

    def resident_asarray(value, *args, **kwargs):
        assert not isinstance(value, np.ndarray), 'Unexpected host array upload'
        return asarray(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cp, 'asarray', resident_asarray)
        for _ in range(3):
            noise = factory.make_random(rng)
            noise.shift([0.001, -0.002])
            result = noise()
            assert isinstance(noise.C.data, cp.ndarray)
            assert isinstance(result.data, cp.ndarray)
