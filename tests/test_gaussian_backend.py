"""Scalar propagation with CPU setup grids and resident backend arrays."""
import numpy as np
import pytest

import hcipy as hp
from hcipy.field import NewStyleField
from hcipy._math.backends import to_numpy
from conftest import get_backend


@pytest.fixture(params=['numpy', 'cupy', 'array_api_strict'])
def backend(request):
    return get_backend(request.param)


@pytest.mark.parametrize('dtype,tolerance', [('complex64', 2e-6), ('complex128', 2e-12)])
@pytest.mark.parametrize('padding', [1, 2])
def test_gaussian_propagation(backend, dtype, tolerance, padding):
    grid = hp.make_pupil_grid(128, 0.02)
    waist, wavelength, distance = 0.001, 1e-6, 0.5
    source = hp.GaussianBeam(waist, 0, wavelength)(grid)
    source.total_power = 1
    data = backend.asarray(np.asarray(source.electric_field), dtype=getattr(backend, dtype))
    wavefront = hp.Wavefront(NewStyleField(data, grid), wavelength)
    prop = hp.FresnelPropagator(grid, distance, num_oversampling=1, zero_padding=padding)
    result = prop(wavefront)

    # Positive-phase Fresnel convention, independently derived from the paraxial solution.
    zr = np.pi * waist**2 / wavelength
    q = 1 + 1j * distance / zr
    reference = np.sqrt(2 / (np.pi * waist**2)) / q * np.exp(2j * np.pi / wavelength * distance) * np.exp(
        -(grid.x**2 + grid.y**2) / (waist**2 * q))
    actual = to_numpy(result.electric_field.data)
    assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < tolerance
    assert abs(float(result.total_power) - 1) < tolerance
    assert result.electric_field.dtype == getattr(backend, dtype)
    np.testing.assert_allclose(to_numpy(result.intensity.data), abs(actual)**2, rtol=tolerance)

    aperture = backend.asarray(np.asarray(hp.evaluate_supersampled(hp.make_circular_aperture(0.002), grid, 8)))
    measured = float(backend.sum(result.power.data * aperture))
    radius = waist * np.sqrt(1 + (distance / zr)**2)
    # Pixel quadrature error dominates, unlike the smooth-field FFT error.
    assert abs(measured - (1 - np.exp(-2 * (0.001 / radius)**2))) < 0.004

    recovered = prop.backward(result)
    assert np.linalg.norm(to_numpy(recovered.electric_field.data) - to_numpy(data)) / np.linalg.norm(to_numpy(data)) < tolerance
    np.testing.assert_array_equal(to_numpy(wavefront.electric_field.data), to_numpy(data))
    wavefront.total_power = 2
    assert abs(float(wavefront.total_power) - 2) < 2 * tolerance


def test_gaussian_grid_padding_convergence():
    errors = []
    for n in (64, 128, 256):
        grid = hp.make_pupil_grid(n, 0.02)
        wavefront = hp.GaussianBeam(0.001, 0, 1e-6)(grid)
        wavefront.total_power = 1
        outputs = [hp.FresnelPropagator(grid, 0.5, num_oversampling=1, zero_padding=q)(wavefront) for q in (1, 2, 4)]
        np.testing.assert_allclose(outputs[1].intensity, outputs[2].intensity, atol=1e-8, rtol=1e-10)
        aperture = hp.evaluate_supersampled(hp.make_circular_aperture(0.002), grid, 8)
        exact = 1 - np.exp(-2 / (1 + (0.5 / np.pi)**2))
        errors.append(abs(np.sum(outputs[2].power * aperture) - exact))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.001


def test_resident_filter_cache_and_backend_switch(monkeypatch):
    cp = get_backend('cupy')
    grid = hp.make_pupil_grid(64, 0.02)
    source = hp.GaussianBeam(0.001, 0, 1e-6)(grid)
    prop = hp.FresnelPropagator(grid, 0.5, num_oversampling=1)
    gpu = hp.Wavefront(NewStyleField(cp.asarray(np.asarray(source.electric_field)), grid), 1e-6)
    expected = prop(source)
    result = prop(gpu)  # Same instance switches from CPU to GPU.
    saved = result.electric_field.data.copy()

    def no_host_work(*args, **kwargs):
        raise AssertionError('Repeated propagation must not invoke CPU array conversion or kernel evaluation.')

    filt = prop.get_instance_data(grid, None, 1e-6).fourier_filter
    kernel, scratch = filt._transfer_function, filt.internal_array
    asarray = cp.asarray

    def resident_asarray(a, *args, **kwargs):
        if isinstance(a, np.ndarray):
            no_host_work()
        return asarray(a, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(filt, 'transfer_function', no_host_work)
        patch.setattr(cp, 'asarray', resident_asarray)
        for _ in range(3):
            result = prop(gpu)
            power = result.total_power
            assert isinstance(power, cp.ndarray)
            assert isinstance(result.electric_field.data, cp.ndarray)
            assert filt._transfer_function is kernel
            assert filt.internal_array is scratch
    cp.testing.assert_array_equal(saved, result.electric_field.data)
    np.testing.assert_allclose(cp.asnumpy(saved), expected.electric_field, atol=1e-14)
    # Changing precision and switching back must invalidate both kernel and scratch.
    gpu.electric_field = NewStyleField(cp.asarray(np.asarray(source.electric_field), dtype=cp.complex64), grid)
    assert prop(gpu).electric_field.dtype == cp.complex64
    np.testing.assert_allclose(prop(source).electric_field, expected.electric_field, atol=1e-14)


@pytest.mark.parametrize('distance', [0.5, 5])
def test_fresnel_setup_branches(backend, distance):
    grid = hp.make_pupil_grid(128, 0.02)
    source = hp.GaussianBeam(0.001, 0, 1e-6)(grid)
    prop = hp.FresnelPropagator(grid, distance)  # Default oversampling; both setup branches.
    expected = prop(source)
    actual = prop(hp.Wavefront(NewStyleField(backend.asarray(np.asarray(source.electric_field)), grid), 1e-6))
    np.testing.assert_allclose(to_numpy(actual.electric_field.data), expected.electric_field, atol=1e-13)
    reference = hp.GaussianBeam(0.001, distance, 1e-6)(grid).intensity
    # Default cell-averaged kernels introduce attenuation; this is a discretization
    # tolerance, not a floating-point tolerance or an exact unitary propagator.
    assert np.linalg.norm(to_numpy(actual.intensity.data) - reference) / np.linalg.norm(reference) < 0.004


def test_chained_propagation(backend):
    grid = hp.make_pupil_grid(128, 0.02)
    source = hp.GaussianBeam(0.001, 0, 1e-6)(grid)
    wavefront = hp.Wavefront(NewStyleField(backend.asarray(np.asarray(source.electric_field)), grid), 1e-6)
    prop = hp.FresnelPropagator(grid, 0.1, num_oversampling=1)
    for _ in range(5):
        wavefront = prop(wavefront)
    expected = hp.FresnelPropagator(grid, 0.5, num_oversampling=1)(source)
    # Accumulated longitudinal-phase roundoff is O(eps * k * distance).
    np.testing.assert_allclose(to_numpy(wavefront.electric_field.data), expected.electric_field, atol=1e-9)
