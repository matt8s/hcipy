"""Seeded host atmosphere with portable, device-resident phase application."""
import numpy as np
import pytest

import hcipy as hp
from hcipy.field import NewStyleField
from hcipy._math.backends import to_numpy
from conftest import get_backend


@pytest.fixture(params=['numpy', 'cupy', 'array_api_strict'])
def backend(request):
    return get_backend(request.param)


def make_layer(grid, seed=12):
    strength = hp.Cn_squared_from_fried_parameter(0.003, 1e-6)
    return hp.FiniteAtmosphericLayer(grid, strength, L0=0.02, velocity=[0.01, -0.005], seed=seed)


@pytest.mark.parametrize('precision,tolerance', [('complex64', 2e-6), ('complex128', 2e-12)])
def test_seeded_phase_application(backend, precision, tolerance):
    grid = hp.make_pupil_grid(64, 0.02)
    source = hp.GaussianBeam(0.001, 0, 1e-6)(grid)
    source.total_power = 1
    layer = make_layer(grid)
    field = backend.asarray(np.asarray(source.electric_field), dtype=getattr(backend, precision))
    wavefront = hp.Wavefront(NewStyleField(field, grid), 1e-6)
    original = to_numpy(field).copy()
    previous = None
    for time in (0, 0.02):
        layer.evolve_until(time)
        phase = layer.phase_for(1e-6)
        snapshot = hp.PhaseApodizer(NewStyleField(backend.asarray(np.asarray(phase)), grid))
        screened = snapshot(wavefront)
        assert screened.electric_field.dtype == getattr(backend, precision)
        assert abs(float(screened.total_power) - 1) < tolerance
        recovered = snapshot.backward(screened)
        assert np.linalg.norm(to_numpy(recovered.electric_field.data) - original) / np.linalg.norm(original) < tolerance
        # Direct atmospheric application explicitly uploads its current host screen.
        direct = layer(wavefront)
        np.testing.assert_allclose(to_numpy(direct.electric_field.data), to_numpy(screened.electric_field.data), atol=tolerance)
        recovered = layer.backward(direct)
        assert np.linalg.norm(to_numpy(recovered.electric_field.data) - original) / np.linalg.norm(original) < tolerance

        prop = hp.FresnelPropagator(grid, 0.1, num_oversampling=1)
        actual = prop(screened)
        reference = prop(layer(source))
        error = np.linalg.norm(to_numpy(actual.electric_field.data) - reference.electric_field) / np.linalg.norm(reference.electric_field)
        assert error < tolerance
        assert abs(float(actual.total_power) - 1) < 1e-3  # Cropping of scattered light.
        if previous is not None:
            assert np.linalg.norm(to_numpy(actual.electric_field.data) - previous) / np.linalg.norm(previous) > 1e-3
        previous = to_numpy(actual.electric_field.data).copy()
    np.testing.assert_array_equal(to_numpy(field), original)


def test_scalar_phase_and_resident_updates(backend):
    grid = hp.make_pupil_grid(16)
    wavefront = hp.Wavefront(NewStyleField(backend.ones((grid.size,), dtype=backend.float32), grid))
    for phase in (0, 0.4):
        result = hp.PhaseApodizer(phase)(wavefront)
        np.testing.assert_allclose(to_numpy(result.electric_field.data), np.exp(1j * phase), atol=1e-7)
    values = backend.zeros((grid.size,), dtype=backend.float32)
    apodizer = hp.PhaseApodizer(NewStyleField(values, grid))
    apodizer(wavefront)
    values[...] = 0.4
    result = apodizer(wavefront)
    np.testing.assert_allclose(to_numpy(result.electric_field.data), np.exp(0.4j), atol=1e-7)


def test_resident_phase_snapshot(monkeypatch):
    cp = get_backend('cupy')
    grid = hp.make_pupil_grid(64, 0.02)
    source = hp.GaussianBeam(0.001, 0, 1e-6)(grid)
    layer = make_layer(grid)
    snapshot = hp.PhaseApodizer(NewStyleField(cp.asarray(np.asarray(layer.phase_for(1e-6))), grid))
    wavefront = hp.Wavefront(NewStyleField(cp.asarray(np.asarray(source.electric_field)), grid), 1e-6)
    prop = hp.FresnelPropagator(grid, 0.1, num_oversampling=1)
    expected = prop(snapshot(wavefront)).electric_field.data.copy()
    asarray = cp.asarray

    def no_host_upload(a, *args, **kwargs):
        assert not isinstance(a, np.ndarray)
        return asarray(a, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cp, 'asarray', no_host_upload)
        for _ in range(3):
            result = prop(snapshot(wavefront))
            assert isinstance(result.total_power, cp.ndarray)
    cp.testing.assert_array_equal(result.electric_field.data, expected)


def test_seeded_frozen_flow_snapshots():
    grid = hp.make_pupil_grid(32, 0.02)
    first, replay, other = make_layer(grid), make_layer(grid), make_layer(grid, seed=13)
    for time in (0, 0.01, 0.02, 0):
        for layer in (first, replay, other):
            layer.evolve_until(time)
        np.testing.assert_array_equal(first.phase_for(1e-6), replay.phase_for(1e-6))
        assert not np.allclose(first.phase_for(1e-6), other.phase_for(1e-6))
        # Achromatic optical path implies phase inversely proportional to wavelength.
        np.testing.assert_allclose(first.phase_for(2e-6), first.phase_for(1e-6) / 2)


def test_von_karman_structure_function_ensemble():
    grid = hp.make_pupil_grid(64, 0.1)
    r0, outer_scale, wavelength = 0.01, 0.02, 1e-6
    strength = hp.Cn_squared_from_fried_parameter(r0, wavelength)
    lags = (3, 5, 8)
    samples = []
    # Independent seed streams; frozen-flow frames are correlated and must not
    # be counted as independent realizations when estimating uncertainty.
    for seed in np.random.SeedSequence(871).spawn(64):
        layer = hp.FiniteAtmosphericLayer(grid, strength, outer_scale, seed=seed)
        phase = layer.phase_for(wavelength).shaped
        samples.append([np.mean((phase[:, lag:] - phase[:, :-lag])**2) for lag in lags])
    samples = np.asarray(samples)
    separations = hp.CartesianGrid(hp.UnstructuredCoords([np.asarray(lags) * grid.delta[0], np.zeros(len(lags))]))
    theory = hp.phase_structure_function_von_karman(r0, outer_scale)(separations)
    sem = samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
    # Six SEM plus an 8% finite-grid spectral truncation allowance. The chosen
    # separations exceed two pixels and L0 is resolved within the domain.
    assert np.all(abs(samples.mean(axis=0) - theory) < 6 * sem + 0.08 * theory)
