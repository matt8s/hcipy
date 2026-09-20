"""Ensemble validation of dissipative phase-screen spatial increments."""

import numpy as np
import pytest

import hcipy as hp


@pytest.mark.parametrize('parameter,value', [('inner_scale', 0.008), ('outer_scale', 0.04)])
def test_infinite_covariance_update_failure_preserves_state(monkeypatch, parameter, value):
    grid = hp.make_pupil_grid(8, 0.04)
    layer = hp.InfiniteAtmosphericLayer(grid, hp.Cn_squared_from_fried_parameter(0.01, 1e-6), 0.02, velocity=0.005, seed=51, inner_scale=0.004)
    layer.evolve_until(1)
    phase = layer.phase_for(1e-6).copy()
    covariance = layer.cov_matrix_vertical
    rng_state = layer.rng.bit_generator.state
    original_value = getattr(layer, parameter)

    def fail_factorization(self):
        raise np.linalg.LinAlgError('Test factorization failure')

    monkeypatch.setattr(hp.InfiniteAtmosphericLayer, '_make_ab_matrices', fail_factorization)
    with pytest.raises(np.linalg.LinAlgError, match='Test factorization'):
        setattr(layer, parameter, value)
    assert getattr(layer, parameter) == original_value
    assert layer.t == 1
    assert layer.cov_matrix_vertical is covariance
    assert layer.rng.bit_generator.state == rng_state
    np.testing.assert_array_equal(layer.phase_for(1e-6), phase)


def test_inner_scale_small_separation_quadratic_limit():
    grid = hp.CartesianGrid(hp.UnstructuredCoords([[0, 1e-8, 2e-8], [0, 0, 0]]))
    structure = hp.phase_structure_function_von_karman(0.01, 0.02, 0.006)(grid)
    assert structure[0] == 0
    assert structure[1] > 0
    # A dissipative screen is mean-square differentiable: D(r) ~ r^2.
    np.testing.assert_allclose(structure[2] / structure[1], 4, rtol=1e-9)


def test_multiscale_structure_spatial_convergence():
    separation = 0.005
    grid = hp.CartesianGrid(hp.UnstructuredCoords([[separation], [0]]))
    reference = hp.phase_structure_function_von_karman(0.01, 0.02, 0.006)(grid)[0]
    errors = []
    psd = hp.power_spectral_density_von_karman(0.01, 0.02, 0.006)
    for n in [16, 32, 64, 128]:
        factory = hp.SpectralNoiseFactoryMultiscale(psd, hp.make_pupil_grid(n, 0.08), 2)
        boundary = np.max(np.abs(factory.input_grid_2.x))
        sampled_structure = 0
        for band, k in enumerate([factory.input_grid_1, factory.input_grid_2]):
            radius = np.hypot(k.x, k.y)
            mask = radius >= boundary if band == 0 else radius < boundary
            sampled_structure += np.sum(2 * psd(k) * mask * (1 - np.cos(k.x * separation))) * k.weights / (2 * np.pi) ** 2
        errors.append(abs(sampled_structure / reference - 1))
    # Compare expectations at fixed physical separation, not same-seed draws
    # on different grids. The coarse grid misses the dissipation bandwidth.
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < 1e-6


@pytest.mark.parametrize('inner_scale', [0.004, 0.012])
def test_multiscale_inner_scale_structure_ensemble(inner_scale):
    grid = hp.make_uniform_grid([32, 24], [0.08, 0.06])
    wavelength = 1e-6
    r0, outer_scale = 0.01, 0.02
    layer = hp.FiniteAtmosphericLayer(
        grid, hp.Cn_squared_from_fried_parameter(r0, wavelength), outer_scale, oversampling=2, seed=191, inner_scale=inner_scale
    )
    shifts = [1, 2, 4]
    expected = np.zeros(len(shifts))
    # Independent discrete quadrature of the implemented two-band model.
    # Zero DC does not affect increments; no fitted amplitude or same-seed
    # comparisons across different grids are used.
    factory = layer.noise_factory
    boundary = np.max(np.abs(factory.input_grid_2.x))
    for band, k in enumerate([factory.input_grid_1, factory.input_grid_2]):
        radius = np.hypot(k.x, k.y)
        mask = radius >= boundary if band == 0 else radius < boundary
        spectrum = (
            0.0229 * r0 ** (-5 / 3) * np.exp(-((radius * inner_scale / 5.92) ** 2)) / ((radius / (2 * np.pi)) ** 2 + outer_scale ** (-2)) ** (11 / 6)
        )
        for i, shift in enumerate(shifts):
            expected[i] += np.sum(2 * spectrum * mask * (1 - np.cos(k.x * grid.delta[0] * shift))) * k.weights / (2 * np.pi) ** 2

    measurements = []
    for _ in range(128):
        layer.reset(make_independent_realization=True)
        phase = layer.phase_for(wavelength).shaped
        measurements.append([np.mean((phase[:, shift:] - phase[:, :-shift]) ** 2) for shift in shifts])
    measurements = np.asarray(measurements)
    sem = measurements.std(axis=0, ddof=1) / np.sqrt(len(measurements))
    assert np.all(np.abs(measurements.mean(axis=0) - expected) < 6 * sem + 0.005 * expected)


def test_infinite_inner_scale_evolved_structure_ensemble():
    grid = hp.make_pupil_grid(12, 0.06)
    wavelength = 1e-6
    r0, outer_scale, inner_scale = 0.01, 0.02, 0.006
    layer = hp.InfiniteAtmosphericLayer(
        grid,
        hp.Cn_squared_from_fried_parameter(r0, wavelength),
        outer_scale,
        velocity=[grid.delta[0], 0],
        interpolation_order=0,
        seed=72,
        inner_scale=inner_scale,
    )
    separation_grid = hp.CartesianGrid(hp.UnstructuredCoords([[grid.delta[0], 2 * grid.delta[0]], [0, 0]]))
    expected = hp.phase_structure_function_von_karman(r0, outer_scale, inner_scale)(separation_grid)
    measurements = []
    for _ in range(128):
        layer.reset(make_independent_realization=True)
        # Replace the entire initial window twice before measuring. Compare
        # independent runs, not correlated frames as independent samples.
        layer.evolve_until(24)
        phase = layer.phase_for(wavelength).shaped
        measurements.append([np.mean((phase[:, shift:] - phase[:, :-shift]) ** 2) for shift in [1, 2]])
    measurements = np.asarray(measurements)
    sem = measurements.std(axis=0, ddof=1) / np.sqrt(len(measurements))
    assert np.all(np.abs(measurements.mean(axis=0) - expected) < 6 * sem + 0.05 * expected)
