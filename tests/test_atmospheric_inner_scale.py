"""Inner-scale conventions, independent quadrature, and layer state replay."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0

import hcipy as hp


WAVELENGTH = 1e-6
STRENGTH = hp.Cn_squared_from_fried_parameter(0.01, WAVELENGTH)


def radial_grid(radii):
    return hp.CartesianGrid(hp.UnstructuredCoords([radii, np.zeros(len(radii))]))


def test_psd_cutoff_angular_frequency_convention():
    inner_scale = 0.006
    kappa = np.array([0, 0.1, 1, 2, 4]) * 5.92 / inner_scale
    grid = radial_grid(kappa)
    legacy = hp.power_spectral_density_von_karman(0.01, 0.02)(grid)
    zero = hp.power_spectral_density_von_karman(0.01, 0.02, 0)(grid)
    modified = hp.power_spectral_density_von_karman(0.01, 0.02, inner_scale)(grid)
    np.testing.assert_array_equal(zero, legacy)
    assert modified[0] == 0  # Same piston removal.
    np.testing.assert_allclose(modified[1:] / legacy[1:], np.exp(-((kappa[1:] * inner_scale / 5.92) ** 2)), rtol=2e-12)


@pytest.mark.parametrize('inner_scale', [0.002, 0.01])
def test_covariance_and_structure_against_hankel_quadrature(inner_scale):
    r0, outer_scale = 0.01, 0.02
    radii = np.array([0, 0.0001, 0.001, 0.01, 0.03])
    grid = radial_grid(radii)
    cutoff = 5.92 / inner_scale

    # Integrate independently in cycles/m, including the radial measure.
    def integrand(f, radius):
        density = 0.0229 * r0 ** (-5 / 3) * (f * f + outer_scale ** (-2)) ** (-11 / 6)
        return 2 * np.pi * f * density * np.exp(-((2 * np.pi * f / cutoff) ** 2)) * j0(2 * np.pi * f * radius)

    expected = np.array([quad(integrand, 0, 10 * cutoff / (2 * np.pi), args=(r,), epsabs=1e-13, epsrel=1e-11, limit=500)[0] for r in radii])
    covariance = hp.phase_covariance_von_karman(r0, outer_scale, inner_scale)(grid)
    structure = hp.phase_structure_function_von_karman(r0, outer_scale, inner_scale)(grid)
    np.testing.assert_allclose(covariance, expected, rtol=2e-9, atol=1e-13)
    np.testing.assert_allclose(structure, 2 * (expected[0] - expected), rtol=2e-9, atol=1e-13)
    assert structure[0] == 0


def test_covariance_small_inner_scale_legacy_normalization():
    grid = radial_grid([0, 0.001, 0.01])
    analytic = hp.phase_covariance_von_karman(0.01, 0.02)(grid)
    numerical = hp.phase_covariance_von_karman(0.01, 0.02, 1e-8)(grid)
    # Legacy analytic normalization differs slightly from the PSD's 0.0229.
    np.testing.assert_allclose(numerical, analytic, rtol=2e-4)


@pytest.mark.parametrize('invalid', [-0.001, np.nan, np.inf, [0.001]])
def test_invalid_inner_scale(invalid):
    grid = hp.make_pupil_grid(8, 0.04)
    constructors = [
        lambda: hp.power_spectral_density_von_karman(0.01, 0.02, invalid),
        lambda: hp.phase_covariance_von_karman(0.01, 0.02, invalid),
        lambda: hp.phase_structure_function_von_karman(0.01, 0.02, invalid),
        lambda: hp.FiniteAtmosphericLayer(grid, STRENGTH, 0.02, inner_scale=invalid),
        lambda: hp.InfiniteAtmosphericLayer(grid, STRENGTH, 0.02, inner_scale=invalid),
    ]
    for constructor in constructors:
        with pytest.raises(ValueError, match='inner_scale'):
            constructor()


@pytest.mark.parametrize('layer_type', [hp.FiniteAtmosphericLayer, hp.InfiniteAtmosphericLayer])
def test_zero_inner_scale_and_seeded_reset(layer_type):
    grid = hp.make_pupil_grid(8, 0.04)
    legacy = layer_type(grid, STRENGTH, 0.02, seed=51)
    zero = layer_type(grid, STRENGTH, 0.02, seed=51, inner_scale=0)
    np.testing.assert_array_equal(legacy.phase_for(WAVELENGTH), zero.phase_for(WAVELENGTH))
    layer = layer_type(grid, STRENGTH, 0.02, velocity=0.005, seed=51, inner_scale=0.004)
    initial = layer.phase_for(WAVELENGTH).copy()
    layer.evolve_until(1)
    moved = layer.phase_for(WAVELENGTH).copy()
    assert not np.allclose(initial, moved)
    layer.reset()
    np.testing.assert_array_equal(layer.phase_for(WAVELENGTH), initial)
    layer.evolve_until(1)
    np.testing.assert_array_equal(layer.phase_for(WAVELENGTH), moved)
    layer.reset(True)
    assert not np.allclose(layer.phase_for(WAVELENGTH), initial)


def test_finite_scale_update_replays_selected_stream_at_current_position():
    grid = hp.make_pupil_grid(8, 0.04)
    layer = hp.FiniteAtmosphericLayer(grid, STRENGTH, 0.02, velocity=0.005, seed=51, inner_scale=0.004)
    reference = hp.FiniteAtmosphericLayer(grid, STRENGTH, 0.02, velocity=0.005, seed=51, inner_scale=0.008)
    for obj in [layer, reference]:
        obj.reset(True)  # Use a later selected realization, not just the original seed.
        obj.evolve_until(0.3)
    before = layer.phase_for(WAVELENGTH).copy()
    layer.inner_scale = 0.008
    assert layer.t == 0.3
    np.testing.assert_array_equal(layer.center, reference.center)
    np.testing.assert_array_equal(layer.phase_for(WAVELENGTH), reference.phase_for(WAVELENGTH))
    assert not np.allclose(layer.phase_for(WAVELENGTH), before)
    for obj in [layer, reference]:
        obj.reset(True)
    np.testing.assert_array_equal(layer.phase_for(WAVELENGTH), reference.phase_for(WAVELENGTH))


@pytest.mark.parametrize('parameter,value', [('inner_scale', 0.008), ('outer_scale', 0.04)])
def test_infinite_scale_update_rebuilds_and_replays(parameter, value):
    grid = hp.make_pupil_grid(8, 0.04)
    layer = hp.InfiniteAtmosphericLayer(grid, STRENGTH, 0.02, seed=51, velocity=0.005, inner_scale=0.004)
    reference = hp.InfiniteAtmosphericLayer(
        grid,
        STRENGTH,
        value if parameter == 'outer_scale' else 0.02,
        seed=51,
        velocity=0.005,
        inner_scale=value if parameter == 'inner_scale' else 0.004,
    )
    old_covariance = layer.cov_matrix_vertical.copy()
    layer.evolve_until(1)
    setattr(layer, parameter, value)
    assert layer.t == 0
    assert not np.allclose(layer.cov_matrix_vertical, old_covariance, rtol=1e-6, atol=0)
    np.testing.assert_array_equal(layer.phase_for(WAVELENGTH), reference.phase_for(WAVELENGTH))
    for obj in [layer, reference]:
        obj.evolve_until(1)
    np.testing.assert_array_equal(layer.phase_for(WAVELENGTH), reference.phase_for(WAVELENGTH))


def test_positive_inner_scale_covariance_requires_finite_outer_scale():
    with pytest.raises(ValueError, match='finite positive L0'):
        hp.phase_covariance_von_karman(0.01, np.inf, 0.004)
    with pytest.raises(ValueError, match='finite positive L0'):
        hp.InfiniteAtmosphericLayer(hp.make_pupil_grid(8, 0.04), STRENGTH, inner_scale=0.004)
    # A finite sampled spectrum with removed DC can still use infinite L0.
    layer = hp.FiniteAtmosphericLayer(hp.make_pupil_grid(8, 0.04), STRENGTH, inner_scale=0.004)
    assert np.all(np.isfinite(layer.phase_for(WAVELENGTH)))


@pytest.mark.parametrize('site', ['mauna_kea', 'las_campanas', 'keck'])
def test_standard_factory_and_multilayer_inner_scale(site):
    atmosphere = hp.make_standard_atmosphere(hp.make_pupil_grid(8, 0.04), outer_scale=0.02, site=site, inner_scale=0.004)
    assert all(layer.inner_scale == 0.004 for layer in atmosphere.layers)
    atmosphere.inner_scale = 0.008
    assert atmosphere.inner_scale == 0.008
    assert all(layer.inner_scale == 0.008 for layer in atmosphere.layers)


def test_modal_wrapper_inner_scale_refreshes_correction():
    grid = hp.make_pupil_grid(8, 0.04)
    layer = hp.InfiniteAtmosphericLayer(grid, STRENGTH, 0.02, velocity=0.005, seed=51, inner_scale=0.004)
    layer.evolve_until(1)
    modes = hp.make_zernike_basis(3, 0.04, grid)
    wrapper = hp.ModalAdaptiveOpticsLayer(layer, modes, lag=0)
    assert layer.t == 1  # Wrapping must not reset an evolved inner-scale layer.
    assert wrapper.inner_scale == 0.004
    wrapper.inner_scale = 0.008
    assert wrapper.t == layer.t == 0
    expected = hp.ModalAdaptiveOpticsLayer(layer, modes, lag=0)
    np.testing.assert_array_equal(wrapper.phase_for(WAVELENGTH), expected.phase_for(WAVELENGTH))
