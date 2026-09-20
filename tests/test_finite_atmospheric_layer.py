"""State-transition regressions for seeded finite frozen-flow layers."""
import numpy as np
import pytest

import hcipy as hp


WAVELENGTH = 1e-6
STRENGTH = hp.Cn_squared_from_fried_parameter(0.003, WAVELENGTH)


def make_layer(strength=STRENGTH, outer_scale=0.02):
    return hp.FiniteAtmosphericLayer(hp.make_pupil_grid(16, 0.02), strength,
                                     outer_scale, velocity=[0.01, -0.005], seed=12)


def test_finite_layer_absolute_time_and_rewind():
    layer = make_layer()
    initial = layer.phase_for(WAVELENGTH).copy()
    for time in (0.1, 0.02, -0.03, 0):
        layer.evolve_until(time)
        assert layer.t == time
        np.testing.assert_array_equal(layer.center, layer.velocity * time)
    np.testing.assert_array_equal(layer.phase_for(WAVELENGTH), initial)
    layer.t = 0.05
    assert layer.t == 0.05
    np.testing.assert_array_equal(layer.center, layer.velocity * 0.05)


@pytest.mark.parametrize('independent', [False, True])
def test_finite_layer_reset_restores_origin_and_replay(independent):
    layer = make_layer()
    initial = layer.phase_for(WAVELENGTH).copy()
    layer.evolve_until(0.1)
    assert not np.allclose(layer.phase_for(WAVELENGTH), initial)
    layer.reset(make_independent_realization=independent)
    assert layer.t == 0
    np.testing.assert_array_equal(layer.center, [0, 0])
    selected = layer.phase_for(WAVELENGTH).copy()
    if independent:
        assert not np.allclose(selected, initial)
    else:
        np.testing.assert_array_equal(selected, initial)
    layer.evolve_until(0.03)
    displaced = layer.phase_for(WAVELENGTH).copy()
    layer.reset()
    np.testing.assert_array_equal(layer.phase_for(WAVELENGTH), selected)
    layer.evolve_until(0.03)
    np.testing.assert_array_equal(layer.phase_for(WAVELENGTH), displaced)


@pytest.mark.parametrize('cached_phase', [False, True])
@pytest.mark.parametrize('parameter,value', [('Cn_squared', 4 * STRENGTH), ('outer_scale', 0.04), ('L0', 0.04)])
def test_finite_layer_parameter_update_preserves_position(parameter, value, cached_phase):
    layer = make_layer()
    layer.evolve_until(0.02)
    if cached_phase:
        layer.phase_for(WAVELENGTH)
    setattr(layer, parameter, value)
    actual = layer.phase_for(WAVELENGTH)
    reference = make_layer(strength=value) if parameter == 'Cn_squared' else make_layer(outer_scale=value)
    reference.evolve_until(0.02)
    np.testing.assert_allclose(actual, reference.phase_for(WAVELENGTH), rtol=1e-13, atol=1e-13)
    assert layer.t == 0.02
    np.testing.assert_array_equal(layer.center, layer.velocity * 0.02)
    if parameter == 'Cn_squared':
        original = make_layer()
        original.evolve_until(0.02)
        # PSD is linear in integrated Cn²: 4x strength gives 2x phase for the
        # same realization. This catches accidental selection of a new draw.
        np.testing.assert_allclose(actual, 2 * original.phase_for(WAVELENGTH), rtol=1e-13, atol=1e-13)


def test_finite_layer_parameter_rebuild_preserves_selected_random_stream():
    layer, reference = make_layer(), make_layer()
    # Select a later realization, rather than testing only the initial seed.
    for _ in range(2):
        layer.reset(make_independent_realization=True)
        reference.reset(make_independent_realization=True)
    for obj in (layer, reference):
        obj.evolve_until(0.05)
    phase = layer.phase_for(WAVELENGTH).copy()
    layer.Cn_squared = 4 * STRENGTH
    np.testing.assert_allclose(layer.phase_for(WAVELENGTH), 2 * phase, rtol=1e-13, atol=1e-13)
    layer.outer_scale = 0.04
    layer.phase_for(WAVELENGTH)
    # Multiple pending changes and repeated reads must not consume a new
    # realization, or change what the next independent reset selects.
    layer.Cn_squared = STRENGTH
    layer.L0 = 0.02
    for _ in range(2):
        np.testing.assert_array_equal(layer.phase_for(WAVELENGTH), reference.phase_for(WAVELENGTH))
    for _ in range(2):
        layer.reset(make_independent_realization=True)
        reference.reset(make_independent_realization=True)
        np.testing.assert_array_equal(layer.phase_for(WAVELENGTH), reference.phase_for(WAVELENGTH))
