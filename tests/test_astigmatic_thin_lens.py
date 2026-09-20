"""Independent-axis phase and analytical paraxial propagation of a thin lens."""

import numpy as np
import pytest

import hcipy as hp


@pytest.mark.parametrize('focal_lengths', [(0.2, 0.4), (-0.2, 0.4), (np.inf, 0.3), (np.inf, np.inf)])
@pytest.mark.parametrize('wavelength', [0.8e-6, 1e-6])
def test_astigmatic_lens_phase_dispersion_and_inverse(focal_lengths, wavelength):
    grid = hp.make_uniform_grid([24, 16], [0.003, 0.002])
    refractive_index = lambda w: 1.5 + 0.1 * (w / 1e-6 - 1)
    lens = hp.ThinLens(focal_lengths, refractive_index, 1e-6)
    wavefront = hp.Wavefront(hp.Field(np.exp(-(grid.x**2 + grid.y**2) / 0.001**2).astype(complex), grid), wavelength)
    before = wavefront.electric_field.copy()
    phase = -np.pi / wavelength * (grid.x**2 / focal_lengths[0] + grid.y**2 / focal_lengths[1])
    phase *= (refractive_index(wavelength) - 1) / (refractive_index(1e-6) - 1)
    result = lens.forward(wavefront)
    np.testing.assert_allclose(result.electric_field, before * np.exp(1j * phase), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(result.total_power, wavefront.total_power, rtol=1e-14)
    np.testing.assert_allclose(lens.backward(result).electric_field, before, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(wavefront.electric_field, before)
    assert result.electric_field.grid == grid


def test_equal_axis_lens_matches_scalar_and_updates():
    grid = hp.make_pupil_grid(16, 0.003)
    wavefront = hp.Wavefront(grid.ones(dtype=complex), 1e-6)
    circular = hp.ThinLens(0.2, 1.5, 1e-6)
    astigmatic = hp.ThinLens([0.2, 0.2], 1.5, 1e-6)
    np.testing.assert_allclose(astigmatic(wavefront).electric_field, circular(wavefront).electric_field, atol=1e-13)
    astigmatic.focal_length = [0.3, np.inf]
    expected = np.exp(-1j * np.pi * grid.x**2 / (wavefront.wavelength * 0.3))
    np.testing.assert_allclose(astigmatic(wavefront).electric_field, expected, atol=1e-13)


@pytest.mark.parametrize('distance', [0.15, 0.3])
def test_astigmatic_gaussian_fresnel_propagation(distance):
    grid = hp.make_pupil_grid(128, 0.004)
    waist, wavelength = 0.00035, 1e-6
    focal_lengths = (0.15, 0.3)
    wavefront = hp.Wavefront(hp.Field(np.exp(-(grid.x**2 + grid.y**2) / waist**2).astype(complex), grid), wavelength)
    lens = hp.ThinLens(focal_lengths, 1.5, wavelength)
    propagator = hp.FresnelPropagator(grid, distance, num_oversampling=1, zero_padding=2)
    actual = propagator(lens(wavefront))

    # Independent Gaussian Fresnel integral, retaining absolute complex phase.
    # Each axis has its own quadratic coefficient and complex propagation factor.
    k = 2 * np.pi / wavelength
    ax, ay = [1 / waist**2 + 1j * k / (2 * f) for f in focal_lengths]
    qx, qy = 1 + 2j * distance * ax / k, 1 + 2j * distance * ay / k
    expected = np.exp(1j * k * distance) / (np.sqrt(qx) * np.sqrt(qy))
    expected *= np.exp(-ax * grid.x**2 / qx - ay * grid.y**2 / qy)
    error = np.linalg.norm(actual.electric_field - expected) / np.linalg.norm(expected)
    assert error < 2e-10
    np.testing.assert_allclose(actual.total_power, wavefront.total_power, rtol=2e-10)


@pytest.mark.parametrize('focal_lengths', [[0.2], [0.2, 0.3, 0.4], [0, 0.3], [0.2, np.nan]])
def test_invalid_axis_focal_lengths(focal_lengths):
    lens = hp.ThinLens(focal_lengths, 1.5, 1e-6)
    with pytest.raises(ValueError, match='two nonzero'):
        lens.surface_sag(hp.make_pupil_grid(8))
