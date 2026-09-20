from .surface_profiles import parabolic_surface_sag
from .optical_element import OpticalElement
from .wavefront import Wavefront
from ..field import Field
import numpy as np


class ThinLens(OpticalElement):
    '''A parabolic thin lens.

    Parameters
    ----------
    focal_length : scalar or array_like with two elements
        The focal length of the thin lens at the reference wavelength, in meters.
        A pair specifies independent focal lengths along the grid x and y axes,
        respectively. Positive values focus and negative values defocus. Use
        infinity for zero optical power on an axis (a cylindrical lens).
    refractive_index : scalar or function of wavelength
        The refractive index of the lens material.
    reference_wavelength : scalar
        The wavelength for which the focal length is defined.

    Notes
    -----
    This is an ideal scalar paraxial phase element with unlimited clear aperture.
    An independent-axis lens has phase -k/2 * (x**2/fx + y**2/fy) at the reference
    wavelength. Both axes use the same material dispersion. Combine with an
    apodizer to specify a finite clear aperture; an elliptical clear aperture
    alone does not produce different optical powers.
    '''

    def __init__(self, focal_length, refractive_index, reference_wavelength):
        self.reference_wavelength = reference_wavelength
        self.refractive_index = refractive_index
        self.focal_length = focal_length

    def _get_refractive_index(self, wavelength):
        if callable(self.refractive_index):
            return self.refractive_index(wavelength)
        else:
            return self.refractive_index

    @property
    def surface_sag(self):
        radii = self.radius_of_curvature
        if np.ndim(radii) == 0:
            return parabolic_surface_sag(-radii)

        def sag(grid):
            cartesian = grid.as_('cartesian')
            return Field(-0.5 * (cartesian.x**2 / radii[0] + cartesian.y**2 / radii[1]), grid)

        return sag

    @property
    def radius_of_curvature(self):
        n0 = self._get_refractive_index(self.reference_wavelength)
        if np.ndim(self.focal_length) != 0:
            focal_lengths = np.asarray(self.focal_length, dtype=float)
            if focal_lengths.shape != (2,) or np.any(np.isnan(focal_lengths)) or np.any(focal_lengths == 0):
                raise ValueError('Axis focal lengths must contain two nonzero values; infinity is allowed.')
            return focal_lengths * (n0 - 1)
        return self.focal_length * (n0 - 1)

    def forward(self, wavefront):
        surface_sag = self.surface_sag(wavefront.electric_field.grid)
        n = self._get_refractive_index(wavefront.wavelength)

        new_field = wavefront.electric_field * np.exp(1j * (n - 1) * surface_sag * wavefront.wavenumber)
        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)

    def backward(self, wavefront):
        surface_sag = self.surface_sag(wavefront.electric_field.grid)
        n = self._get_refractive_index(wavefront.wavelength)

        new_field = wavefront.electric_field * np.exp(-1j * (n - 1) * surface_sag * wavefront.wavenumber)
        return Wavefront(new_field, wavefront.wavelength, wavefront.input_stokes_vector)
