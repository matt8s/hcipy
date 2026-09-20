# Thin lenses with independent x/y optical powers

`ThinLens` accepts a scalar focal length or a pair `(fx, fy)` in meters.
A pair models an axis-aligned astigmatic lens: the two transverse directions
have independent optical powers. An elliptical clear aperture is a separate
amplitude mask and does not, by itself, change those powers.

```python
import numpy as np
import hcipy as hp

wavelength = 1e-6
grid = hp.make_pupil_grid(128, 0.004)
waist = 0.00035
amplitude = np.exp(-(grid.x**2 + grid.y**2) / waist**2)
wavefront = hp.Wavefront(hp.Field(amplitude.astype(complex), grid), wavelength)

# Reference focal lengths: 0.15 m along x, 0.30 m along y.
lens = hp.ThinLens((0.15, 0.30), 1.5, wavelength)
after_lens = lens(wavefront)
propagator = hp.FresnelPropagator(
    grid, 0.15, num_oversampling=1, zero_padding=2)
output = propagator(after_lens)

# Optional finite rectangular clear aperture, specified independently.
stop = hp.Apodizer(hp.make_rectangular_aperture((0.001, 0.0005))(grid))
clipped_output = propagator(lens(stop(wavefront)))
```

At the reference wavelength the transmitted complex field is multiplied by

\[
 \exp\!\left[-\frac{ik}{2}
 \left(\frac{x^2}{f_x}+\frac{y^2}{f_y}\right)\right],
 \qquad k=2\pi/\lambda.
\]

Positive focal lengths are converging; negative values are diverging. Setting
one focal length to `np.inf` gives a cylindrical lens; setting both to infinity
gives zero power. Equal finite focal lengths reproduce the scalar lens. Pairs
must contain two nonzero values and no NaNs; invalid pairs are rejected when
the surface/radii are evaluated. Assigning a new `lens.focal_length` updates
subsequent evaluations.

For dispersive material `n(wavelength)`, the phase above is multiplied by
`(n(wavelength) - 1) / (n(reference_wavelength) - 1)`. Both principal axes share
that material; the reference index must differ from one to define the surface
curvature from a finite focal length. The implementation retains the existing
scalar-lens convention.

This is a scalar paraxial thin phase element, with no absorption, finite
thickness or higher-order aberrations. Its axes are the grid x/y axes. Sampling
must resolve the quadratic phase and the propagated field. For a clipped beam,
check aperture sampling, padding and output-window convergence; a smaller
aperture generally requires a larger diffraction field of view. The finite
rectangle example is illustrative, not a convergence certificate for arbitrary
aperture sizes. Subwavelength openings require a different physical model.

The tests compare absolute complex phase, dispersion, power, cylindrical and
equal-power limits, and Fresnel propagation against independent separable
Gaussian integrals. The tests use the existing NumPy field path; this extension
does not establish GPU-native lens evaluation. `lens.backward()` remains the
mathematical inverse phase operation, not a diffuse-reflection return model.
