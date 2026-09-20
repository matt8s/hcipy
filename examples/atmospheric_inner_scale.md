# Atmospheric inner and outer scales

Finite and infinite atmospheric layers accept the keyword `inner_scale`, in
meters. It defaults to zero, retaining the previous spectrum and seeded draws.
`L0` / `outer_scale` continues to specify the outer scale. The same keyword is
available on the three standard site-layer factories and `make_standard_atmosphere`;
it is a user-selected parameter, not an inferred measurement of those sites.

```python
import hcipy as hp

grid = hp.make_pupil_grid(128, 0.08)  # meters
wavelength = 1e-6                   # meters
strength = hp.Cn_squared_from_fried_parameter(0.01, wavelength)
layer = hp.FiniteAtmosphericLayer(
    grid, strength, L0=0.02, velocity=[0.01, -0.005],
    oversampling=2, seed=12, inner_scale=0.004)
layer.evolve_until(0.1)             # seconds
phase = layer.phase_for(wavelength) # radians
layer.inner_scale = 0.008
updated_phase = layer.phase_for(wavelength)
```

## Physical convention

The isotropic modified von Kármán phase spectrum is

\[
 W_\phi(f)=0.0229\,r_0^{-5/3}
 (f^2+L_0^{-2})^{-11/6}
 \exp[-(2\pi f l_0/5.92)^2].
\]

Here `inner_scale` is \(l_0\), \(f\) is in cycles/m and the Fried parameter
\(r_0\) specifies inertial-range strength at the wavelength of the phase.
Increasing the inner scale suppresses high spatial frequencies; the spectrum
is not renormalized to recover the old total variance. This is the exponential
dissipation model, without a dissipation-range spectral bump.

`power_spectral_density_von_karman(r0, L0, inner_scale=...)` takes an angular
frequency grid \(\kappa=2\pi f\) and returns a density to integrate with
\(d^2\kappa/(2\pi)^2\). The sampled DC coefficient remains zero. The cutoff
convention is also used by the independent FOSS
[AOtools phase-screen implementation](https://github.com/AOtools/aotools/blob/master/aotools/turbulence/phasescreen.py),
which uses the rounded coefficient 0.023 rather than HCIPy's 0.0229; exact
amplitude equality between the packages is therefore not implied.

The covariance and structure-function generators accept the same keyword.
For positive inner scales they integrate this spectrum numerically, using a
nonoscillatory Gaussian-mixture integral. The structure function evaluates
\(2[B(0)-B(r)]\) with `expm1` to avoid cancellation near the origin and is zero
at exactly zero separation. These continuous statistics include frequencies
outside a finite screen's sampled bandwidth. Covariance includes piston;
spatial increments do not depend on piston.

The numerical statistics currently require finite positive `L0`. The finite
layer and PSD also permit infinite `L0`, with the DC sample removed. The existing
zero-inner-scale analytic statistics are retained exactly. Their normalization
differs from the PSD's 0.0229 coefficient by approximately 0.0193%, so the
positive-inner-scale covariance approaches that slightly different normalization
as the inner scale tends to zero.

For reproducibility, the numerical covariance uses \(a=2\pi/L_0\),
\(q=(a l_0/5.92)^2\), \(\nu=11/6\), and
\(A=0.0229(2\pi)^{11/3}r_0^{-5/3}\):

\[
 B(r)=\frac{A a^{2-2\nu}}{4\pi\Gamma(\nu)}
 \int_0^\infty \frac{s^{\nu-1}e^{-s}}{s+q}
 \exp\!\left[-\frac{(ar)^2}{4(s+q)}\right]ds.
\]

This follows by writing the power law as a Laplace integral and integrating
the radial Fourier kernel analytically. Adaptive vector quadrature uses
\(z=\log s\) on `[-80, 5]`, relative tolerance `1e-11`, and dimensionless
absolute tolerance `1e-25`. Repeated radii are evaluated once. For the structure
function, the last exponential is replaced by twice one minus that exponential.
These bounds resolve ordinary optical simulation scales; extreme length-scale
ratios still require convergence checks. A quadrature convergence failure is
reported rather than silently returning a covariance.

## Evolution and parameter updates

- **Finite layers:** both spectral bands receive the cutoff. Changing the inner
  scale rebuilds the spectrum lazily with the selected seed at the current time
  and displacement. It does not consume the next independent realization.
- **Infinite layers:** both the initial finite screen and the autoregressive
  extension covariance use the inner scale. Changing it rebuilds the extension
  matrices and resets the selected realization to time zero. Changing the outer
  scale with a positive inner scale also resets it. A failed factorization leaves
  the previous state intact. Reset is necessary because already-generated screen
  samples cannot be retained consistently under a new covariance.
- **Multilayer atmospheres:** set each layer's `inner_scale` for a height-dependent
  profile. Setting `atmosphere.inner_scale` updates every layer; its getter reports
  the first layer's value, following the existing outer-scale convention.
- **Modal AO wrappers:** the property forwards to the underlying layer and clears
  and reconstructs the modal correction history after the update.

## Resolution and limits

The cutoff angular frequency is \(\kappa_m=5.92/l_0\). Check spatial convergence
with a Nyquist frequency \(\pi/\Delta x\) sufficiently above it when gradients or
small-separation statistics matter. If the cutoff lies above the grid bandwidth,
the simulation may show little inner-scale dependence. Increasing oversampling
extends the low-frequency representation; it does not raise the Nyquist limit.
The finite two-band model and its wrapping behavior remain in use.

Very smooth infinite screens (large inner scale relative to pixel spacing), or
large outer scales relative to the stencil, can produce ill-conditioned
covariance matrices. Factorization can fail; no artificial white-noise floor is
added to hide this limitation. Numerical covariance setup is CPU work. This
addition does not make finite or infinite atmospheric evolution GPU-native.

Regression checks include independent radial Fourier quadrature, deterministic
state replay, and 128-realization spatial-increment ensembles for the finite
two-band spectrum and evolved infinite screens. Finite-bandwidth statistics are
compared to their discrete quadrature, rather than assuming they reproduce the
unbounded continuous spectrum exactly.
