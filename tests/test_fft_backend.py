"""Direct quadrature references for FFTs using backend Fields and CPU grids."""
import copy
import numpy as np
import pytest

import hcipy as hp
from hcipy.field import NewStyleField
from hcipy._math.backends import to_numpy
from conftest import get_backend


@pytest.fixture(params=['numpy', 'cupy', 'array_api_strict'])
def backend(request):
    return get_backend(request.param)


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'complex64', 'complex128'])
@pytest.mark.parametrize('tensor_shape', [(), (2, 3)])
@pytest.mark.parametrize('emulate', [False, True])
@pytest.mark.parametrize('padding,fov,shift', [(1, 1, 0), (2, 1, [3, -7]), (2, 0.5, [8, 0])])
def test_fft_direct_quadrature(backend, dtype, tensor_shape, emulate, padding, fov, shift):
    grid = hp.make_uniform_grid([7, 8], [0.02, 0.03]).shifted([0.001, -0.002])
    fft = hp.FastFourierTransform(grid, q=padding, fov=fov, shift=shift, emulate_fftshifts=emulate)
    rng = np.random.default_rng(12)
    original = rng.normal(size=tensor_shape + (grid.size,))
    if dtype.startswith('complex'):
        original = original + 1j * rng.normal(size=original.shape)
    original = original.astype(dtype)
    data = backend.asarray(original)
    field = NewStyleField(data, grid)
    transformed = fft.forward(field)
    actual = to_numpy(transformed.data).copy()
    output_grid = fft.output_grid
    kernel = np.exp(-1j * (np.asarray(output_grid.x)[:, None] * np.asarray(grid.x)[None, :]
                           + np.asarray(output_grid.y)[:, None] * np.asarray(grid.y)[None, :]))
    expected = original @ kernel.T * grid.weights
    tolerance = 2e-6 if dtype in ('float32', 'complex64') else 2e-12
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < tolerance
    assert transformed.shape == tensor_shape + (output_grid.size,)
    assert transformed.dtype == getattr(backend, 'complex64' if dtype in ('float32', 'complex64') else 'complex128')
    assert transformed.grid is output_grid
    recovered = fft.backward(transformed)
    backward_reference = actual @ kernel.conj() * output_grid.weights / (2 * np.pi)**2
    assert np.linalg.norm(to_numpy(recovered.data) - backward_reference) / np.linalg.norm(backward_reference) < tolerance
    assert recovered.grid is grid
    if fov == 1:
        assert np.linalg.norm(to_numpy(recovered.data) - original) / np.linalg.norm(original) < tolerance
        power_in = np.sum(abs(original)**2) * grid.weights
        power_out = np.sum(abs(actual)**2) * output_grid.weights / (2 * np.pi)**2
        assert abs(power_out / power_in - 1) < tolerance
    # Further calls cannot overwrite either input or earlier returned fields.
    fft.forward(field)
    np.testing.assert_array_equal(to_numpy(data), original)
    np.testing.assert_array_equal(to_numpy(transformed.data), actual)


@pytest.mark.parametrize('dims', [(9,), (5, 6, 7)])
@pytest.mark.parametrize('emulate', [False, True])
def test_fft_other_dimensions(backend, dims, emulate):
    grid = hp.make_uniform_grid(dims, [0.02] * len(dims))
    fft = hp.FastFourierTransform(grid, q=2, emulate_fftshifts=emulate)
    original = np.random.default_rng(5).normal(size=grid.size)
    field = NewStyleField(backend.asarray(original), grid)
    result = fft.backward(fft.forward(field))
    np.testing.assert_allclose(to_numpy(result.data), original, atol=2e-12)


def test_fft_resident_cache_and_switching(monkeypatch):
    cp = get_backend('cupy')
    grid = hp.make_pupil_grid(32, 0.02)
    fft = hp.FastFourierTransform(grid, q=2, shift=[1, -2])
    original = np.random.default_rng(4).normal(size=grid.size)
    expected = fft.forward(hp.Field(original, grid)).copy()
    field = NewStyleField(cp.asarray(original), grid)
    fft.forward(field)
    scratch = fft._array_api_array
    shift_input, shift_output = fft._array_api_shift_input, fft._array_api_shift_output
    asarray = cp.asarray

    def no_host_upload(a, *args, **kwargs):
        assert not isinstance(a, np.ndarray)
        return asarray(a, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cp, 'asarray', no_host_upload)
        for _ in range(3):
            result = fft.forward(field)
            recovered = fft.backward(result)
            assert isinstance(result.data, cp.ndarray)
            assert fft._array_api_array is scratch
            assert fft._array_api_shift_input is shift_input
            assert fft._array_api_shift_output is shift_output
    np.testing.assert_allclose(cp.asnumpy(result.data), expected, atol=1e-15)
    np.testing.assert_allclose(cp.asnumpy(recovered.data), original, atol=2e-12)
    # Same instance can alternate explicit backends/precisions and legacy Fields.
    for xp, dtype in [(np, np.complex64), (cp, cp.complex64), (cp, cp.complex128)]:
        result = fft.forward(NewStyleField(xp.asarray(original, dtype=dtype), grid))
        np.testing.assert_allclose(to_numpy(result.data), expected, rtol=2e-5, atol=1e-10)
        assert result.dtype == dtype
    np.testing.assert_array_equal(fft.forward(hp.Field(original, grid)), expected)


def test_warmed_fft_deepcopy(backend):
    grid = hp.make_pupil_grid(16, 0.02)
    fft = hp.FastFourierTransform(grid, q=2)
    field = NewStyleField(backend.asarray(np.random.default_rng(8).normal(size=grid.size)), grid)
    expected = fft.forward(field)
    cloned = copy.deepcopy(fft)
    assert cloned._array_api_array is not fft._array_api_array
    np.testing.assert_allclose(to_numpy(cloned.forward(field).data), to_numpy(expected.data), atol=1e-15)
