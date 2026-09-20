"""State isolation and statistical checks independent of backend sample values."""
import numpy as np
import pytest

from conftest import get_backend
from hcipy._math.backends import to_numpy
from hcipy._math.random import make_random_generator


@pytest.fixture(params=['numpy', 'cupy', 'array_api_strict'])
def backend(request):
    return get_backend(request.param)


def test_copy_after_mixed_draws(backend):
    rng = make_random_generator(backend, seed=123)
    rng.normal(size=17)
    rng.poisson(2, size=(3, 5))
    clone = rng.copy()
    for method, kwargs in [('normal', {}), ('gamma', {'shape_param': 2}),
                           ('choice', {'a': 100}), ('uniform', {}), ('exponential', {})]:
        for size in ((), (7,), (3, 4)):
            a = getattr(rng, method)(size=size, **kwargs)
            b = getattr(clone, method)(size=size, **kwargs)
            np.testing.assert_array_equal(to_numpy(a), to_numpy(b))
    # Advancing a clone must not alter the original's state.
    reference = rng.copy()
    clone.normal(size=100)
    np.testing.assert_array_equal(to_numpy(rng.normal(size=31)), to_numpy(reference.normal(size=31)))


def test_independent_streams_and_normal_statistics(backend):
    # Different fixed seeds: compare moments/correlation, never backend draws.
    n = 65536
    a = to_numpy(make_random_generator(backend, seed=102).normal(size=n))
    b = to_numpy(make_random_generator(backend, seed=203).normal(size=n))
    assert abs(a.mean()) < 6 / np.sqrt(n)
    assert abs(a.var() - 1) < 6 * np.sqrt(2 / (n - 1))
    assert abs(np.corrcoef(a, b)[0, 1]) < 6 / np.sqrt(n)
    assert abs(np.corrcoef(a[:-1], a[1:])[0, 1]) < 6 / np.sqrt(n)


def test_cupy_residency_and_global_state():
    cp = get_backend('cupy')
    global_state = cp.random.get_random_state()
    rng = make_random_generator(cp, seed=4)
    for method, kwargs in [('normal', {}), ('gamma', {}), ('poisson', {}),
                           ('uniform', {}), ('exponential', {}), ('choice', {'a': 20})]:
        samples = getattr(rng, method)(size=(8,), **kwargs)
        assert isinstance(samples, cp.ndarray)
        assert samples.device.id == cp.cuda.Device().id
    assert cp.random.get_random_state() is global_state
