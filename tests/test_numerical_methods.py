import numpy as np

from snnlab.core.numerical_methods import IzhikevichParameters, available_methods, get_stepper


def test_all_methods_produce_expected_shapes() -> None:
    v = np.array([-65.0, -64.0])
    u = 0.2 * v
    current = np.array([5.0, 6.0])
    params = IzhikevichParameters()

    assert len(available_methods()) == 7
    for method_id in available_methods():
        v_new, u_new = get_stepper(method_id)(v, u, current, 0.1, params)
        assert v_new.shape == v.shape
        assert u_new.shape == u.shape
        assert np.all(np.isfinite(v_new))
        assert np.all(np.isfinite(u_new))
