import numpy as np

from scripts.common import config


def _support():
    return next(w for w in config.WALLS if w["name"] == "support")


def test_sample_support_fills_gap_below_object():
    assert config.apply_object_placement("sample")

    support = _support()
    np.testing.assert_allclose(support["position"], [-0.1, 0.8, -0.0925])
    np.testing.assert_allclose(support["dimensions"], [0.2, 0.3, 0.165])


def test_raised_object_gets_taller_support():
    assert config.apply_object_placement("curved_structure")

    support = _support()
    np.testing.assert_allclose(support["position"], [-0.1, 0.8, 0.0075])
    np.testing.assert_allclose(support["dimensions"], [0.2, 0.3, 0.365])
