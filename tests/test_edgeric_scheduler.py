import numpy as np
import pytest

from edgeric import EdgeRICScheduler


def make_scheduler():
    # B=1 MHz, P=N0*B=1 W: rate=log2(1+gain) Mbps.
    return EdgeRICScheduler(1e6, 1.0, 1e-6, 0.1)


def test_pf_formula_and_monotonicity():
    scheduler = make_scheduler()
    gain = np.array([[1., 3., 1.]])
    history = np.array([1., 1., 0.5])
    weights = scheduler.compute_weights(gain, np.ones_like(gain),
                                        np.ones_like(gain, dtype=bool), history)
    np.testing.assert_allclose(weights, [1 / 1.1, 2 / 1.1, 1 / 0.6])
    assert weights[1] > weights[0]  # Better channel.
    assert weights[2] > weights[0]  # Less historical service.
    np.testing.assert_array_equal(scheduler.get_priority(), [1, 2, 0])


def test_zero_history_ties_and_no_coverage():
    scheduler = make_scheduler()
    weights = scheduler.compute_weights([[1., 1., 100.]], [[1., 1., 3.]],
                                        [[True, True, False]], [0., 0., 0.])
    np.testing.assert_allclose(weights, [10., 10., 0.])
    np.testing.assert_array_equal(scheduler.get_priority(), [0, 1, 2])
    np.testing.assert_array_equal(scheduler.selected_uav, [0, 0, -1])
    assert np.isfinite(weights).all()
    scheduler.reset()
    with pytest.raises(RuntimeError):
        scheduler.get_priority()


def test_local_weights_aggregate_nearest_covered_not_strongest():
    scheduler = make_scheduler()
    scheduler.compute_weights([[1., 3., 7.], [15., 7., 3.]],
                              [[1., 4., 2.], [2., 1., 2.]],
                              [[True, False, True], [True, True, True]],
                              [1., 1., 1.])
    np.testing.assert_array_equal(scheduler.selected_uav, [0, 1, 0])
    np.testing.assert_allclose(scheduler.local_weights,
                               [[1/1.1, 0., 3/1.1], [4/1.1, 3/1.1, 2/1.1]])
    np.testing.assert_allclose(scheduler.weights, [1/1.1, 3/1.1, 3/1.1])


def test_zero_gain_and_all_uncovered():
    scheduler = make_scheduler()
    for cover in (True, False):
        weights = scheduler.compute_weights(np.zeros((2, 3)), np.ones((2, 3)),
                                            np.full((2, 3), cover), np.zeros(3))
        np.testing.assert_array_equal(weights, np.zeros(3))
        assert np.isfinite(scheduler.local_weights).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.])
@pytest.mark.parametrize("field", ["gain", "distance", "history"])
def test_invalid_telemetry_rejected(field, bad):
    args = dict(channel_gains=[[1.]], distances=[[1.]], coverage=[[True]], avg_rate_mbps=[1.])
    key = {"gain": "channel_gains", "distance": "distances", "history": "avg_rate_mbps"}[field]
    args[key] = [bad] if field == "history" else [[bad]]
    with pytest.raises(ValueError):
        make_scheduler().compute_weights(**args)


def test_invalid_parameters_and_shapes():
    with pytest.raises(ValueError):
        EdgeRICScheduler(1e6, 1., 1e-6, 0.)
    with pytest.raises(ValueError):
        make_scheduler().compute_weights([[1.]], [[1., 2.]], [[True]], [0.])
