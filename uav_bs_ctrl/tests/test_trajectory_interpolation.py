import copy
import numpy as np
import pytest

from envs.mubs_cov.mubs_cov import MultiUbsCoverageEnv
from test_edgeric_integration import fixed_positions


def _env(mode="original", trajectory_mode="endpoint_hold", **kwargs):
    env = MultiUbsCoverageEnv(
        "4ubs", scheduler_mode=mode, fast_scheduling=True, dt_sched=1.0,
        trajectory_mode=trajectory_mode, record=False, record_substeps=True, **kwargs
    )
    fixed_positions(env)
    return env


def test_default_endpoint_hold_and_validation():
    env = MultiUbsCoverageEnv("4ubs", record=False, fast_scheduling=True)
    assert env.trajectory_mode == "endpoint_hold"
    with pytest.raises(ValueError, match="linear_interpolation"):
        MultiUbsCoverageEnv("4ubs", record=False, fast_scheduling=False,
                             trajectory_mode="linear_interpolation")
    with pytest.raises(ValueError, match="trajectory_mode"):
        MultiUbsCoverageEnv("4ubs", record=False, trajectory_mode="cubic")


@pytest.mark.parametrize("mode", ["original", "edgeric_pf"])
def test_linear_position_formula_and_endpoint(mode):
    env = _env(mode, "linear_interpolation")
    env.reset()
    start = env.pos_ubs.copy()
    action = np.array([1, 1, 1, 1])
    target = np.clip(start + env.avail_moves[action], 0, env.range_pos)
    env.step(action)
    assert len(env.trajectory_positions) == 40
    np.testing.assert_allclose(env.trajectory_positions[0], start + (target - start) / 40)
    np.testing.assert_allclose(env.trajectory_positions[19], start + 20 * (target - start) / 40)
    np.testing.assert_allclose(env.trajectory_positions[-1], target)
    np.testing.assert_allclose(env.pos_ubs, target)
    assert all(((p >= 0) & (p <= env.range_pos)).all() for p in env.trajectory_positions)
    assert env.t == 1 and env.scheduling_count == 40
    assert np.isfinite(env.rate_per_gt).all()


def test_hover_interpolation_matches_endpoint_hold():
    a = _env("original", "endpoint_hold")
    b = _env("original", "linear_interpolation")
    np.random.seed(7); a.reset()
    np.random.seed(7); b.reset()
    action = np.zeros(4, dtype=int)
    a.step(action); b.step(action)
    np.testing.assert_allclose(a.pos_ubs, b.pos_ubs)
    np.testing.assert_allclose(a.rate_per_gt, b.rate_per_gt, atol=1e-7)
    np.testing.assert_allclose(a.rate_per_ubs, b.rate_per_ubs, atol=1e-7)
    np.testing.assert_allclose(a.avg_rate_per_gt, b.avg_rate_per_gt, atol=1e-7)
    np.testing.assert_allclose(a.total_throughput, b.total_throughput)
    np.testing.assert_array_equal(a.sched, b.sched)
    assert len(b.trajectory_positions) == 40
    for p in b.trajectory_positions:
        np.testing.assert_allclose(p, b.pos_ubs)


def test_dynamic_geometry_and_channel_change():
    env = _env("original", "linear_interpolation")
    env.reset()
    start_dist = env.d_u2g.copy()
    start_gain = env.chan.estimate_chan_gain(start_dist, env.h_ubs)
    env.step([1, 1, 1, 1])
    assert any(not np.allclose(s["pos_ubs"], env.trajectory_positions[-1])
               for s in env.last_radio_trace)
    assert not np.allclose(env.d_u2g, start_dist)
    end_gain = env.chan.estimate_chan_gain(env.d_u2g, env.h_ubs)
    assert np.max(np.abs(start_gain - end_gain)) > 1e-14


def test_users_can_enter_or_leave_coverage_during_cycle():
    env = _env("original", "linear_interpolation")
    env.map = copy.copy(env.map)
    uavs = np.array([[1000., 1000.], [3000., 3000.],
                     [4000., 4000.], [4500., 4500.]], dtype=np.float32)
    gts = np.full((50, 2), [5000., 5000.], dtype=np.float32)
    gts[0] = [1050., 1000.]
    env.map.set_positions = lambda: dict(ubs=uavs.copy(), gt=gts.copy())
    env.reset()
    env.step([1, 0, 0, 0])
    covered = [bool(s["coverage"][0, 0]) for s in env.last_radio_trace]
    assert covered[0] and not covered[-1]
    assert any(covered) and not all(covered)


@pytest.mark.parametrize("mode", ["original", "edgeric_pf"])
def test_interpolation_preserves_rb_constraints_each_substep(mode):
    env = _env(mode, "linear_interpolation")
    env.reset()
    env.step([1, 0, 3, 0])
    for sample in env.last_radio_trace:
        sched = sample["sched"]
        assert (sched.sum(axis=1) <= 1).all()
        assert (sched.sum(axis=(0, 2)) <= 1).all()
        assert (sched.sum(axis=(1, 2)) <= env.n_rbs).all()
        assert not sched[~sample["coverage"]].any()
        assert np.isfinite(sample["rate_per_gt"]).all()
        assert np.isfinite(sample["channel_gain"]).all()


def test_substep_integral_and_history_use_real_service_time():
    env = _env("original", "linear_interpolation")
    env.reset()
    assert env.service_time_s == 0 and env.total_throughput == 0
    env.step([0, 0, 0, 0])
    data = sum((s["rate_per_gt"] * s["duration_s"] for s in env.last_radio_trace),
               np.zeros(env.n_gts))
    np.testing.assert_allclose(env.delivered_mbit_per_gt, data)
    np.testing.assert_allclose(env.avg_rate_per_gt, data / 40)
    np.testing.assert_allclose(env.total_throughput, data.sum() / 1000)
    assert env.service_time_s == 40
    env.reset()
    assert env.service_time_s == 0 and env.total_throughput == 0
    assert env.scheduling_count == 0 and env.delivered_mbit_per_gt.sum() == 0


def test_mid_period_collision_is_one_cycle_event():
    env = _env("original", "linear_interpolation")
    env.map = copy.copy(env.map)
    uavs = np.array([[1000., 1000.], [1020., 1000.],
                     [4000., 4000.], [4500., 4500.]], dtype=np.float32)
    gts = np.full((50, 2), [5000., 5000.], dtype=np.float32)
    env.map.set_positions = lambda: dict(ubs=uavs.copy(), gt=gts.copy())
    env.reset()
    assert env.n_colls == 0
    env.step([1, 3, 0, 0])
    assert env.n_colls == 1
    assert env.trajectory_collision_mask[:2].all()


def test_return_interface_and_period_reward():
    env = _env("edgeric_pf", "linear_interpolation")
    obs, state = env.reset()
    out = env.step([0, 0, 0, 0])
    assert len(out) == 5
    obs, state, reward, done, info = out
    assert len(obs) == env.n_agents and state.shape == (208,)
    assert reward.shape == (env.n_agents,) and np.isfinite(reward).all()
    assert info["EpLen"] == 1 and info["ProbCollision"] >= 0
