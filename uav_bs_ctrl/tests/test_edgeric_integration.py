import copy
import random
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest

from envs.mubs_cov.mubs_cov import MultiUbsCoverageEnv

UPSTREAM_SHA = "5ef26f2c0c324a5888eca999beff3d42b6137ace"


def fixed_positions(env):
    """Contention, overlapping coverage, and one definitely uncovered GT."""
    env.map = copy.copy(env.map)
    uavs = np.array([[1000., 1000.], [1120., 1000.],
                     [2000., 2000.], [2120., 2000.]], dtype=np.float32)
    rng = np.random.RandomState(123)
    gts = np.vstack([rng.uniform([-60., -60.], [60., 60.], (25, 2)) + [1060., 1000.],
                     rng.uniform([-60., -60.], [60., 60.], (25, 2)) + [2060., 2000.]]).astype(np.float32)
    gts[-1] = [5000., 5000.]
    env.map.set_positions = lambda: dict(ubs=uavs.copy(), gt=gts.copy())


def assert_constraints(env):
    assert env.sched.shape == (env.n_ubs, env.n_gts, env.n_rbs)
    assert (env.sched.sum(axis=1) <= 1).all()  # one GT / UAV / RB
    assert (env.sched.sum(axis=(0, 2)) <= 1).all()  # one RB / GT
    assert (env.sched.sum(axis=(1, 2)) <= env.n_rbs).all()
    assert not env.sched[env.d_u2g > env.r_cov].any()
    assert np.isfinite(env.rate_per_gt).all()


@pytest.mark.parametrize("mode", ["original", "edgeric_pf"])
def test_reset_step_constraints_and_interfaces(mode):
    env = MultiUbsCoverageEnv("4ubs", scheduler_mode=mode, record=False)
    fixed_positions(env)
    random.seed(42)
    np.random.seed(42)
    obs, state = env.reset()
    assert env.dt == 40 and env.n_actions == 9
    assert len(obs) == 4 and state.shape == (208,)
    assert set(obs[0]) == {"agent", "ubs", "gt"}
    assert obs[0]["agent"].shape == (2,)
    assert obs[0]["ubs"].shape == (3, 3)
    assert obs[0]["gt"].shape == (50, 5)
    assert env.action_space[0].n == 9
    assert_constraints(env)
    assert not env.sched[:, -1, :].any()
    assert (env.sched.sum(axis=(1, 2)) > 0).all()
    if mode == "edgeric_pf":
        assert np.max(env.edgeric_scheduler.weights) <= 20. + 1e-9
        np.testing.assert_array_equal(env.edgeric_scheduler.history_mbps, np.zeros(50))
    for t in range(1, 51):
        history = env.avg_rate_per_gt.copy()
        out = env.step([0] * 4)
        assert len(out) == 5
        obs, state, reward, done, info = out
        assert reward.shape == (4,)
        assert done == (t == 50)
        assert set(info) == {"EpRet", "EpLen", "AvgGlobalUtility", "FairIdx",
                             "TotalThroughput", "ProbCollision", "BadMask"}
        assert_constraints(env)
        if mode == "edgeric_pf":
            np.testing.assert_array_equal(env.edgeric_scheduler.history_mbps, history)
            np.testing.assert_allclose(env.edgeric_scheduler.weights,
                env.edgeric_scheduler.estimated_rates_mbps /
                (history + env.edgeric_scheduler.epsilon_mbps), rtol=1e-6)
            np.testing.assert_array_equal(env.prior_gts, env.edgeric_scheduler.get_priority())
        np.testing.assert_array_equal(env.avg_rate_per_gt,
                                      (history * t + env.rate_per_gt) / (t + 1))
    env.reset()
    if mode == "edgeric_pf":
        np.testing.assert_array_equal(env.edgeric_scheduler.history_mbps, np.zeros(50))


def test_nearest_full_falls_back_to_other_covered_uav():
    env = MultiUbsCoverageEnv("4ubs", scheduler_mode="edgeric_pf", record=False)
    fixed_positions(env)
    env.map.set_positions = lambda: dict(
        ubs=np.array([[1000., 1000.], [1020., 1000.], [3000., 3000.], [4000., 4000.]]),
        gt=np.tile([1001., 1000.], (50, 1)))
    env.reset()
    np.testing.assert_array_equal(env.edgeric_scheduler.selected_uav, np.zeros(50))
    np.testing.assert_array_equal(env.sched.sum(axis=(1, 2)), [5, 5, 0, 0])
    assert_constraints(env)


def equal_nested(actual, expected):
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equal_nested(a, b)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal_nested(actual[key], expected[key])
    else:
        assert actual == expected


@pytest.mark.parametrize("seed", [0, 10, 20])
@pytest.mark.parametrize("layout", ["native", "contention"])
@pytest.mark.parametrize("mode_kwargs", [{}, {"scheduler_mode": "original"}])
def test_original_exact_upstream_regression(seed, layout, mode_kwargs):
    # Execute the immutable Git object, never a copy of the modified class.
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ["git", "show", UPSTREAM_SHA + ":envs/mubs_cov/mubs_cov.py"], cwd=root, text=True)
    upstream = types.ModuleType("upstream_mubs_cov")
    exec(compile(source, "upstream_mubs_cov.py", "exec"), upstream.__dict__)
    reference = upstream.MultiUbsCoverageEnv("4ubs", record=False)
    actual = MultiUbsCoverageEnv("4ubs", record=False, **mode_kwargs)
    if layout == "contention":
        fixed_positions(reference)
        fixed_positions(actual)
    actions = np.random.RandomState(seed + 1000).randint(0, 9, (50, 4))

    def run(env):
        random.seed(seed)
        np.random.seed(seed)
        output = env.reset()
        frames = []
        for action in [None] + actions.tolist():
            if action is not None:
                output = env.step(action)
            frames.append(copy.deepcopy(dict(output=output, sched=env.sched,
                rate=env.rate_per_gt, avg_rate=env.avg_rate_per_gt,
                throughput=env.total_throughput, fairness=env.fair_idx,
                utility=env.avg_global_util, reward=env._get_reward(),
                priorities=env.prior_gts, pos_ubs=env.pos_ubs, pos_gts=env.pos_gts)))
        return frames
    equal_nested(run(actual), run(reference))


def test_current_channel_used_before_rb_allocation():
    env = MultiUbsCoverageEnv("4ubs", scheduler_mode="edgeric_pf", record=False)
    fixed_positions(env)
    env.reset()
    old_rates = env.edgeric_scheduler.local_rates_mbps.copy()
    old_history = env.avg_rate_per_gt.copy()
    env.step([1, 1, 1, 1])
    assert not np.array_equal(old_rates, env.edgeric_scheduler.local_rates_mbps)
    gains = env.chan.estimate_chan_gain(env.d_u2g, env.h_ubs)
    expected = env.bw * np.log2(1 + env.p_tx * gains / (env.n0 * env.bw)) * 1e-6
    expected[env.d_u2g > env.r_cov] = 0
    np.testing.assert_allclose(env.edgeric_scheduler.local_rates_mbps, expected, rtol=1e-6)
    np.testing.assert_array_equal(env.edgeric_scheduler.history_mbps, old_history)


def test_invalid_mode():
    with pytest.raises(ValueError):
        MultiUbsCoverageEnv("4ubs", scheduler_mode="pf_typo")
