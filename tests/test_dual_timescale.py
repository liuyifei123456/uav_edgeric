"""Stage-1 frozen-reference regression and independent radio accounting checks."""
import copy
import random
import subprocess
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from envs.common import compute_jain_fairness_index
from envs.mubs_cov.mubs_cov import MultiUbsCoverageEnv
from test_edgeric_integration import fixed_positions, equal_nested

STAGE1_SHA = "4eff8b9ad05873c873cd245b4058da9fc5d86b28"


def frozen_stage1():
    source = subprocess.check_output(
        ["git", "show", STAGE1_SHA + ":envs/mubs_cov/mubs_cov.py"], text=True,
        cwd=Path(__file__).resolve().parents[1])
    module = types.ModuleType("stage1_frozen")
    exec(compile(source, "stage1_frozen.py", "exec"), module.__dict__)
    return module.MultiUbsCoverageEnv


def make_fast(mode="edgeric_pf", dt=1., **kwargs):
    env = MultiUbsCoverageEnv("4ubs", record=False, scheduler_mode=mode,
                              fast_scheduling=True, dt_sched=dt, record_substeps=True, **kwargs)
    fixed_positions(env)
    return env


def assert_sample_constraints(sample):
    sched = sample["sched"]
    assert (sched.sum(axis=1) <= 1).all()
    assert (sched.sum(axis=(0, 2)) <= 1).all()
    assert (sched.sum(axis=(1, 2)) <= 5).all()
    assert not sched[~sample["coverage"]].any()
    for key in ("rate_per_gt", "rate_per_ubs", "history_before", "history_after"):
        assert np.isfinite(sample[key]).all()
    if "weights" in sample:
        assert np.isfinite(sample["weights"]).all()


@pytest.mark.parametrize("mode", ["original", "edgeric_pf"])
@pytest.mark.parametrize("seed", [0, 10, 20])
@pytest.mark.parametrize("explicit", [False, True])
def test_nonfast_exact_stage1_regression(mode, seed, explicit):
    reference = frozen_stage1()("4ubs", record=False, scheduler_mode=mode)
    actual = MultiUbsCoverageEnv("4ubs", record=False, scheduler_mode=mode,
                               **({"fast_scheduling": False} if explicit else {}))
    for env in (reference, actual):
        fixed_positions(env)
    actions = np.zeros((50, 4), dtype=int)
    actions[::10] = 1
    actions[1::10] = 3
    def rollout(env):
        random.seed(seed)
        np.random.seed(seed)
        output = env.reset()
        rows = []
        for action in [None] + list(actions):
            if action is not None:
                output = env.step(action)
            row = dict(output=output, sched=env.sched, rate=env.rate_per_gt,
                       ubs_rate=env.rate_per_ubs, history=env.avg_rate_per_gt,
                       throughput=env.total_throughput, fairness=env.fair_idx,
                       util=env.global_util, avg_util=env.avg_global_util,
                       priorities=env.prior_gts, collisions=env.n_colls,
                       reward=env._get_reward())
            if mode == "edgeric_pf":
                row["pf_state"] = env.edgeric_scheduler.__dict__
            rows.append(copy.deepcopy(row))
        return rows
    equal_nested(rollout(actual), rollout(reference))


@pytest.mark.parametrize("mode", ["original", "edgeric_pf"])
def test_fast_reset_is_zero_duration(mode):
    env = make_fast(mode)
    obs, state = env.reset()
    assert env.t == env.scheduling_count == env.service_time_s == env.total_throughput == 0
    assert env.n_colls == env.global_util == env.avg_global_util == 0
    assert not env.sched.any()
    np.testing.assert_array_equal(env.rate_per_gt, np.zeros(50))
    np.testing.assert_array_equal(env.avg_rate_per_gt, np.zeros(50))
    assert env.last_radio_trace == []
    assert env.scheduling_durations_s == []
    assert len(obs) == 4 and state.shape == (208,)
    env.step([0] * 4)
    assert env.total_throughput > 0
    env.reset()
    assert env.scheduling_count == 0 and env.total_throughput == 0


@pytest.mark.parametrize("mode", ["original", "edgeric_pf"])
def test_40_decisions_per_uav_step_and_history_integrals(mode):
    env = make_fast(mode)
    env.reset()
    data_gt = np.zeros(50)
    utility = []
    for t, action in enumerate(([0]*4, [1]*4, [3]*4), start=1):
        before_count = env.scheduling_count
        obs, state, reward, done, info = env.step(action)
        assert env.t == t and env.scheduling_count - before_count == 40
        assert env.scheduling_count == 40 * t and env.service_time_s == 40 * t
        assert env.n_substeps == 40 and env.dt == env.dt_uav == 40 and env.n_actions == 9
        assert len(env.last_radio_trace) == 40
        period_rates = []
        for k, sample in enumerate(env.last_radio_trace):
            assert_sample_constraints(sample)
            elapsed = (t - 1) * 40 + k
            np.testing.assert_allclose(sample["history_before"],
                                       data_gt / elapsed if elapsed else np.zeros(50),
                                       rtol=1e-12, atol=1e-12)
            data_gt += sample["rate_per_gt"].astype(np.float64) * sample["duration_s"]
            np.testing.assert_allclose(sample["history_after"], data_gt / (elapsed + 1))
            if mode == "edgeric_pf":
                # Independent channel-based oracle at this actual position.
                gain = env.chan.estimate_chan_gain(env.d_u2g, env.h_ubs)
                nearest = np.argmin(np.where(sample["coverage"], env.d_u2g, np.inf), axis=0)
                rate = env.bw * np.log2(1 + env.p_tx * gain / (env.n0 * env.bw)) * 1e-6
                expected = rate[nearest, np.arange(50)] / (
                    sample["history_before"] + env.edgeric_scheduler.epsilon_mbps)
                expected[~sample["coverage"].any(axis=0)] = 0
                np.testing.assert_allclose(sample["weights"], expected, rtol=1e-6)
            elif k:
                np.testing.assert_array_equal(sample["priority"], np.argsort(
                    env.last_radio_trace[k-1]["history_after"]))
            np.testing.assert_array_equal(sample["pos_ubs"], env.pos_ubs)
            period_rates.append(sample["rate_per_gt"])
        np.testing.assert_allclose(env.total_throughput, data_gt.sum() / 1e3)
        np.testing.assert_allclose(env.avg_rate_per_gt, data_gt / (40*t))
        np.testing.assert_allclose(env.rate_per_gt, np.mean(period_rates, axis=0, dtype=np.float64))
        np.testing.assert_allclose(env.rate_per_ubs, np.mean(
            [s["rate_per_ubs"] for s in env.last_radio_trace], axis=0, dtype=np.float64))
        np.testing.assert_array_equal(env.instant_rate_per_gt, env.last_radio_trace[-1]["rate_per_gt"])
        assert env.fair_idx == compute_jain_fairness_index(env.avg_rate_per_gt)
        assert env.global_util == env.fair_idx * env.rate_per_gt.mean()
        utility.append(env.global_util)
        np.testing.assert_allclose(env.avg_global_util, np.mean(utility))
        expected_reward = env.reward_scale_rate * env.global_util / env.max_rate
        np.testing.assert_allclose(reward, expected_reward * (env.rate_per_ubs > 0), rtol=1e-6)
        assert reward.shape == (4,) and state.shape == (208,) and len(obs) == 4
        assert set(info) == {"EpRet", "EpLen", "AvgGlobalUtility", "FairIdx",
                             "TotalThroughput", "ProbCollision", "BadMask"}
        for i in range(4):
            visible = env.d_u2g[i] <= env.r_sns
            np.testing.assert_allclose(obs[i]["gt"][visible, 3],
                                       env.rate_per_gt[visible] / env.max_rate, rtol=1e-6)
        assert np.isfinite(state).all() and np.isfinite(reward).all()


@pytest.mark.parametrize("dt", [0.5, 1., 2., 5., 10., 40.])
@pytest.mark.parametrize("mode", ["original", "edgeric_pf"])
def test_period_units_against_analytic_constant_link(dt, mode):
    # One isolated covered GT, no scheduling contention/interference:
    # changing the number of scheduling samples cannot multiply its data.
    env = make_fast(mode, dt)
    uavs = np.array([[1000.,1000.],[2000.,2000.],[3000.,3000.],[4000.,4000.]])
    gts = np.tile([5500.,5500.], (50,1))
    gts[0] = [1000.,1000.]
    env.map.set_positions = lambda: dict(ubs=uavs.copy(), gt=gts.copy())
    env.reset()
    env.step([0]*4)
    rate = env.bw * np.log2(1+env.p_tx*env.chan.estimate_chan_gain(0,env.h_ubs) /
                            (env.n0*env.bw)) * 1e-6
    assert env.scheduling_count == round(40/dt)
    np.testing.assert_allclose(env.rate_per_gt[0], rate, rtol=1e-6)
    np.testing.assert_allclose(env.total_throughput, rate*40/1e3, rtol=1e-6)
    np.testing.assert_allclose(env.avg_rate_per_gt[0], rate, rtol=1e-6)
    env.step([0]*4)
    np.testing.assert_allclose(env.total_throughput, rate*80/1e3, rtol=1e-6)
    assert env.service_time_s == 80


@pytest.mark.parametrize("dt", [1., 40.])
def test_collision_count_once_per_uav_period(dt):
    env = make_fast(dt=dt)
    positions = env.map.set_positions()
    positions["ubs"][1] = positions["ubs"][0]  # One colliding pair.
    env.map.set_positions = lambda: copy.deepcopy(positions)
    env.reset()
    assert env.n_colls == 0
    for t in range(1, 4):
        _, _, reward, _, info = env.step([0]*4)
        assert env.n_colls == t
        assert info["ProbCollision"] == 1.
        np.testing.assert_array_equal(reward[:2], [-env.penalty, -env.penalty])


@pytest.mark.parametrize("mode", ["original", "edgeric_pf"])
def test_no_coverage_is_finite_and_zero_service(mode):
    env = make_fast(mode)
    positions = env.map.set_positions()
    positions["gt"][:] = [5500.,5500.]
    env.map.set_positions = lambda: copy.deepcopy(positions)
    env.reset()
    _, _, reward, _, _ = env.step([0]*4)
    assert env.total_throughput == 0 and not env.sched.any()
    assert np.isfinite(env.fair_idx) and np.isfinite(reward).all()
    assert all(not s["sched"].any() for s in env.last_radio_trace)


@pytest.mark.parametrize("dt", [0, -1, np.nan, np.inf, 3, 41, True, "bad", None])
def test_invalid_period_rejected(dt):
    with pytest.raises(ValueError, match="dt_sched"):
        make_fast(dt=dt)


def test_nonfast_period_need_not_divide_uav_dt():
    env = MultiUbsCoverageEnv("4ubs", fast_scheduling=False, dt_sched=3, record=False)
    assert env.n_substeps == 1


def test_latency_percentiles_and_deadline_exceedance():
    env = make_fast()
    env.reset()
    # Deterministic time source verifies summaries without making flaky
    # assertions about the runtime of the test machine.
    durations = np.arange(1, 41) * .04
    clock_values = []
    start = 0.
    for duration in durations:
        clock_values.extend([start, start + duration])
        start += 2.
    with patch("envs.mubs_cov.mubs_cov.perf_counter", side_effect=clock_values):
        env.step([0]*4)
    stats = env.get_scheduling_stats()
    assert stats["scheduling_count"] == 40 and stats["dt_sched_s"] == 1
    assert stats["initialization_scheduling_count"] == 0
    np.testing.assert_allclose(stats["mean_compute_s"], durations.mean())
    np.testing.assert_allclose(stats["p95_compute_s"], np.percentile(durations,95))
    np.testing.assert_allclose(stats["p99_compute_s"], np.percentile(durations,99))
    assert stats["over_period_count"] == 15


def test_fast_episode_termination_still_counts_uav_actions():
    env = make_fast(dt=40)
    env.reset()
    for i in range(1,51):
        _, _, _, done, info = env.step([0]*4)
        assert done == (i==50)
        assert info["BadMask"] == (i==50)
    assert env.t == 50 and env.service_time_s == 2000
