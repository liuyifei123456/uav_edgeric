import json
from pathlib import Path
import numpy as np
import pandas as pd
from algos.common import set_rand_seed
from scripts.formal_training import FormalScenarioEnv

RESULTS = Path("/workspace/uav_edgeric_stage1/results/stage3/formal_training")

def test_formal_environment_scheduler_modes_and_config():
    for mode in ("original", "edgeric_pf"):
        env = FormalScenarioEnv("4ubs", scheduler_mode=mode, fast_scheduling=True,
                                dt_sched=1.0, trajectory_mode="linear_interpolation",
                                scene_profile="clustered_v1")
        assert env.scheduler_mode == mode
        assert env.fast_scheduling is True
        assert env.dt_sched == 1.0
        assert env.trajectory_mode == "linear_interpolation"

def test_clustered_profile_produces_service_signal():
    set_rand_seed(123)
    env = FormalScenarioEnv("4ubs", scheduler_mode="original", fast_scheduling=True,
                            dt_sched=1.0, trajectory_mode="linear_interpolation",
                            scene_profile="clustered_v1")
    env.reset()
    _, _, reward, _, _ = env.step(np.zeros(env.n_agents, dtype=int))
    assert np.isfinite(reward).all()
    assert env.total_throughput > 0
    assert np.count_nonzero(env.rate_per_gt > 0) > 0

def test_train_and_test_seed_sets_are_disjoint():
    train = {0, 1, 2}
    validation = set(range(1000, 1010))
    test = set(range(2000, 2020))
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)

def test_calibration_checkpoint_and_updates_exist():
    metadata = RESULTS / "calibration/original_seed0/formal_metadata.json"
    checkpoint = RESULTS / "calibration/original_seed0/checkpoint_epoch2.pt"
    assert metadata.exists() and checkpoint.exists()
    data = json.loads(metadata.read_text())
    assert data["steps"] == 1000
    assert data["optimizer_updates"] > 0
    assert data["checkpoint"].endswith("checkpoint_epoch2.pt")

def test_cross_evaluation_has_all_four_groups_and_finite_metrics():
    csv = RESULTS / "cross_evaluation/cross_evaluation.csv"
    assert csv.exists()
    frame = pd.read_csv(csv)
    groups = set(zip(frame.trained_mode, frame.eval_mode))
    assert groups == {("original", "original"), ("original", "edgeric_pf"),
                      ("edgeric_pf", "original"), ("edgeric_pf", "edgeric_pf")}
    assert len(frame) == 240
    metrics = ["TotalThroughput", "FairIdx", "AvgGlobalUtility", "GTServiceFraction",
               "UAVCollisionRate", "RBUtilization", "MeanGTRate"]
    assert np.isfinite(frame[metrics].to_numpy()).all()
