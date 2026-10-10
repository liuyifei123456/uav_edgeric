"""Evaluate formal-training checkpoints on independent scenarios and cross schedulers."""
import argparse, json, subprocess, sys
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algos.common import set_rand_seed, check_args_sanity
from algos.madrqn.config import DEFAULT_CONFIG
from algos.madrqn.learner import MultiAgentQLearner
from algos.madrqn.utils.env_wrappers import make_env
from scripts.formal_training import FormalScenarioEnv

def agent_args():
    cfg = dict(device="cuda", cuda_index=0, cuda_deterministic=False,
        o="gnn", c=None, hidden_size=256, n_layers=2, n_heads=4,
        msg_size=64, key_size=16, n_rounds=1, embed_dim=32, lr=2.5e-4,
        polyak=.999, gamma=.99, batch_size=32, replay_size=5000,
        max_seq_len=10, epochs=1, steps_per_epoch=1, update_after=1000,
        num_test_episodes=1, save_freq=1, decay_steps=200000, norm_r=True,
        anneal_lr=False, double_q=True, dueling=False, mixer=False,
        share_reward=False)
    cfg = dict(DEFAULT_CONFIG, **cfg)
    return check_args_sanity(type("Args", (), cfg)())

def run_episode(checkpoint, trained_mode, eval_mode, seed):
    set_rand_seed(seed)
    args = agent_args()
    env_kwargs = dict(map_id="4ubs", fair_service=True, avoid_collision=True,
        scheduler_mode=eval_mode, fast_scheduling=True, dt_sched=1.0,
        trajectory_mode="linear_interpolation", scene_profile="clustered_v1")
    wrapped = make_env(partial(FormalScenarioEnv, **env_kwargs, record=True), args)
    learner = MultiAgentQLearner(wrapped.get_env_info(), args)
    learner.load_checkpoint(str(checkpoint))
    (obs, _), hidden, done = wrapped.reset(), learner.init_hidden(), False
    rewards = []
    covered_steps = served_steps = 0
    rb_util = []
    while not done:
        obs = obs.to(args.device)
        action, hidden = learner.act(obs, hidden, 0.0)
        obs, _, reward, done, info = wrapped.step(action)
        rewards.append(float(np.mean(reward)))
        base = wrapped.env
        covered_steps += int(np.count_nonzero((base.d_u2g <= base.r_cov).any(axis=0)) > 0)
        served_steps += int(np.count_nonzero(base.rate_per_gt > 0) > 0)
        rb_util.append(float(base.sched.sum() / (base.n_ubs * base.n_rbs)))
    base = wrapped.env
    delivered = np.asarray(getattr(base, "delivered_mbit_per_gt", np.zeros(base.n_gts)))
    return dict(trained_mode=trained_mode, train_seed=int(checkpoint.parent.name.replace("seed","")),
        eval_mode=eval_mode, eval_seed=seed, checkpoint=str(checkpoint),
        EpRet=float(info["EpRet"]), EpLen=int(info["EpLen"]),
        TotalThroughput=float(info["TotalThroughput"]), FairIdx=float(info["FairIdx"]),
        AvgGlobalUtility=float(info["AvgGlobalUtility"]),
        GTServiceFraction=float(np.mean(delivered > 0)),
        UAVCollisionRate=float(info["ProbCollision"]), RBUtilization=float(np.mean(rb_util)),
        CoveredStepFraction=covered_steps / max(1, info["EpLen"]),
        ServedStepFraction=served_steps / max(1, info["EpLen"]),
        MeanGTRate=float(np.mean(base.rate_per_gt)), MeanReward=float(np.mean(rewards)))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--test-seeds",type=int,nargs="+",default=list(range(2000,2020)))
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for trained_mode in ("original","edgeric_pf"):
        for train_seed in (0,1,2):
            ckpt=args.root/trained_mode/f"seed{train_seed}"/"checkpoint_epoch12.pt"
            if not ckpt.exists(): raise FileNotFoundError(str(ckpt))
            for eval_mode in ("original","edgeric_pf"):
                for eval_seed in args.test_seeds:
                    rows.append(run_episode(ckpt,trained_mode,eval_mode,int(eval_seed)))
    frame=pd.DataFrame(rows); frame.to_csv(args.output/"cross_evaluation.csv",index=False)
    summary=frame.groupby(["trained_mode","eval_mode"]).agg(
      TotalThroughput_mean=("TotalThroughput","mean"),TotalThroughput_std=("TotalThroughput","std"),
      FairIdx_mean=("FairIdx","mean"),FairIdx_std=("FairIdx","std"),
      AvgGlobalUtility_mean=("AvgGlobalUtility","mean"),AvgGlobalUtility_std=("AvgGlobalUtility","std"),
      GTServiceFraction_mean=("GTServiceFraction","mean"),UAVCollisionRate_mean=("UAVCollisionRate","mean"),
      RBUtilization_mean=("RBUtilization","mean"),CoveredStepFraction_mean=("CoveredStepFraction","mean"),
      MeanGTRate_mean=("MeanGTRate","mean")).reset_index()
    summary.to_csv(args.output/"summary.csv",index=False); print(summary.to_string(index=False))
    meta=dict(training_seeds=[0,1,2],test_seeds=list(args.test_seeds),checkpoint_epoch=12,
      trajectory_mode="linear_interpolation",dt_sched=1.0,
      git_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    (args.output/"evaluation_config.json").write_text(json.dumps(meta,indent=2)+"\n")
if __name__=="__main__": main()
