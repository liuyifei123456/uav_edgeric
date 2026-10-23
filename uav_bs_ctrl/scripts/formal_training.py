"""Bounded formal GNN/MARL training with a reproducible covered 4ubs scene profile."""
import argparse
import copy
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from algos.madrqn.run import train
from envs.mubs_cov.mubs_cov import MultiUbsCoverageEnv
from algos.common import set_rand_seed


class FormalScenarioEnv(MultiUbsCoverageEnv):
    """4ubs radio parameters with a sampled, covered training layout.

    The native MAPS['4ubs'] object is not modified. Each reset samples two
    nearby GT clusters and two UAVs around each cluster, preserving the
    original action space and communication model while avoiding all-zero
    training episodes.
    """
    scene_profile = "clustered_v1"

    def __init__(self, *args, scene_profile="clustered_v1", **kwargs):
        if scene_profile != self.scene_profile:
            raise ValueError("unsupported formal scene profile")
        super().__init__(*args, **kwargs)
        self.scene_profile = scene_profile
        self.map = copy.copy(self.map)
        self.map.set_positions = self._sample_positions

    def _sample_positions(self):
        jitter = np.random.uniform(-250.0, 250.0, size=(2, 2))
        centers = np.array([[1500.0, 1500.0], [4500.0, 4500.0]], dtype=np.float32) + jitter
        uav_offsets = np.array([[-80., 0.], [80., 0.]], dtype=np.float32)
        uavs = np.vstack([centers[0] + uav_offsets, centers[1] + uav_offsets])
        uavs = np.clip(uavs, 0, self.range_pos).astype(np.float32)
        gt_parts = [
            centers[i] + np.random.uniform(-55.0, 55.0, size=(25, 2))
            for i in range(2)
        ]
        gts = np.vstack(gt_parts).astype(np.float32)
        gts[-1] = np.array([5900.0, 5900.0], dtype=np.float32)
        return {"ubs": uavs, "gt": np.clip(gts, 0, self.range_pos)}

    def reset(self):
        return super().reset()


def make_train_kwargs(steps_per_epoch, epochs, update_after, batch_size,
                      max_seq_len, device):
    return dict(
        device=device, cuda_index=0, cuda_deterministic=False,
        o="gnn", c=None, hidden_size=256, n_layers=2, n_heads=4,
        msg_size=64, key_size=16, n_rounds=1, embed_dim=32,
        lr=2.5e-4, polyak=0.999, gamma=0.99, batch_size=batch_size,
        replay_size=5000, max_seq_len=max_seq_len,
        epochs=epochs, steps_per_epoch=steps_per_epoch,
        update_after=update_after, num_test_episodes=1, save_freq=1,
        decay_steps=200000, norm_r=True, anneal_lr=True,
        double_q=True, dueling=False, mixer=False, share_reward=False,
    )


def diagnose(mode, seed, episodes, output):
    set_rand_seed(seed)
    env = FormalScenarioEnv("4ubs", record=False, scheduler_mode=mode,
        fast_scheduling=True, dt_sched=1.0,
        trajectory_mode="linear_interpolation")
    rows = []
    rng = np.random.RandomState(seed + 700000)
    for episode in range(episodes):
        env.reset()
        ep = {"mode": mode, "seed": seed, "episode": episode,
              "covered_steps": 0, "served_steps": 0, "nonzero_reward_steps": 0,
              "covered_gt": [], "served_gt": [], "rb_util": [],
              "reward": [], "throughput": []}
        for _ in range(env.episode_limit):
            ep["covered_gt"].append(int(np.count_nonzero(
                (env.d_u2g <= env.r_cov).any(axis=0))))
            action = rng.randint(0, env.n_actions, env.n_agents)
            _, _, reward, _, info = env.step(action)
            ep["covered_steps"] += int(ep["covered_gt"][-1] > 0)
            served = int(np.count_nonzero(env.rate_per_gt > 0))
            ep["served_gt"].append(served)
            ep["served_steps"] += int(served > 0)
            ep["nonzero_reward_steps"] += int(np.any(np.asarray(reward) != 0))
            ep["rb_util"].append(float(env.sched.sum() /
                (env.n_ubs * env.n_rbs)))
            ep["reward"].append(float(np.mean(reward)))
            ep["throughput"].append(float(env.total_throughput))
        rows.append({
            "mode": mode, "seed": seed, "episode": episode,
            "covered_step_fraction": ep["covered_steps"] / env.episode_limit,
            "served_step_fraction": ep["served_steps"] / env.episode_limit,
            "nonzero_reward_fraction": ep["nonzero_reward_steps"] / env.episode_limit,
            "mean_covered_gt": np.mean(ep["covered_gt"]),
            "mean_served_gt": np.mean(ep["served_gt"]),
            "mean_gt_service_fraction": np.mean(ep["served_gt"]) / env.n_gts,
            "mean_rb_utilization": np.mean(ep["rb_util"]),
            "mean_reward": np.mean(ep["reward"]),
            "final_throughput_gb": ep["throughput"][-1],
            "fair_idx": float(env.fair_idx),
            "avg_global_util": float(env.avg_global_util),
            "collision_rate": float(env.n_colls / env.t),
        })
    frame = __import__("pandas").DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(frame.mean(numeric_only=True).to_string())


def train_one(mode, seed, output, steps, steps_per_epoch, update_after,
              batch_size, max_seq_len, device):
    if steps % steps_per_epoch:
        raise ValueError("steps must be divisible by steps_per_epoch")
    epochs = steps // steps_per_epoch
    output.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(
        map_id="4ubs", fair_service=True, avoid_collision=True,
        scheduler_mode=mode, fast_scheduling=True, dt_sched=1.0,
        trajectory_mode="linear_interpolation", scene_profile="clustered_v1",
    )
    train_kwargs = make_train_kwargs(
        steps_per_epoch, epochs, update_after, batch_size, max_seq_len, device)
    logger_kwargs = {"output_dir": str(output),
                     "exp_name": f"formal_{mode}_seed{seed}"}
    started = time.time()
    train(FormalScenarioEnv, env_kwargs, seed, train_kwargs, logger_kwargs)
    checkpoint = output / f"checkpoint_epoch{epochs}.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"missing final checkpoint: {checkpoint}")
    updates = sum(1 for t in range(steps)
                  if t >= max(update_after, batch_size * max_seq_len)
                  and t % max_seq_len == 0)
    metadata = dict(mode=mode, seed=seed, scene_profile="clustered_v1",
        steps=steps, steps_per_epoch=steps_per_epoch, epochs=epochs,
        update_after=update_after, update_every=max_seq_len,
        batch_size=batch_size, max_seq_len=max_seq_len,
        optimizer_updates=updates, checkpoint=str(checkpoint),
        network="gnn hidden_size=256 n_layers=2 protocol=None",
        dt_uav=40.0, dt_sched=1.0, trajectory_mode="linear_interpolation",
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        torch=torch.__version__, cuda=torch.version.cuda,
        elapsed_seconds=time.time() - started,
        git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                        text=True).strip())
    (output / "formal_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("diagnose")
    d.add_argument("--mode", choices=("original", "edgeric_pf"), required=True)
    d.add_argument("--seed", type=int, required=True)
    d.add_argument("--episodes", type=int, default=5)
    d.add_argument("--output", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--mode", choices=("original", "edgeric_pf"), required=True)
    t.add_argument("--seed", type=int, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--steps-per-epoch", type=int, default=500)
    t.add_argument("--update-after", type=int, default=200)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--max-seq-len", type=int, default=10)
    t.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.command == "diagnose":
        diagnose(args.mode, args.seed, args.episodes, args.output)
    else:
        train_one(args.mode, args.seed, args.output, args.steps,
                  args.steps_per_epoch, args.update_after, args.batch_size,
                  args.max_seq_len, args.device)


if __name__ == "__main__":
    main()
