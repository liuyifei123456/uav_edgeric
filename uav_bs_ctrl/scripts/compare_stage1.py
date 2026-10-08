"""Paired open-loop scheduling experiments; no MARL training or policy claim."""
import argparse
import copy
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from envs.mubs_cov.mubs_cov import MultiUbsCoverageEnv


class TracedEnv(MultiUbsCoverageEnv):
    def _transmit_data(self):
        original_priority = self.prior_gts.copy()
        super()._transmit_data()
        self.used_priority = (original_priority if self.scheduler_mode == "original"
                              else self.prior_gts.copy())


def scenario_inputs(seed, scenario, steps):
    random.seed(seed)
    np.random.seed(seed)
    template = MultiUbsCoverageEnv("4ubs", record=False)
    rng = np.random.RandomState(seed + 1000)
    if scenario == "native_random":
        positions = template.map.set_positions()
        actions = rng.randint(0, template.n_actions, (steps, template.n_agents))
    else:
        # Deliberately synthetic overloaded overlapping cells. Map/radio/RB
        # parameters are still 4ubs defaults; these are not native map samples.
        uavs = np.array([[1000., 1000.], [1120., 1000.],
                         [2000., 2000.], [2120., 2000.]], dtype=np.float32)
        gts = np.vstack([rng.uniform(-60., 60., (25, 2)) + [1060., 1000.],
                         rng.uniform(-60., 60., (25, 2)) + [2060., 2000.]]).astype(np.float32)
        gts[-1] = [5000., 5000.]
        positions = dict(ubs=uavs, gt=gts)
        actions = np.zeros((steps, 4), dtype=int)
        if scenario == "contention_motion":
            # Legal original actions: +200m and -200m along x, then hover.
            # Both schedulers replay this same sequence with the same dt=40s.
            actions[::10, :] = 1
            actions[1::10, :] = 3
    return positions, actions


def run_mode(seed, mode, positions, actions):
    random.seed(seed)
    np.random.seed(seed)
    env = TracedEnv("4ubs", record=False, scheduler_mode=mode)
    env.map = copy.copy(env.map)
    env.map.set_positions = lambda: {k: v.copy() for k, v in positions.items()}
    env.reset()
    rows, buffers = [], {}
    for t in range(len(actions) + 1):
        if t:
            env.step(actions[t - 1])
        assert (env.sched.sum(axis=1) <= 1).all()
        assert (env.sched.sum(axis=(0, 2)) <= 1).all()
        assert (env.sched.sum(axis=(1, 2)) <= env.n_rbs).all()
        assert not env.sched[env.d_u2g > env.r_cov].any()
        rows.append(dict(t=t, time_s=t * env.dt, TotalThroughput=env.total_throughput,
                         FairIdx=env.fair_idx, AvgGlobalUtility=env.avg_global_util,
                         sum_rate_mbps=env.rate_per_gt.sum(),
                         served_gts=int(env.sched.sum()), reward_mean=env._get_reward().mean()))
        data = dict(sched=env.sched, rate_per_gt=env.rate_per_gt,
                    avg_rate_per_gt=env.avg_rate_per_gt,
                    priority=env.used_priority, pos_ubs=env.pos_ubs, pos_gts=env.pos_gts)
        if mode == "edgeric_pf":
            s = env.edgeric_scheduler
            data.update(weights=s.weights, local_weights=s.local_weights,
                        estimated_rates_mbps=s.estimated_rates_mbps,
                        local_rates_mbps=s.local_rates_mbps, history_before=s.history_mbps,
                        selected_uav=s.selected_uav)
            assert np.isfinite(s.weights).all()
        for key, value in data.items():
            buffers.setdefault(key, []).append(np.array(value, copy=True))
    arrays = {k: np.stack(v) for k, v in buffers.items()}
    arrays["actions"] = actions
    return pd.DataFrame(rows), arrays, dict(dt=env.dt, n_rbs=env.n_rbs,
        bw=env.bw, p_tx=env.p_tx, n0=env.n0, r_cov=env.r_cov, max_rate=env.max_rate,
        epsilon_mbps=0.05 * env.max_rate)


def plot_pair(output, scenario, seed, tables, traces):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for mode, label in (("original", "Original"), ("edgeric_pf", "EdgeRIC-PF")):
        df = tables[mode]
        for ax, metric, title in zip(axes.flat[:3],
            ["TotalThroughput", "FairIdx", "AvgGlobalUtility"],
            ["Cumulative data (Gb)", "Jain fairness", "Average global utility (Mbps)"]):
            ax.plot(df.time_s, df[metric], label=label)
            ax.set(xlabel="Trajectory time (s)", ylabel=title)
            ax.grid(alpha=0.25)
            ax.legend()
    pf = traces["edgeric_pf"]
    # A reproducible selection: highest variance of PF weight over the run.
    selected = np.argsort(-np.var(pf["weights"], axis=0), kind="stable")[:4]
    for gt in selected:
        axes[1, 1].plot(tables["edgeric_pf"].time_s, pf["weights"][:, gt], label=f"GT {gt}")
    axes[1, 1].set(xlabel="Trajectory time (s)", ylabel="PF weight (dimensionless)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)
    fig.suptitle(f"4ubs | {scenario} | seed {seed} | paired fixed action sequence")
    fig.savefig(output / f"{scenario}_seed{seed}.png", dpi=160)
    fig.savefig(output / f"{scenario}_seed{seed}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("../results/stage1"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 10, 20])
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.steps <= 50:
        parser.error("--steps must be between 1 and the 4ubs episode limit 50")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, comparisons = [], []
    for scenario in ("native_random", "contention_hover", "contention_motion"):
        for seed in args.seeds:
            positions, actions = scenario_inputs(seed, scenario, args.steps)
            tables, traces = {}, {}
            for mode in ("original", "edgeric_pf"):
                table, trace, config = run_mode(seed, mode, positions, actions)
                tables[mode], traces[mode] = table, trace
                stem = f"{scenario}_seed{seed}_{mode}"
                table.to_csv(args.output / (stem + ".csv"), index=False)
                np.savez_compressed(args.output / (stem + ".npz"), **trace)
                summaries.append(dict(scenario=scenario, seed=seed, mode=mode,
                    **table.iloc[-1].to_dict()))
            a, b = traces["original"], traces["edgeric_pf"]
            np.testing.assert_array_equal(a["pos_ubs"], b["pos_ubs"])
            np.testing.assert_array_equal(a["pos_gts"], b["pos_gts"])
            np.testing.assert_array_equal(a["actions"], b["actions"])
            comparisons.append(dict(scenario=scenario, seed=seed,
                frames_including_reset=args.steps + 1,
                different_priority_frames=int(np.any(a["priority"] != b["priority"], axis=1).sum()),
                different_rb_frames=int(np.any(a["sched"] != b["sched"], axis=(1, 2, 3)).sum()),
                different_served_set_frames=int(np.any(a["sched"].any(axis=(1, 3)) !=
                                                          b["sched"].any(axis=(1, 3)), axis=1).sum()),
                max_weight=float(b["weights"].max()),
                max_gt_weight_span=float(np.ptp(b["weights"], axis=0).max())))
            if seed == args.seeds[0]:
                plot_pair(args.output, scenario, seed, tables, traces)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(args.output / "allocation_comparison.csv", index=False)
    metrics = ["TotalThroughput", "FairIdx", "AvgGlobalUtility"]
    means = summary.groupby(["scenario", "mode"])[metrics].mean()
    means.to_csv(args.output / "summary_mean.csv")
    metadata = dict(map_id="4ubs", seeds=args.seeds, steps=args.steps, config=config,
        main_base_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        comparison="paired initial positions and open-loop actions; no trained policy",
        native="Unmodified map distribution; independent pre-generated random actions.",
        contention="Synthetic two overlapping UAV pairs, 49 nearby GTs, one uncovered GT.",
        reset_accounting="Upstream counts reset transmission too: 51 samples at 40s = 2040s of service; trajectory clock ends at 2000s.",
        fairness_caveat="Upstream clips all GT rates to >=1e-6 for Jain index; all-zero service returns approximately 1.",
        throughput_unit="Gb, per upstream rate_Mbps * dt_s / 1000, despite outdated Mb comment.")
    (args.output / "experiment_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(means.to_string())
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
