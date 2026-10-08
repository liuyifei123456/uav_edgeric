"""Paired A/B/C/D experiments with complete radio traces and honest time accounting."""
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

from envs.common import compute_jain_fairness_index
from envs.mubs_cov.mubs_cov import MultiUbsCoverageEnv
from scripts.compare_stage1 import scenario_inputs

CONFIGS = {"A": ("original", False), "B": ("edgeric_pf", False),
           "C": ("original", True), "D": ("edgeric_pf", True)}


def run_case(seed, label, positions, actions, dt_sched):
    mode, fast = CONFIGS[label]
    random.seed(seed)
    np.random.seed(seed)
    env = MultiUbsCoverageEnv("4ubs", record=False, scheduler_mode=mode,
                             fast_scheduling=fast, dt_sched=dt_sched, record_substeps=True)
    env.map = copy.copy(env.map)
    env.map.set_positions = lambda: {k: v.copy() for k,v in positions.items()}
    env.reset()
    reset_throughput = env.total_throughput
    radio_samples = list(env.last_radio_trace)
    rows = []
    gt_data = np.zeros(env.n_gts)
    window_utilities = []
    paths = [env.pos_ubs.copy()]
    period_rates = []
    for t, action in enumerate(actions, start=1):
        _, _, reward, done, info = env.step(action)
        assert env.t == t
        assert len(env.last_radio_trace) == (env.n_substeps if fast else 1)
        for sample in env.last_radio_trace:
            sched = sample["sched"]
            assert (sched.sum(axis=1) <= 1).all()
            assert (sched.sum(axis=(0,2)) <= 1).all()
            assert (sched.sum(axis=(1,2)) <= env.n_rbs).all()
            assert not sched[~sample["coverage"]].any()
            gt_data += sample["rate_per_gt"].astype(np.float64) * sample["duration_s"]
        radio_samples.extend(env.last_radio_trace)
        elapsed = t * env.dt
        window_fair = compute_jain_fairness_index(gt_data / elapsed)
        window_utilities.append(window_fair * env.rate_per_gt.mean())
        np.testing.assert_allclose(env.total_throughput-reset_throughput,
                                   gt_data.sum()/1e3, rtol=2e-7, atol=1e-10)
        row = dict(t=t, time_s=elapsed, TotalThroughput_Gb=env.total_throughput,
                   ResetThroughput_Gb=reset_throughput, ServiceThroughput_Gb=gt_data.sum()/1e3,
                   FairIdx=env.fair_idx, AvgGlobalUtility=env.avg_global_util,
                   WindowFairIdx=window_fair, WindowAvgUtility=np.mean(window_utilities),
                   rate_mbps=env.rate_per_gt.sum(), reward_mean=reward.mean(),
                   scheduling_count=env.scheduling_count)
        rows.append(row)
        paths.append(env.pos_ubs.copy())
        period_rates.append(env.rate_per_gt.copy())
    arrays = {key: np.stack([s[key] for s in radio_samples]) for key in radio_samples[0]}
    arrays.update(actions=actions, trajectory=np.stack(paths), initial_gt=positions["gt"],
                  period_rate_per_gt=np.stack(period_rates))
    valid = arrays["uav_step"] > 0
    alloc = arrays["sched"][valid]
    changes = int(np.any(alloc[1:] != alloc[:-1], axis=(1,2,3)).sum())
    metrics = dict(label=label, mode=mode, fast_scheduling=fast, seed=seed,
                   **{**rows[-1], **env.get_scheduling_stats()},
                   service_decisions=int(valid.sum()), rb_change_count=changes,
                   zero_service=bool(gt_data.sum()==0),
                   max_weight_span=float(np.ptp(arrays["weights"][valid], axis=0).max())
                       if mode=="edgeric_pf" else 0.)
    return pd.DataFrame(rows), arrays, metrics


def plot_scenario(output, scenario, tables, traces):
    fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for label, table in tables.items():
        mode, fast = CONFIGS[label]
        period = float(traces[label]["duration_s"][0])
        legend = f"{label}: {mode}, {period:g}s"
        for ax, metric, title in zip(axes.flat[:3],
                ["ServiceThroughput_Gb","WindowFairIdx","WindowAvgUtility"],
                ["Post-reset service data (Gb)", "Post-reset Jain fairness",
                 "Post-reset average utility (Mbps)"]):
            ax.plot(table.time_s, table[metric], label=legend)
            ax.set(xlabel="UAV simulation time (s)", ylabel=title)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    data = traces["D"]
    ids = np.argsort(-np.var(data["weights"],axis=0),kind="stable")[:4]
    for gt in ids:
        axes[1,1].plot(data["service_time_s"],data["weights"][:,gt],label=f"GT {gt}",linewidth=.8)
    axes[1,1].set(xlabel="Fast radio service time (s)",ylabel="D: PF weight")
    axes[1,1].legend()
    fig.suptitle(scenario+" | seed 0 | paired trajectories | aligned post-reset metrics")
    fig.savefig(output/(scenario+".png"),dpi=160)
    fig.savefig(output/(scenario+".pdf"))
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("../results/stage2/experiments"))
    parser.add_argument("--steps",type=int,default=20)
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,10,20])
    parser.add_argument("--dt-sched",type=float,default=1.)
    args=parser.parse_args()
    if not 1 <= args.steps <= 50:
        parser.error("steps must be in [1,50]")
    args.output.mkdir(parents=True,exist_ok=True)
    summary, paired, timing = [], [], {}
    for scenario in ("contention_hover","contention_motion"):
        for seed in args.seeds:
            positions, actions=scenario_inputs(seed,scenario,args.steps)
            tables,traces={},{}
            for label in CONFIGS:
                table,trace,metrics=run_case(seed,label,positions,actions,args.dt_sched)
                tables[label],traces[label]=table,trace
                summary.append(dict(scenario=scenario,**metrics))
                stem=f"{scenario}_seed{seed}_{label}"
                table.to_csv(args.output/(stem+".csv"),index=False)
                np.savez_compressed(args.output/(stem+".npz"),**trace)
                timing.setdefault(label,[]).extend(trace["compute_s"].tolist())
            for label in ("B","C","D"):
                np.testing.assert_array_equal(traces["A"]["trajectory"],traces[label]["trajectory"])
                np.testing.assert_array_equal(traces["A"]["actions"],traces[label]["actions"])
                np.testing.assert_array_equal(traces["A"]["initial_gt"],traces[label]["initial_gt"])
            for left,right in (("A","B"),("C","D")):
                a,b=traces[left],traces[right]
                valid=a["uav_step"]>0
                aa,bb=a["sched"][valid],b["sched"][valid]
                paired.append(dict(scenario=scenario,seed=seed,pair=left+"-"+right,
                    service_decisions=len(aa),
                    different_rb_decisions=int(np.any(aa!=bb,axis=(1,2,3)).sum()),
                    different_served_sets=int(np.any(aa.any(axis=(1,3))!=bb.any(axis=(1,3)),axis=1).sum()),
                    throughput_delta_Gb=tables[right].ServiceThroughput_Gb.iloc[-1]-
                                        tables[left].ServiceThroughput_Gb.iloc[-1]))
            if seed==args.seeds[0]:
                plot_scenario(args.output,scenario,tables,traces)
    frame=pd.DataFrame(summary)
    frame.to_csv(args.output/"summary.csv",index=False)
    mean=frame.groupby(["scenario","label"])[
        ["TotalThroughput_Gb","ServiceThroughput_Gb","FairIdx","AvgGlobalUtility",
         "WindowFairIdx","WindowAvgUtility","scheduling_count"]].mean()
    mean.to_csv(args.output/"summary_mean.csv")
    pd.DataFrame(paired).to_csv(args.output/"paired_comparison.csv",index=False)
    timing_rows=[]
    for label, samples in timing.items():
        values=np.array(samples)
        period=args.dt_sched if CONFIGS[label][1] else 40.
        timing_rows.append(dict(label=label,samples=len(values),period_s=period,
            mean_ms=values.mean()*1e3,p95_ms=np.percentile(values,95)*1e3,
            p99_ms=np.percentile(values,99)*1e3,max_ms=values.max()*1e3,
            over_period_count=int((values>period).sum())))
    pd.DataFrame(timing_rows).to_csv(args.output/"timing_pooled.csv",index=False)
    metadata=dict(configs=CONFIGS,seeds=args.seeds,steps=args.steps,dt_uav=40,dt_sched=args.dt_sched,
        motion="Version A: original action applied once, fixed UAV/GT positions within each period",
        traffic="Original saturated traffic/channel model; no new time-varying demand",
        reset="A/B keep stage1 reset service; C/D have no reset service",
        metrics="TotalThroughput/FairIdx/AvgGlobalUtility are native env metrics. ServiceThroughput excludes legacy reset. WindowFairIdx/WindowAvgUtility are evaluator-only aligned post-reset metrics; they do not modify rewards.",
        timing_scope="perf_counter CPU geometry + channel + weights + allocation + SINR/rates; excludes accumulation, trace copying, Python orchestration, and MARL GPU work. No hard-real-time claim.",
        base_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    (args.output/"experiment_config.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(mean.to_string())
    print(pd.DataFrame(paired).to_string(index=False))
    print(pd.DataFrame(timing_rows).to_string(index=False))


if __name__=="__main__":
    main()
