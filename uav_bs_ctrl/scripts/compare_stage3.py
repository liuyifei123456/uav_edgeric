"""Stage-3 fixed-trajectory paired experiments for dynamic UAV geometry."""
import argparse, copy, json, random, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from envs.mubs_cov.mubs_cov import MultiUbsCoverageEnv
from scripts.compare_stage1 import scenario_inputs

CONFIGS = {
    "A": ("endpoint_hold", "original"),
    "B": ("endpoint_hold", "edgeric_pf"),
    "C": ("linear_interpolation", "original"),
    "D": ("linear_interpolation", "edgeric_pf"),
}

def _env(seed, trajectory_mode, scheduler_mode, positions, dt_sched):
    random.seed(seed); np.random.seed(seed)
    env = MultiUbsCoverageEnv(
        "4ubs", record=False, scheduler_mode=scheduler_mode,
        fast_scheduling=True, dt_sched=dt_sched,
        trajectory_mode=trajectory_mode, record_substeps=True)
    env.map = copy.copy(env.map)
    env.map.set_positions = lambda: {k: v.copy() for k, v in positions.items()}
    return env

def run_case(seed, label, positions, actions, dt_sched):
    trajectory_mode, scheduler_mode = CONFIGS[label]
    env = _env(seed, trajectory_mode, scheduler_mode, positions, dt_sched)
    env.reset()
    rows, traces = [], []
    delivered = np.zeros(env.n_gts, dtype=np.float64)
    for t, action in enumerate(actions, 1):
        _, _, reward, done, info = env.step(action)
        assert env.t == t and len(env.last_radio_trace) == env.n_substeps
        step_data = np.zeros(env.n_gts, dtype=np.float64)
        for sample in env.last_radio_trace:
            sched = sample["sched"]
            assert (sched.sum(axis=1) <= 1).all()
            assert (sched.sum(axis=(0, 2)) <= 1).all()
            assert (sched.sum(axis=(1, 2)) <= env.n_rbs).all()
            assert not sched[~sample["coverage"]].any()
            assert np.isfinite(sample["rate_per_gt"]).all()
            delivered += sample["rate_per_gt"].astype(np.float64) * sample["duration_s"]
            step_data += sample["rate_per_gt"].astype(np.float64) * sample["duration_s"]
            traces.append(sample)
        rows.append(dict(
            t=t, time_s=t * env.dt, TotalThroughput_Gb=env.total_throughput,
            ServiceThroughput_Gb=delivered.sum()/1e3,
            FairIdx=env.fair_idx, AvgGlobalUtility=env.avg_global_util,
            reward_mean=reward.mean(),
            collision_events=env.n_colls, step_served_gt=int((step_data > 0).sum()),
            mean_distance=float(np.mean(env.d_u2g)),
            covered_gt=int(np.count_nonzero((env.d_u2g <= env.r_cov).any(axis=0)))))
    arrays = {key: np.stack([s[key] for s in traces]) for key in traces[0]}
    arrays["actions"] = np.asarray(actions)
    arrays["trajectory"] = np.stack([s["pos_ubs"] for s in traces])
    arrays["initial_gt"] = positions["gt"]
    valid = arrays["uav_step"] > 0
    alloc = arrays["sched"][valid]
    rb_changes = int(np.any(alloc[1:] != alloc[:-1], axis=(1,2,3)).sum()) if len(alloc) > 1 else 0
    covered_fraction = float(np.mean(arrays["coverage"].any(axis=1)))
    metrics = dict(
        label=label, trajectory_mode=trajectory_mode, scheduler_mode=scheduler_mode,
        seed=seed, **rows[-1], service_decisions=int(valid.sum()),
        rb_change_count=rb_changes, covered_fraction=covered_fraction,
        zero_service=bool(delivered.sum() == 0), collision_rate=float(env.n_colls/env.t),
        max_weight_span=float(np.ptp(arrays["weights"][valid], axis=0).max())
            if scheduler_mode == "edgeric_pf" else 0.0,
        **env.get_scheduling_stats())
    return pd.DataFrame(rows), arrays, metrics

def plot_seed(out, scenario, tables, traces):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for label, table in tables.items():
        ax[0,0].plot(table.time_s, table.ServiceThroughput_Gb, label=label)
        ax[0,1].plot(table.time_s, table.FairIdx, label=label)
        ax[1,0].plot(table.time_s, table.AvgGlobalUtility, label=label)
    for label in ("C", "D"):
        tr = traces[label]
        ax[1,1].plot(np.arange(len(tr["pos_ubs"])) * 1.0, tr["pos_ubs"][:,0,0], label=label+" UAV0 x")
    for a, title in zip(ax.flat, ["Service throughput (Gb)", "Jain fairness",
                                  "Average utility", "Interpolated UAV x (m)"]):
        a.set_title(title); a.grid(alpha=.25); a.legend()
    fig.suptitle(f"{scenario} | seed {0} | A/B/C/D")
    fig.savefig(out/(scenario+".png"), dpi=160); fig.savefig(out/(scenario+".pdf"))
    plt.close(fig)

def ci95(values):
    values = np.asarray(values, dtype=float)
    return 1.96 * values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("../results/stage3/fixed_trajectory"))
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    p.add_argument("--dt-sched", type=float, default=1.0)
    args=p.parse_args()
    if args.steps < 1 or args.steps > 50: p.error("steps must be in [1,50]")
    if args.dt_sched <= 0 or not np.isclose(40 / args.dt_sched, round(40 / args.dt_sched)):
        p.error("dt-sched must divide 40 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    summary, paired, timing = [], [], {}
    for scenario in ("contention_hover", "contention_motion"):
        for seed in args.seeds:
            positions, actions = scenario_inputs(seed, scenario, args.steps)
            tables, traces = {}, {}
            for label in CONFIGS:
                table, trace, metrics = run_case(seed, label, positions, actions, args.dt_sched)
                tables[label], traces[label] = table, trace
                summary.append(dict(scenario=scenario, **metrics))
                stem=f"{scenario}_seed{seed}_{label}"
                table.to_csv(args.output/(stem+".csv"), index=False)
                np.savez_compressed(args.output/(stem+".npz"), **trace)
                timing.setdefault(label, []).extend(trace["compute_s"].tolist())
            for label in ("B","C","D"):
                np.testing.assert_array_equal(traces["A"]["actions"], traces[label]["actions"])
                np.testing.assert_array_equal(traces["A"]["initial_gt"], traces[label]["initial_gt"])
            for left,right in (("A","B"),("C","D")):
                l,r=tables[left].iloc[-1],tables[right].iloc[-1]
                paired.append(dict(scenario=scenario, seed=seed, pair=left+"-"+right,
                    throughput_delta_Gb=r.ServiceThroughput_Gb-l.ServiceThroughput_Gb,
                    fairness_delta=r.FairIdx-l.FairIdx,
                    utility_delta=r.AvgGlobalUtility-l.AvgGlobalUtility,
                    rb_delta=0))
            if seed == args.seeds[0]: plot_seed(args.output, scenario, tables, traces)
    frame=pd.DataFrame(summary); frame.to_csv(args.output/"summary.csv", index=False)
    metrics=["ServiceThroughput_Gb","FairIdx","AvgGlobalUtility","covered_fraction","collision_rate"]
    aggregate=[]
    for (scenario,label), group in frame.groupby(["scenario","label"]):
        row=dict(scenario=scenario,label=label,n=len(group))
        for metric in metrics:
            row[metric+"_mean"]=group[metric].mean()
            row[metric+"_std"]=group[metric].std(ddof=1)
            row[metric+"_ci95"]=ci95(group[metric].values)
        aggregate.append(row)
    pd.DataFrame(aggregate).to_csv(args.output/"summary_stats.csv", index=False)
    pairf=pd.DataFrame(paired); pairf.to_csv(args.output/"paired_comparison.csv", index=False)
    pair_stats=[]
    for (scenario,pair), group in pairf.groupby(["scenario","pair"]):
        row=dict(scenario=scenario,pair=pair,n=len(group))
        for metric in ("throughput_delta_Gb","fairness_delta","utility_delta"):
            row[metric+"_mean"]=group[metric].mean()
            row[metric+"_std"]=group[metric].std(ddof=1)
            row[metric+"_ci95"]=ci95(group[metric].values)
        pair_stats.append(row)
    pd.DataFrame(pair_stats).to_csv(args.output/"paired_stats.csv", index=False)
    timing_rows=[]
    for label, vals in timing.items():
        vals=np.asarray(vals); timing_rows.append(dict(
            label=label,samples=len(vals),period_s=args.dt_sched,
            mean_ms=vals.mean()*1e3,p95_ms=np.percentile(vals,95)*1e3,
            p99_ms=np.percentile(vals,99)*1e3,max_ms=vals.max()*1e3,
            over_period_count=int((vals > args.dt_sched).sum())))
    pd.DataFrame(timing_rows).to_csv(args.output/"timing_pooled.csv", index=False)
    metadata=dict(configs=CONFIGS,seeds=args.seeds,steps=args.steps,dt_uav=40,
        dt_sched=args.dt_sched,trajectory_sampling="substep end positions",
        traffic="saturated original channel model; geometry-only dynamics",
        metrics="native env period metrics; service throughput excludes reset",
        git_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    (args.output/"experiment_config.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(pd.DataFrame(aggregate).to_string(index=False))
    print(pd.DataFrame(pair_stats).to_string(index=False))
    print(pd.DataFrame(timing_rows).to_string(index=False))

if __name__=="__main__": main()
