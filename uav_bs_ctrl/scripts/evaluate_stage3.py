"""Cross-evaluate stage-3 pilot checkpoints under both schedulers."""
import argparse, copy, json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from algos.madrqn.run import load_and_run_policy
from envs.mubs_cov.mubs_cov import MultiUbsCoverageEnv
from scripts.compare_stage1 import scenario_inputs

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    args=p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    common=dict(device="cuda",cuda_index=0,cuda_deterministic=False,
        o="gnn",c=None,hidden_size=64,n_layers=1,msg_size=32,lr=5e-4,
        polyak=.995,gamma=.99,batch_size=4,replay_size=500,max_seq_len=5,
        epochs=1,steps_per_epoch=60,update_after=20,num_test_episodes=1,
        save_freq=1,decay_steps=100,norm_r=True,anneal_lr=False,double_q=True,
        dueling=False,mixer=False,share_reward=False)
    rows=[]
    for trained_mode in ("original","edgeric_pf"):
        for seed in args.seeds:
            ckpt=args.root/f"stage3_{trained_mode}_linear_interpolation"/f"stage3_{trained_mode}_linear_interpolation_s{seed}"/"checkpoint_epoch1.pt"
            if not ckpt.exists(): raise FileNotFoundError(ckpt)
            eval_seed=10000+seed
            positions, _ = scenario_inputs(eval_seed, "contention_motion", 20)
            for eval_mode in ("original","edgeric_pf"):
                env_kwargs=dict(map_id="4ubs",fair_service=True,avoid_collision=True,
                    scheduler_mode=eval_mode,fast_scheduling=True,dt_sched=1.0,
                    trajectory_mode="linear_interpolation")
                def env_factory(**kwargs):
                    env=MultiUbsCoverageEnv(**kwargs)
                    env.map=copy.copy(env.map)
                    env.map.set_positions=lambda: {k:v.copy() for k,v in positions.items()}
                    env.episode_limit=20
                    return env
                out=args.output/f"{trained_mode}_seed{seed}_eval_{eval_mode}"
                out.mkdir(parents=True,exist_ok=True)
                df=load_and_run_policy(str(ckpt),env_factory,env_kwargs,
                    eval_seed,common,1,str(out))
                df.to_csv(out/"metrics.csv",index=False)
                row=df.iloc[0].to_dict()
                rows.append(dict(trained_scheduler=trained_mode,train_seed=seed,
                    eval_scheduler=eval_mode,checkpoint=str(ckpt),
                    eval_scene="contention_motion",eval_steps=20,**row))
    frame=pd.DataFrame(rows)
    frame.to_csv(args.output/"cross_evaluation.csv",index=False)
    meta=dict(training_seeds=args.seeds,eval_seed_offset=10000,
        trajectory_mode="linear_interpolation",dt_sched=1.0,
        git_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    (args.output/"evaluation_config.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(frame.to_string(index=False))
if __name__=="__main__": main()
