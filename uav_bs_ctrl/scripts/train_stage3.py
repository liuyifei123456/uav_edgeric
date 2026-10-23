"""Bounded stage-3 GNN/MARL GPU pilot and evaluation."""
import argparse, json, subprocess, time, sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algos.madrqn.run import train, load_and_run_policy
from utils.run_utils import setup_logger_kwargs
from envs.mubs_cov.mubs_cov import MultiUbsCoverageEnv

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scheduler-mode", choices=("original","edgeric_pf"), required=True)
    p.add_argument("--trajectory-mode", default="linear_interpolation",
                   choices=("endpoint_hold","linear_interpolation"))
    p.add_argument("--dt-sched", type=float, default=1.0)
    p.add_argument("--map-id", default="4ubs")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps-per-epoch", type=int, default=60)
    p.add_argument("--output", type=Path, required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run_name=f"stage3_{args.scheduler_mode}_{args.trajectory_mode}"
    logger_kwargs=setup_logger_kwargs(run_name, seed=args.seed,
                                      data_dir=str(args.output), datestamp=False)
    env_kwargs=dict(map_id=args.map_id, fair_service=True, avoid_collision=True,
                    scheduler_mode=args.scheduler_mode, fast_scheduling=True,
                    dt_sched=args.dt_sched, trajectory_mode=args.trajectory_mode)
    train_kwargs=dict(device=args.device, cuda_index=0, cuda_deterministic=False,
                      o="gnn", c=None, hidden_size=64, n_layers=1, msg_size=32,
                      lr=5e-4, polyak=0.995, gamma=0.99, batch_size=4,
                      replay_size=500, max_seq_len=5, epochs=args.epochs,
                      steps_per_epoch=args.steps_per_epoch, update_after=20,
                      num_test_episodes=1, save_freq=1, decay_steps=100,
                      norm_r=True, anneal_lr=False, double_q=True,
                      dueling=False, mixer=False, share_reward=False)
    started=time.time()
    train(MultiUbsCoverageEnv, env_kwargs, args.seed, train_kwargs, logger_kwargs)
    run_dir=Path(logger_kwargs["output_dir"])
    checkpoint=run_dir/f"checkpoint_epoch{args.epochs}.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"checkpoint missing: {checkpoint}")
    eval_dir=run_dir/"evaluation"
    eval_dir.mkdir(exist_ok=True)
    eval_df=load_and_run_policy(str(checkpoint), MultiUbsCoverageEnv, env_kwargs,
                                args.seed+10000, train_kwargs, 1, str(eval_dir))
    eval_df.to_csv(eval_dir/"metrics.csv", index=False)
    meta=dict(scheduler_mode=args.scheduler_mode, trajectory_mode=args.trajectory_mode,
              dt_sched=args.dt_sched, map_id=args.map_id, seed=args.seed,
              epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
              total_env_steps=args.epochs*args.steps_per_epoch,
              update_after=max(20, 4 * 5), update_every=5,
              optimizer_updates=sum(
                  1 for t in range(args.epochs * args.steps_per_epoch)
                  if t >= max(20, 4 * 5) and t % 5 == 0),
              checkpoint=str(checkpoint),
              gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
              torch_version=torch.__version__, cuda_version=torch.version.cuda,
              elapsed_s=time.time()-started,
              git_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    (run_dir/"stage3_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta, indent=2))
if __name__=="__main__": main()
