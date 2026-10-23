# UAV-EdgeRIC Formal GPU Training and System Evaluation

This record covers the bounded third-stage GNN/MARL training. The existing trajectory and radio algorithms were reused. Original and EdgeRIC-PF differ only in the wireless scheduler.

## Configuration

- Map: 4ubs. Training profile: clustered_v1. Each reset samples two GT clusters and places four UAVs near them; the native MAPS["4ubs"] definition is unchanged.
- UAV decision period: 40 s. Radio scheduling period: 1 s. fast_scheduling=True. trajectory_mode=linear_interpolation.
- GNN, hidden size 256, two layers, communication protocol None, learning rate 2.5e-4, batch size 32, replay size 5000, BPTT length 10, update_after=1000, and one learner update every 10 environment steps.
- Training seeds: 0, 1, 2. Final test seeds: 2000--2019. Each environment step contains 40 radio substeps.

The clustered_v1 profile is an independent, reproducible scene sampler used to avoid all-zero training signals. It does not alter the native 4ubs map or the radio equations. Five episode diagnostics produced non-zero service and rewards for both schedulers.

## Calibration and formal runs

GPU calibration ran 1,000 environment steps per scheduler and completed real learner updates with finite loss and saved checkpoints. The GPU is an NVIDIA RTX 5880 Ada with PyTorch 2.1.2 and CUDA 12.1.

All six formal runs completed 30,000 environment steps, 12 epochs, and 2,900 optimizer updates:

| Scheduler | Seed | Seconds | Final LossQ | Final epoch throughput | Final epoch Jain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 0 | 3081.7 | 0.182 | 4.30 | 0.763 |
| Original | 1 | 3042.0 | 0.172 | 4.48 | 0.794 |
| Original | 2 | 3082.7 | 0.177 | 4.06 | 0.748 |
| EdgeRIC-PF | 0 | 3393.1 | 0.191 | 4.39 | 0.754 |
| EdgeRIC-PF | 1 | 3249.7 | 0.172 | 4.17 | 0.782 |
| EdgeRIC-PF | 2 | 3420.9 | 0.170 | 3.69 | 0.754 |

Reward and throughput show rising segments but substantial epoch-to-epoch variation; this budget does not establish convergence. Checkpoints contain policy, target, optimizer, and learning-rate scheduler state. Replay contents and complete random states are not stored, so checkpoints were verified for loading and inference rather than claimed as lossless full-training resumes.

## Independent 2x2 evaluation

Each of the four policy/scheduler combinations was evaluated for three training seeds and 20 independent test seeds: 240 episodes total. Evaluation used epsilon=0.

| Trained policy | Evaluation scheduler | Throughput mean +/- sd | Jain mean +/- sd | Utility mean +/- sd | Collision rate | RB utilization | GT service |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | Original | 37.574 +/- 4.534 | 0.901 +/- 0.085 | 0.349 +/- 0.037 | 0.153 | 0.394 | 0.980 |
| Original | PF | 37.893 +/- 4.747 | 0.896 +/- 0.095 | 0.351 +/- 0.040 | 0.134 | 0.385 | 0.980 |
| PF | Original | 34.156 +/- 3.316 | 0.863 +/- 0.095 | 0.308 +/- 0.041 | 0.202 | 0.425 | 0.980 |
| PF | PF | 34.249 +/- 3.571 | 0.862 +/- 0.095 | 0.308 +/- 0.044 | 0.199 | 0.419 | 0.980 |

Paired differences and 95% confidence intervals are in results/stage3/formal_training/reports/paired_differences_ci95.csv. Replacing the scheduler for an Original-trained policy changed throughput by +0.319 (95% CI [-0.512, 1.150]) and Jain by -0.004. For the PF-trained policy the throughput change was +0.093 (95% CI [-0.267, 0.452]). PF-trained policies were lower than Original-trained policies by 3.418 throughput units under Original scheduling and 3.644 under PF scheduling. These are observed results, not a claim that PF is always worse.

The raw CSV includes TotalThroughput, FairIdx, AvgGlobalUtility, GT service fraction, collision rate, RB utilization, and mean GT rate. Training curves, confidence intervals, and PNG figures are under results/stage3/formal_training/reports and figures.

## Reproduction

```bash
cd /workspace/uav_edgeric_stage1/uav_bs_ctrl
/opt/conda/bin/conda run -n uav_edgeric_py39 env PYTHONPATH=. pytest tests -q
/opt/conda/bin/conda run -n uav_edgeric_gpu python scripts/formal_training.py diagnose --mode original --seed 0 --episodes 5 --output ../results/stage3/formal_training/calibration/diagnose_original.csv
/opt/conda/bin/conda run -n uav_edgeric_gpu python scripts/evaluate_formal_training.py --root ../results/stage3/formal_training --output ../results/stage3/formal_training/cross_evaluation
```

Formal training uses scripts/formal_training.py train with the fixed parameters above. Do not treat exploratory training reward as final policy performance.

## Conclusions and limits

Both schedulers completed real GNN/DGL CUDA updates and their checkpoints loaded successfully. The fixed Original-trained policy showed a small PF throughput increase whose confidence interval crosses zero. PF-trained policies were lower on the reported system metrics in this run. Three training seeds and 20 test seeds show variability but do not support a strong convergence or significance claim. More budget, more training seeds, and a pre-fixed validation set are needed to establish whether PF-specific trajectory coordination is learned.

The source changes are on feat/edgeric-formal-training. GitHub v0.2.0 and the publish workspace were not changed. Checkpoints, replay buffers, NPZ files, and temporary logs remain outside the Git commit.
