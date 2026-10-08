# UAV-EdgeRIC 第二阶段：双时间尺度调度仿真

## 目标与保护措施

主项目为 uav_bs_ctrl，工程仍位于 /workspace/uav_edgeric_stage1。复用 Liuyifei-env 中的 uav_edgeric_py39 和 uav_edgeric_gpu，未安装依赖、修改环境配置或 Docker/驱动，也未修改 EdgeRIC-on-5G。

开始前原工作区有阶段一和 GPU 验证的未提交内容。先执行原有 31 项测试并通过，再创建源文件备份 results/stage2/stage1_gpu_source_backup.tar.gz，并在 feat/edgeric-pf-stage1 保存本地提交：

    4eff8b9ad05873c873cd245b4058da9fc5d86b28

第二阶段独立分支为 feat/edgeric-dual-timescale；未强制覆盖分支、丢弃改动或推送 GitHub。原始上游 commit 仍为 5ef26f2c0c324a5888eca999beff3d42b6137ace。

对阶段一、GPU 验证结果及四个既有配置/说明文件共 102 个文件做了 SHA-256 前后比较，完全一致；两套环境的 pip freeze 与原交付清单一致。校验报告见 results/stage2/preservation_report.json。

## 配置与调用

    env = MultiUbsCoverageEnv(
        map_id="4ubs",
        scheduler_mode="edgeric_pf",  # 或 "original"
        fast_scheduling=True,        # 默认 False
        dt_sched=1.0,                # 秒
        record=False,
        record_substeps=False,       # 实验追踪时设 True
    )

self.dt 仍是地图的 UAV 决策周期；新增 dt_uav 作为别名。4ubs 的 dt=dt_uav=40s、n_actions=9、UAV 位移集合、GNN/MARL 模型及训练接口不变。dt_sched 必须为有限正数；快速模式要求 dt_uav/dt_sched 是大于等于 1 的整数，否则抛出明确 ValueError。支持 0.5、1、2、5、10、40 等整除周期，第一版实验采用 1 秒。

原训练入口可通过 env_kwargs 增加 fast_scheduling=True、dt_sched=1.0；不需要修改 algos/madrqn/run.py。每个 UAV step 仍仅产生一个 transition 和一次 reward，快速子步不进入 MARL replay 作为额外轨迹动作。

### 调用流程

普通模式：
reset/step -> _transmit_data -> _schedule_once -> _update_geometry -> _allocate_resources -> 阶段一历史/吞吐量/公平性/效用公式 -> 原观测/reward。

快速模式：
reset -> 初始化位置/几何/零速率与空 sched，不执行资源调度，也不累计服务时间或碰撞。
step -> self.t 加一 -> 原 UAV 动作一次 -> 40 个无线子步 -> 周期平均反馈 -> 原 _get_reward -> 一次 obs/state/reward/done/info。

每个无线子步：
_prepare_radio_substep -> _schedule_once（更新几何、信道、PF 优先级、原 RB 分配及 SINR/速率）-> 用 dt_sched 累计数据和历史 -> 下一子步。

_prepare_radio_substep 是未来位置插值的扩展点，本版为空操作，采用动作后固定位置的版本 A。尚未实现版本 B；未来插值必须同时定义碰撞和反馈统计，不能只插入位置更新后宣称语义等价。

## 无线资源分配保持不变

edgeric/scheduler.py 未修改。继续用当前无干扰可达速率除以上一子步完成后的平均速率加阶段一 epsilon，按最近覆盖 UAV 的局部权重聚合成全局 GT 顺序。Original 快速模式则在每次子步后按已更新历史平均速率升序生成下一优先级。

原三维 sched、每 GT 最多一个 RB、每 UAV/RB 最多一个 GT、n_rbs 上限、覆盖判断、按距离选 UAV、满载回退、按干扰选空闲 RB、SINR 和信道模型全部复用。删除了 _transmit_data 内原来已注释的 V1 草稿，仅抽取现有 V2 逻辑，并未新增另一套分配算法。

## 时间、历史、观测与奖励的定义

### 普通模式：保留阶段一精确行为

包括 reset 的一次随机初始优先级传输，reset 计入原来的完整 dt 服务；历史仍为 (avg*t+rate)/(t+1)，Original 同值排序和 PF epsilon 均不变。self.t 仍只记录 UAV step。新增计时/诊断字段不进入原 info，也不改变随机数消耗。

20 个 step 后普通模式共 21 次调度，包括 reset，原累计数据对应 840 秒服务而轨迹时钟为 800 秒。这是兼容性保留，不是第二阶段快速模式采用的初始化定义。

### 快速模式：按真实服务时长累计

reset 的 total_throughput、service_time_s、scheduling_count、avg_rate_per_gt、rate_per_gt 均为零；不做“预览传输”。初始几何可见，速率特征为零。与旧模式的初始化观测不同，这是显式的模型选择。

对 GT-m，第 k 个子步的原无线模型输出为 r[m,k]，单位 Mbps：

    delivered_mbit_per_gt[m] += r[m,k] * dt_sched
    service_time_s = scheduling_count * dt_sched
    avg_rate_per_gt[m] = delivered_mbit_per_gt[m] / service_time_s
    total_throughput = sum(delivered_mbit_per_gt) / 1000  # Gb

使用 float64 累计数据；子步实际无线速率仍来自原分配器的 float32 结果。历史没有使用 self.t，也没有引入 EMA、伪样本或未来信息。每次 PF 决策读取前一已完成子步的历史；首个子步历史为零。

每个 UAV 周期结束：

    rate_per_gt = sum_k(r_gt[k] * dt_sched) / dt_uav
    rate_per_ubs = sum_k(r_ubs[k] * dt_sched) / dt_uav
    fair_idx = 原 Jain(avg_rate_per_gt)
    global_util = fair_idx * mean(rate_per_gt)
    avg_global_util = (old_avg_global_util*(t-1) + global_util) / t

因此 rate_per_gt 是当前 UAV 周期平均速率；avg_rate_per_gt 是 episode 至今的服务时间平均。fair_idx 使用长期历史，global_util 使用这个公平性乘当前周期平均速率，不是对子步效用简单求均值。avg_global_util 是已完成 UAV 周期效用的平均，不含 reset。

_get_reward 的数学结构未修改：fair_service 时基于 global_util，否则基于 mean(rate_per_gt)，按原 max_rate/scale 缩放；空闲 UAV 用周期平均 rate_per_ubs 判定，再应用原碰撞惩罚。get_obs/get_state 使用同一组周期平均速率和长期历史，所以观测与 reward 的通信状态一致。

sched、last_prior_gts 和 PF 权重表示最后一个实际子步的分配/优先级；instant_rate_per_gt、instant_rate_per_ubs 保存最后子步瞬时速率。不能把最后 sched 与周期平均用户速率相乘来重建 period rate_per_ubs，实际关联在不同子步可能变化。

### 碰撞与终止

快速 reset 不累计碰撞。版本 A 每个周期位置固定，仅第一个子步按原 mask_collision.sum()/2 累计一次；没有乘 40。该历史约定不是一般情况下的“所有碰撞对数量”，未擅自修正原模型。惩罚仍每个 UAV step 应用一次。episode_limit、done、BadMask 与原 50 个 UAV step 完全一致。

原 Jain 对全零速率裁剪到 1e-6，可能返回接近 1。无覆盖测试检查有限值和零吞吐量，但这种公平性不代表有效服务。

## 调度计时与诊断

get_scheduling_stats() 返回：
- dt_uav_s、实际 dt_sched_s；
- scheduling_count（真正调用分配器的次数）；
- initialization_scheduling_count（普通 reset 为 1，快速 reset 为 0）；
- service_time_s；
- mean/p95/p99/max_compute_s 与 over_period_count。

scheduling_durations_s 保存每次决策的 perf_counter 耗时，reset 后清零。测量范围含 CPU 几何、信道、权重/优先级、RB 分配和 SINR/速率计算；不含历史积分、周期聚合、追踪复制、Python step 其他部分和 MARL GPU 运算。未加入 sleep，也没有用异步 GPU 提交时间假装运算耗时。

record_substeps=True 时 last_radio_trace 保存最近一个 UAV 周期的完整子步数据，下一次 step 即替换，避免完整 episode 的大分配矩阵一直留在环境内。含 duration、service_time、rate、history_before/after、sched、priority、coverage、位置、compute_s；PF 另含 weights。正式训练默认关闭该追踪。

模拟周期为 1s 不意味着物理系统硬实时保证；单次测量也不构成最坏情况执行时间证明。

## 测试

保留原有两个测试文件不动，新增 tests/test_dual_timescale.py。

- 改动前基线：31 passed。
- 改动后 CPU：75 passed（原 31 + 新增 44），18 条既有依赖弃用警告。
- GPU 环境：75 passed，同样 18 条警告。
- 原有 Original 对上游 commit 的回归继续通过。
- 新增 Original/PF 非快速回归：对上述冻结阶段一提交，默认/显式 False × 两模式 × 三种子 × 51 帧，612 对帧逐元素相等。包括 sched、用户/UAV 速率、历史、吞吐量、公平性、效用、priorities、reward、全部返回接口，以及 PF 内部诊断状态。
- 快速 reset 为零时长、每步恰好 40 调度、t 只增一、40s/1s 积分、历史因果时序、覆盖/RB 限制、NaN/Inf、位置固定、反馈/reward 一致均通过。
- 独立恒定单链路解析测试验证 0.5/1/2/5/10/40s 下传输量相同，排除重复乘 dt_uav 的错误。
- 碰撞统计在 1s 与 40s 下每个 UAV step 一次；模拟 perf_counter 验证 P95/P99 和超过周期计数。
- 24 份实验 NPZ 的维度、有限值、RB 约束和图文件完整性校验通过。

## GPU 真实训练验证

复用 GPU 环境（PyTorch 2.1.2、CUDA 12.1、DGL 1.1.3+cu121）与 scripts/verify_gpu.py，新增可选 --fast-scheduling、--dt-sched 参数，默认仍为普通模式。未修改任何网络架构或学习器。

普通模式 6 组 + 快速模式 6 组：Original/PF × 无通信/TarMAC/DISC。每组 2 个完整 4ubs episode、100 次 UAV 环境交互、2 次原 learner.update()。

共 1,200 次 UAV 交互、24 次真实 replay/BPTT/backward/AdamW 参数更新。快速模式部分执行 24,000 次无线决策；普通模式含 reset 为 612 次。两种模式均验证 CUDA 图编码器、策略输出、有限非零梯度、编码器/循环或通信层/输出层参数变化，以及 CUDA AdamW moments，无设备不一致或 OOM。PyTorch peak allocated 约 44–55 MiB，不代表整个进程总显存。

网络/lr 沿用已有 GPU smoke 的 run_exp3 配置，batch=2、replay=8，仅缩小验证规模；所有详细 loss、参数变化和每 episode 调度统计保存在 gpu_legacy/ 与 gpu_fast/。

## 四组配对实验

A=Original 普通，B=PF 普通，C=Original 快速，D=PF 快速。C/D 均为 dt_sched=1s，且使用相同无线参数、GT 分布、初始 UAV、固定动作序列。未修改原饱和业务/信道假设，未添加时变流量。

实验使用 seeds=0,10,20，每次 20 个 UAV step（800 秒），两种人工竞争场景：
- contention_hover：两对 UAV 重叠覆盖、49 个附近用户与 1 个远端用户，全部悬停。
- contention_motion：相同布局，每 10 步执行原 +200m 动作，下一步返回，其余悬停。位置在每个周期内固定。

共 24 个 episode，全部有有效服务和 RB 竞争；没有把零吞吐量随机轨迹当作有效证据。代码断言 A/B/C/D 的轨迹、动作和 GT 分布逐元素一致。

### 原环境指标（三种子均值）

| 场景 | 组 | TotalThroughput Gb | FairIdx | AvgGlobalUtility | 实际调度次数/episode |
|---|---|---:|---:|---:|---:|
| contention_hover | A | 6.616279 | 0.895264 | 0.109088 | 21 |
| contention_hover | B | 6.684345 | 0.892234 | 0.110184 | 21 |
| contention_hover | C | 5.714824 | 0.930179 | 0.132613 | 800 |
| contention_hover | D | 5.760498 | 0.929169 | 0.133544 | 800 |
| contention_motion | A | 6.353083 | 0.876189 | 0.099735 | 21 |
| contention_motion | B | 6.513355 | 0.856226 | 0.094646 | 21 |
| contention_motion | C | 5.351813 | 0.919077 | 0.102816 | 800 |
| contention_motion | D | 5.385702 | 0.919234 | 0.103533 | 800 |

A/B 的 TotalThroughput 包含 reset 40 秒数据，C/D 不包含；FairIdx/AvgGlobalUtility 的历史窗口也有相应差异。因此这张原值表不可直接用 A 与 D 宣称收益。

### 统一 post-reset 800 秒窗口（三种子均值）

评估器另计算 ServiceThroughput_Gb（排除旧 reset 数据）、WindowFairIdx（仅本窗口累计用户数据）和 WindowAvgUtility（本窗口逐 UAV 周期公平性 × 周期平均速率的均值），不回写环境或修改 reward：

| 场景 | 组 | 800s 吞吐量 Gb | WindowFairIdx | WindowAvgUtility |
|---|---|---:|---:|---:|
| contention_hover | A | 5.862594 | 0.864642 | 0.094140 |
| contention_hover | B | 5.839613 | 0.854381 | 0.095216 |
| contention_hover | C | 5.714824 | 0.930179 | 0.132613 |
| contention_hover | D | 5.760498 | 0.929169 | 0.133544 |
| contention_motion | A | 5.599398 | 0.851421 | 0.084554 |
| contention_motion | B | 5.668624 | 0.838219 | 0.088052 |
| contention_motion | C | 5.351813 | 0.919077 | 0.102816 |
| contention_motion | D | 5.385702 | 0.919234 | 0.103533 |

时间窗口对齐仍不能消除 A/B 的初始化历史对后续调度的影响；归因 EdgeRIC-PF 的主比较必须用 C 对 D。C/D 的初始化、历史更新、周期及总服务时长完全相同。

本次 C->D：
- 悬停吞吐量约 +0.7992%，综合效用略升，但 Jain 公平性略降。
- 移动吞吐量约 +0.6332%，公平性和综合效用小幅上升。
- 快速模式的同算法吞吐量低于普通模式在相同 post-reset 窗口的数据量，不能宣称快速调度自动改善吞吐量。
- A->B 悬停平均 post-reset 吞吐量下降，且多个种子存在性能下降；保留真实结果。
- C/D 每场景 2,400 个对应子步的 RB 矩阵全部不同；悬停 2,400 个服务集合不同，移动 2,160 个服务集合不同。权重和 RB 随历史更新发生实际变化。
- 三个种子的短实验不支持统计显著性、收敛性或普适性能结论。

### 实际计算耗时（本次实验全部样本汇总）

| 组 | 样本数 | 仿真周期 s | 均值 ms | P95 ms | P99 ms | 最大 ms | 超周期次数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 126 | 40 | 2.940 | 3.029 | 3.143 | 4.513 | 0 |
| B | 126 | 40 | 3.033 | 3.138 | 3.255 | 3.414 | 0 |
| C | 4800 | 1 | 2.927 | 3.023 | 3.097 | 43.135 | 0 |
| D | 4800 | 1 | 3.060 | 3.161 | 3.252 | 4.746 | 0 |

A/B 每个 episode 实际调度 21 次（20 服务步 + reset），C/D 每个 episode 调度 800 次；每次 step C/D 恰好 40 次。耗时为 CPU 测量，受机器负载影响；本次没有决策超过其配置周期，仍不代表硬实时保证。

## 文件和复现命令

新增：
- tests/test_dual_timescale.py：冻结阶段一回归及快速模式测试。
- scripts/compare_stage2.py：四组配对实验、统一窗口评估、完整子步输出及图。
- 工程根目录 README_STAGE2.md、results/stage2/。

修改：
- envs/mubs_cov/mubs_cov.py：配置、共享分配方法、独立服务时钟、聚合和计时。
- scripts/verify_gpu.py：复用原 GPU smoke，增加快速模式选项和无线计数断言。

未修改：
edgeric/scheduler.py、maps.py、algos/、run_exp1/2/3、原测试文件、CPU/GPU 环境配置、阶段一和 GPU 原结果。

容器内复现：

    cd /workspace/uav_edgeric_stage1/uav_bs_ctrl
    export DGLBACKEND=pytorch MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
    /opt/conda/bin/conda run --no-capture-output -n uav_edgeric_py39 python -B -m pytest -q -p no:cacheprovider tests
    /opt/conda/bin/conda run --no-capture-output -n uav_edgeric_gpu python -B -m pytest -q -p no:cacheprovider tests
    /opt/conda/bin/conda run --no-capture-output -n uav_edgeric_gpu python -B scripts/verify_gpu.py --output ../results/stage2/gpu_legacy
    /opt/conda/bin/conda run --no-capture-output -n uav_edgeric_gpu python -B scripts/verify_gpu.py --fast-scheduling --dt-sched 1 --output ../results/stage2/gpu_fast
    /opt/conda/bin/conda run --no-capture-output -n uav_edgeric_py39 python -B scripts/compare_stage2.py --steps 20 --seeds 0 10 20 --dt-sched 1 --output ../results/stage2/experiments

保留已有实验时请给 --output 指定新的 stage2 子目录。不要把第二阶段 GPU 验证输出写回 results/gpu_validation。

Windows 非交互调用示例：

    ssh -o BatchMode=yes -o ConnectTimeout=10 dell@100.64.193.15 "docker exec -w /workspace/uav_edgeric_stage1/uav_bs_ctrl -e DGLBACKEND=pytorch -e MPLBACKEND=Agg -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 Liuyifei-env /opt/conda/bin/conda run --no-capture-output -n uav_edgeric_py39 python -B -m pytest -q -p no:cacheprovider tests"

results/stage2/experiments 包含：
- summary.csv / summary_mean.csv：24 组原指标与统一窗口指标。
- paired_comparison.csv：A/B 和 C/D 的 RB/用户集合及吞吐量差异。
- timing_pooled.csv：逐样本汇总计算耗时。
- 每组 CSV（周期反馈）与 NPZ（每次调度 sched/rate/history/weights/priority/coverage/time/compute_s，加完整轨迹及动作）。
- 两种场景的 PNG/PDF 对比图与 experiment_config.json。
- 初次报告汇总代码的重复字段错误已修复，首次错误日志单独保存；最终 24 组实验已成功完成。

## 局限与后续

这是仿真中的独立快速无线调度，不是部署的 μApp、RAN 或物理 5G 毫秒级硬实时系统。版本 A 的 UAV 运动是瞬时动作后固定位置，没有连续飞行插值。固定信道和饱和需求下，子步变化仅由服务历史、排序和干扰分配反馈产生，快调度可牺牲吞吐量换取不同公平性表现。

普通/快速初始化定义不同是精确保留旧模式与消除新模式虚假服务时间的折中；对跨时间尺度的因果归因需要额外控制初始化历史。未引入多 RB/GT、MCS、队列、双时间尺度学习器或长期训练。后续可独立评估轨迹插值、历史窗口与 PF epsilon 敏感性，并在同一周期下持续比较 Original/PF。
