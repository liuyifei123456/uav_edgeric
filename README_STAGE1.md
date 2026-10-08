# UAV-EdgeRIC 第一阶段：EdgeRIC-inspired lightweight scheduling

## 1. 目标与完成范围

在 uav_bs_ctrl 中保留原 RB 分配器，新增利用当前信道与历史速率生成优先级的 EdgeRIC 风格轻量化动态权重调度。UAV 移动、GNN/MARL 结构和 4ubs 的 40 秒轨迹步长不变。已实际完成远程克隆、独立环境安装、代码集成、单元/回归测试及 CPU 配对实验。

本实现是“基于 UAV 局部无线状态和全局优先级聚合的轻量 EdgeRIC 风格调度仿真”，并非分布式物理 UAV RIC，也没有安装完整 EdgeRIC、接入真实 gNB 或实现毫秒级实时控制。

## 2. 远程环境、仓库与目录

SSH：dell@100.64.193.15；Docker：Liuyifei-env（原本已运行，未重启）。
全部工程读取、编辑、安装和测试均在该容器中完成；Windows 仅用于发起 SSH。没有创建 /workspace/uav_bs_ctrl。

| 仓库 | 来源 | 基础 commit | 分支 |
|---|---|---|---|
| 主项目 | https://github.com/zhangxiaochen95/uav_bs_ctrl | 5ef26f2c0c324a5888eca999beff3d42b6137ace | feat/edgeric-pf-stage1 |
| 参考项目 | https://github.com/ucsdwcsng/EdgeRIC-on-5G | 40a643ba16df0e2dda86f3e584438b3d6bd8152c | srsran |

主项目改动留在开发分支工作区，未提交、未推送；main 保持原 commit。参考仓库保持干净，仅阅读源码，未运行其安装、编译或服务。

目录：

    /workspace/uav_edgeric_stage1/
      uav_bs_ctrl/
        edgeric/{__init__.py,scheduler.py}
        envs/mubs_cov/mubs_cov.py
        tests/{test_edgeric_scheduler.py,test_edgeric_integration.py}
        scripts/{verify_environment.py,compare_stage1.py}
        .gitignore
      EdgeRIC-on-5G/
      results/stage1/
      environment.yml
      README_STAGE1.md

## 3. Conda 环境与兼容性处理

环境名 uav_edgeric_py39，路径 /opt/conda/envs/uav_edgeric_py39。检查发现该名字原先不存在，也无正在运行的 Conda/pip 安装进程；此前留下的是 /workspace/.envs/uav-bs-phase1 的 Python 基础环境。保留原环境，用 conda --clone 建立独立目标环境后补全依赖，未安装到 base。

| 依赖 | 验证版本 |
|---|---|
| Python | 3.9.23 |
| PyTorch | 1.12.1+cpu |
| DGL | 0.9.0 |
| Gym | 0.21.0 |
| NumPy / SciPy | 1.22.4 / 1.9.0 |
| Pandas / NetworkX | 1.4.3 / 2.8.5 |
| Matplotlib / seaborn | 3.5.3 / 0.11.2 |
| mpi4py / MPICH | 3.1.3 / 4.3.2 |
| cloudpickle / joblib | 2.1.0 / 1.1.0 |
| psutil / tqdm | 5.9.1 / 4.64.0 |
| pip / setuptools / wheel | 22.2.2 / 65.0.1 / 0.37.1 |
| pytest | 7.4.4 |

原 requirements.txt 明确是 osx-arm64 导出，未用来安装。Gym 0.21.0 使用上述历史构建工具成功构建 wheel；DGL 使用 0.9.0 历史 wheel，PyTorch 使用官方 CPU wheel。MPI 使用 conda-forge 的 Linux mpi4py 二进制包与 MPICH。默认 Anaconda channels 曾返回 ToS 未接受错误，随后明确指定 --override-channels -c conda-forge，未接受条款或修改全局 Conda 配置。

CPU 张量反向计算、DGL 消息聚合、MPI 导入及现有 MARL/GNN 推理链路正常。当前 CPU PyTorch 的 cuda.is_available() 为 false、device_count 为 0；容器 nvidia-smi 返回 “Failed to initialize NVML: Unknown Error”。未尝试改驱动、重建容器或 GPU 训练。

environment.yml 是最终环境的 Linux x86_64 精确版本/构建配置，移除了绝对 prefix，设置 conda-forge/nodefaults，并加入 CPU PyTorch 和 DGL wheel 源。results/stage1 中还保存完整原始导出 environment-resolved.yml、conda-explicit-linux-64.txt 和 pip-freeze.txt。

仅在目标环境尚不存在的新环境中重建：

    cd /workspace/uav_edgeric_stage1
    /opt/conda/bin/conda env create -f environment.yml

不要对当前已存在环境重复执行创建，也不要直接用 pip-freeze.txt 替代 Conda 安装 MPI。完整重建流程没有在第二个全新环境重复执行；交付配置来自本次实际通过验证的安装环境。

## 4. 运行、测试与启用方式

以下命令在容器内执行；独立命令均指定环境，不依赖激活状态：

    cd /workspace/uav_edgeric_stage1/uav_bs_ctrl
    export DGLBACKEND=pytorch MPLBACKEND=Agg OMP_NUM_THREADS=1
    /opt/conda/bin/conda run -n uav_edgeric_py39 python --version
    /opt/conda/bin/conda run --no-capture-output -n uav_edgeric_py39 python scripts/verify_environment.py
    /opt/conda/bin/conda run --no-capture-output -n uav_edgeric_py39 python -m pytest -q tests
    /opt/conda/bin/conda run --no-capture-output -n uav_edgeric_py39 python scripts/compare_stage1.py --seeds 0 10 20 --steps 50 --output ../results/stage1
    /opt/conda/envs/uav_edgeric_py39/bin/python -m pip check

从 Windows 非交互运行测试的完整例子：

    ssh -o BatchMode=yes -o ConnectTimeout=10 dell@100.64.193.15 "docker exec -w /workspace/uav_edgeric_stage1/uav_bs_ctrl -e DGLBACKEND=pytorch -e MPLBACKEND=Agg -e OMP_NUM_THREADS=1 Liuyifei-env /opt/conda/bin/conda run --no-capture-output -n uav_edgeric_py39 python -m pytest -q tests"

启用方法：

    from envs.mubs_cov.mubs_cov import MultiUbsCoverageEnv
    original = MultiUbsCoverageEnv("4ubs", record=False)
    pf = MultiUbsCoverageEnv("4ubs", record=False, scheduler_mode="edgeric_pf")

未来训练可通过原有 env_kwargs 传入 scheduler_mode，无需改训练框架。run_exp1.py、run_exp2.py、run_exp3.py 均未修改，默认行为保持 Original。比较脚本重复运行会更新指定输出目录；保留旧实验时请指定新的 --output。

## 5. 源码分析与算法

### Original

reset 先生成随机 GT permutation 并立即传输一次；其后使用 np.argsort(avg_rate_per_gt) 从低历史速率到高历史速率安排下一次调度。保留上游默认 argsort 的同值行为，未改成 stable sort。历史更新仍是：

    avg_rate_per_gt = (avg_rate_per_gt * t + rate_per_gt) / (t + 1)

### EdgeRIC 参考实现

阅读了 edgeric-v2/README.md、send_weight.py、edgeric_messenger.py 及 srsRAN-5G-ER 的 cell_scheduler.cpp、ue_cell_grid_allocator.cpp、lib/edgeric/edgeric.cpp。

参考实现由 RAN 上报 RNTI、CQI、SNR、收发字节数、上下行 buffer 与 DL TBS，messenger 经 ZMQ/Protobuf 接收状态、发送 SchedulingWeights。send_weight.py 当前示例实际是固定的 0.7/0.3 权重，不能将它称为本次 PF 公式的原始实现。

cell_scheduler.cpp:94-96 在 slot 调度前读取权重；edgeric.cpp:255 起解析 RNTI/weight 配对并归一化；ue_cell_grid_allocator.cpp:296-306 将 weight * this_tti_unused_crbs 用于 n_prbs，并施加业务需求和配置上下界。因此原版权重能影响 PRB 数量。本阶段只借鉴“状态 -> 动态权重 -> 调度控制”的思路，将权重映射到 GT 顺序，未移植通信协议或 RAN 代码。

### PF 公式、单位及时间顺序

对每个覆盖内 UAV-i / GT-m：

    Rhat[i,m] = B * log2(1 + Ptx*g[i,m]/(N0*B)) * 1e-6  [Mbps]
    local_weight[i,m] = Rhat[i,m] / (history[m] + epsilon)
    epsilon = 0.05 * max_single_RB_rate = 0.10764378062768 Mbps（本次配置）

本次 max_rate 为 2.15287561255360 Mbps。5% 正则项将此信道模型下零历史权重约束在 20 内，避免机器 epsilon 使未服务用户获得数量级异常权重；它会改变接近零历史区域的纯 PF 行为，不宣称这是最优超参数。

步骤为：

1. reset 置零历史，仍消耗原来的随机 permutation 以保持 RNG 消耗一致。
2. _transmit_data 更新当前位置、距离、覆盖关系及当前信道增益。
3. 在任何实际 RB 分配之前，用上一轮已完成的历史计算本地 PF 权重；reset 时历史为零。
4. 每个 GT 选最近可覆盖 UAV 的局部权重作为全局权重，按降序 stable sort 得到 prior_gts。
5. 原 RB 分配器执行后，按上游公式更新实际速率、历史、吞吐量、公平性及效用。
6. PF 的诊断顺序保留为刚才使用的顺序，下一轮在新位置重新计算。Original 仍在末尾保存下一轮低历史速率顺序。

这是无干扰的调度前估计，不使用当前尚未得到的 SINR 或实际速率。实现以 logaddexp 计算 log(1+SNR)，避免中间 SNR 溢出；非有限/负输入直接报错，不隐式排入优先队列。

### 多 UAV 聚合与 RB 机制

coverage[i,m] 严格复用 d_u2g <= r_cov。每架 UAV 都有局部候选速率与 PF 权重；最近可覆盖 UAV 决定全局 GT 分值，同距时取最小 UAV 索引，同全局权重时取较小 GT 索引。未覆盖 GT 权重为 0、selected_uav=-1，由原覆盖判断保证不分配 RB。

估计关联并不锁定实际关联。最近 UAV 满载后，原分配循环仍尝试后续 UAV。没有根据最终关联回算权重，这也是当前近似的局限。

核心生产文件仅 mubs_cov.py 修改 28 行新增、2 行替换，新增独立 scheduler 模块。原三维 sched[UAV,GT,RB]、每 GT 最多一个 RB、每 UAV/RB 最多一个 GT、每 UAV 的 n_rbs 限制、按距离选择 UAV、按干扰选择空闲 RB、干扰更新、SINR、Jain、reward 和历史平均公式均保留。信道模型、动作空间、观测结构、GNN/MARL 和轨迹周期不变。

## 6. 测试结果

- 修改前基础环境 smoke：4ubs reset + 5 次 step 通过（baseline_smoke.json）。
- scheduler 单元及集成/回归测试：31 passed。含 PF 公式及单调性、零历史、无覆盖、重叠覆盖、最近 UAV 满载回退、同权重确定性、有限值、非法输入、RB 约束、两模式 reset/50 step、历史时刻及当前信道使用、接口保持。
- Original 精确回归：默认与显式 original 各自比较上游固定 commit，3 seeds × 2 layouts × 2 entry modes × 51 frames = 612 对帧。固定 Python/NumPy seeds、布局及动作序列；RB 矩阵、每 GT 速率、历史、吞吐量、公平性、效用、reward、位置与全部返回值逐元素完全相等，无容差放宽。
- CPU PyTorch 反向计算与 DGL 消息聚合通过。
- GNN/MARL：两调度模式 × None/TarMAC/DISC 三通信配置，共 6 条现有 forward 路径通过。
- numpy/scipy/pandas/torch/dgl/gym/mpi4py 及 MARL 模块导入通过，pip check 通过。
- git diff --check 通过；参考仓库无修改。
- 有 18 条旧 Matplotlib/新 pyparsing 的弃用警告，不影响通过。
- 未执行长期 MARL 训练、MPI 多进程训练或 GPU 训练。GPU 检查受 NVML 不可用及本次 CPU 构建限制；CPU 第一阶段不受阻塞。

详细输出位于 tests.log、tests-final.log、tests-junit.xml、environment_validation.json/.log、pip-check.log。

## 7. 配对短实验

三个 seed：0、10、20；每次 reset + 50 个原始 40 秒轨迹步，共 18 个 episode。每组两模式严格断言初始位置、GT 分布、完整 UAV 位置轨迹及动作序列相等；相同 5 RB/UAV、B=180kHz、Ptx=0.01W、N0=1e-20W/Hz、覆盖半径 100m。

- native_random：原生 4ubs 随机地图、预先生成的随机动作，不是训练策略。
- contention_hover：人工设置两对重叠覆盖 UAV，49 个附近 GT 加 1 个未覆盖 GT；动作全部悬停，故权重变化来自服务历史。
- contention_motion：相同人工布局，每 10 步执行一次原动作 +200m、下一步 -200m，其余悬停，轨迹周期仍为 40 秒。

人工场景用于激发资源竞争，不能当作原生 4ubs 的性能基准。没有针对结果筛选种子或调参。初步均值如下（完整逐种子数据见 summary.csv）：

| 场景 | 模式 | TotalThroughput (Gb) | FairIdx | AvgGlobalUtility |
|---|---|---:|---:|---:|
| contention_hover | edgeric_pf | 15.152624 | 0.927871 | 0.121029 |
| contention_hover | original | 15.099064 | 0.928067 | 0.120774 |
| contention_motion | edgeric_pf | 14.743759 | 0.904855 | 0.109359 |
| contention_motion | original | 14.441236 | 0.912267 | 0.111558 |
| native_random | edgeric_pf | 0.000000 | 1.000000 | 0.000000 |
| native_random | original | 0.000000 | 1.000000 | 0.000000 |

TotalThroughput 的实际单位为 Gb：上游执行 sum(rate_Mbps)*dt/1000；保留它，未因旧注释写 Mb 而改公式。AvgGlobalUtility 是每步 Jain × 用户平均速率的累计平均。上游 reset 自带一次传输，所以 50 steps 的轨迹时间为 2000s，但累计包含 51 次 40s 服务样本（2040s）；两模式完全同口径。

原生随机场景两模式均无服务、零吞吐量。Jain 中将全零速率裁剪到 1e-6 后约等于 1，不代表有效公平服务。其优先序虽不同，RB 矩阵完全相同。

竞争悬停场景所有 153/153 帧的 RB 和服务用户集合均发生变化；移动场景 148/153 帧 RB 不同、138/153 帧服务集合不同。PF 最大权重约 19.993，历史驱动下动态变化清晰。悬停平均吞吐量略升，但 seed=10 的吞吐量和效用下降；移动平均吞吐量上升，而平均公平性和效用下降。不能据此声称 PF 全面改善，更不能宣称统计显著。

结果文件：

- summary.csv、summary_mean.csv：逐 episode 与三种子均值。
- allocation_comparison.csv：排序、RB、服务集合差异帧数和权重动态范围。
- experiment_config.json：无线参数、种子、动作/布局说明及统计口径。
- 每场景/seed/mode 的 CSV：逐步吞吐量、公平性、效用、速率与 reward。
- 同名 NPZ：完整 sched、rate_per_gt、avg_rate_per_gt、实际使用 priority、UAV/GT 位置与 actions；PF 另有本地和全局 weights、估计速率、调度前 history、selected_uav。
- 各场景 seed0 的 PNG/PDF：吞吐量、公平性、效用及变化最大的 4 个 GT 权重曲线。

## 8. 修改文件说明

- edgeric/__init__.py：公开调度器入口。
- edgeric/scheduler.py：独立、无网络依赖的本地 PF 与最近 UAV 聚合、数值检查、诊断状态。
- envs/mubs_cov/mubs_cov.py：模式参数与调度前调用，保留 Original 数学行为。
- tests/test_edgeric_scheduler.py：权重、边界和错误输入测试。
- tests/test_edgeric_integration.py：RB/接口/时间顺序检查，以及对不可变 Git 对象的回归。
- scripts/verify_environment.py：可重复的导入、CPU 张量/DGL 和现有 GNN 推理验证。
- scripts/compare_stage1.py：可重复配对实验、全量轨迹输出及绘图。
- .gitignore：忽略本次运行产生的 Python/pytest 缓存，不删除它们。
- 工程根目录 environment.yml、README_STAGE1.md 与 results/stage1：环境复现、交付说明和真实结果，不属于主项目 Git 根目录。

## 9. 局限与第二阶段预留

本阶段只有优先级适配；噪声受限估计忽略真实 RB 干扰，聚合不考虑最近 UAV 剩余容量。使用原累计平均而非滑动/指数历史，正则项和同权重索引偏置均待评估。没有队列、MCS 调节、多 RB/GT、真实链路或 μApp。短实验是固定开环轨迹，没有学习策略闭环比较，也没有统计显著性结论。原信道、Jain 和 reset 计数的已有特性全部保留。

第二阶段可保持 UAV 决策每 40s 更新一次，在轨迹步内增加独立、更短的资源调度仿真周期；需明确每个子步的信道位置、服务数据累计、按真实时长加权的历史、reward 聚合和观测暴露，再做新旧周期等价/收敛验证。本次没有实现子步循环或改变 dt。
