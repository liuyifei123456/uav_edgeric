# UAV-EdgeRIC GPU 环境与训练验证

## 完成情况

已在 dell@100.64.193.15 的 Liuyifei-env 容器中创建独立 Conda 环境 uav_edgeric_gpu。真实 GPU 上的 PyTorch 矩阵运算/自动求导、DGL GraphConv 反向传播以及原 GNN/MARL replay/BPTT/AdamW 更新全部通过。

主项目仍为 /workspace/uav_edgeric_stage1/uav_bs_ctrl，分支 feat/edgeric-pf-stage1，基础 commit 5ef26f2c0c324a5888eca999beff3d42b6137ace。没有重新克隆、修改 Docker/驱动/SSH、升级 CPU 环境或修改无线调度算法。参考仓库 EdgeRIC-on-5G 保持干净，未编译安装。

## GPU 与版本

- GPU：NVIDIA RTX 5880 Ada Generation，compute capability 8.9，显存 49140 MiB。
- 驱动：580.126.09。宿主机及容器 nvidia-smi 正常；设备 nvidia0/nvidiactl/nvidia-uvm/nvidia-uvm-tools 可打开。
- 开始检查时：显存使用 1015 MiB，空闲 47486 MiB（另有驱动保留显存）；磁盘余量约 315 GiB。
- nvidia-smi 的 CUDA 13.0 是驱动支持上限；容器 nvcc/Toolkit 为 12.1，PyTorch 的实际 CUDA 构建为 12.1。
- 环境路径：/opt/conda/envs/uav_edgeric_gpu。

| 软件 | 实际版本 |
|---|---|
| Python | 3.9.23 |
| PyTorch | 2.1.2，Conda build py3.9_cuda12.1_cudnn8.9.2_0 |
| pytorch-cuda | 12.1 |
| DGL | 1.1.3+cu121，官方 CUDA wheel |
| NumPy | 1.23.5 |
| Gym | 0.21.0 |
| SciPy / Pandas | 1.9.0 / 1.4.3 |
| NetworkX / Matplotlib | 2.8.5 / 3.5.3 |
| mpi4py / MPICH | 3.1.3 / 4.3.2 |
| MKL | 2023.2.0 |
| pip / setuptools / wheel | 22.2.2 / 65.0.1 / 0.37.1 |
| fsspec | 2023.12.2 |

Conda 求解器还选取 cuda-opencl=12.9.19、cuda-version=12.9 和一些允许范围内较新的辅助库。它们不表示 torch.version.cuda 变成 12.9；实际 PyTorch runtime、CUDA 图计算与训练已验证为上述 12.1 构建。完整确切版本和包来源在独立配置/explicit 清单中保留，未把这些差异隐藏成“所有 CUDA 子包均为 12.1”。

## 安装过程与兼容性处理

开始前确认目标环境不存在，已有 CPU 环境存在且没有正在进行的安装进程。按以下方式创建新环境（这些命令已经执行；不要对现有环境重复创建）：

    /opt/conda/bin/conda create -n uav_edgeric_gpu python=3.9.23 pip=22.2.2 setuptools=65.0.1 wheel=0.37.1 mpi4py=3.1.3 mpich=4.3.2 --override-channels -c conda-forge -y
    /opt/conda/bin/conda install -n uav_edgeric_gpu pytorch=2.1.2 pytorch-cuda=12.1 numpy=1.23.5 'mkl<2024.1' --override-channels -c pytorch -c nvidia -c conda-forge -y

MKL 上界是旧 PyTorch 的预防性兼容约束，避免新的 MKL ITT 符号兼容问题；本次没有发生该错误。Torch 使用首选 Conda 安装。期间尝试了同版本官方 wheel 备用下载，Conda 下载恢复后已中止备用下载，未安装备用 wheel。

DGL 使用 https://data.dgl.ai/wheels/cu121/repo.html 的官方 1.1.3+cu121 cp39 wheel。wheel 保存在 results/gpu_validation/wheels/。其余依赖参照 CPU 环境和源码安装，仅将 NumPy 改为指定的 1.23.5。Gym 使用历史构建工具，保持 0.21.0；MPI 保持 conda-forge mpi4py/MPICH 配套运行库。未安装 torchvision 或 torchaudio，也未使用 macOS ARM64 requirements.txt。

初次 pip check 发现 Conda PyTorch 包未包含 fsspec，已在 GPU 环境补装 fsspec==2023.12.2；最终 pip check 通过。无需修改任何生产代码即可兼容 PyTorch 2.1.2 / DGL 1.1.3。

## 验证方法与结果

新增独立脚本 uav_bs_ctrl/scripts/verify_gpu.py。原 verify_environment.py 明确仅做 CPU forward 检查，保持原样；GPU 脚本不允许 CPU fallback。

底层 CUDA 检查：
- torch.cuda.is_available() 为 True；张量、矩阵乘法结果和梯度都位于 cuda:0。
- 矩阵乘法与 CPU 参考结果校验通过，梯度有限且非零。
- DGL 图对象迁移到 cuda:0；GraphConv 参数、输入、输出和梯度均位于 GPU，backward 通过。

原 GNN/MARL 最小训练：
- 调度模式 original、edgeric_pf，各覆盖无通信、TarMAC、DISC。
- 使用原 GraphObservationEncoder（GATv2）、MultiAgentQLearner、ReplayBuffer、loss、backward 和 optimizer.step()。
- 每组跑 2 个完整的 4ubs episode，即 100 次 env.step，采集 2 个 50 步 replay 序列，执行 2 次 learner.update()。
- 六组共 600 次环境交互、12 次真实 AdamW 更新。
- batch_size=2、replay_size=8，仅用于减少 smoke test 规模；max_seq_len=None，仍为原 50 步 episode。
- 网络设置沿用 run_exp3：hidden_size=256、n_layers=2、msg_size=64；lr=2.5e-4、polyak=0.999、double_q=True、mixer=False、norm_r=True、anneal_lr=True。没有改原实验配置文件。
- 为确保 GT 图边非空，脚本内部使用人工覆盖初始布局，4ubs 的地图参数、动作空间、40 秒 dt 和 reward 公式不变；动作由实际网络及 epsilon=0.2 选择。
- hooks 检查 CUDA 图特征、hidden、编码器输出、策略输出及 loss；实际 learner.update 内部执行原 replay sampling/BPTT/backward/AdamW。
- 检查编码器、循环/通信层与输出层均有非零 CUDA 梯度；更新前后参数实际变化；AdamW exp_avg/exp_avg_sq 位于 CUDA。标量 step 计数器允许按 PyTorch 实现保留在 CPU。
- 没有 OOM，也没有设备不一致。以下显存为 PyTorch peak allocated，未包含全部驱动/CUDA/DGL 分配。

| 调度模式 | 通信 | 第一次 loss | 第二次 loss | 最大参数变化（两次最大值） | 峰值 allocated MiB |
|---|---|---:|---:|---:|---:|
| original | None | 0.000634324 | 0.000387930 | 0.000250727 | 44.08 |
| original | tarmac | 0.250922203 | 0.246986672 | 0.000251025 | 49.79 |
| original | disc | 6.236356258 | 5.927587032 | 0.000250906 | 55.41 |
| edgeric_pf | None | 0.000634173 | 0.000387873 | 0.000250757 | 44.08 |
| edgeric_pf | tarmac | 0.250956029 | 0.247022659 | 0.000251025 | 49.79 |
| edgeric_pf | disc | 6.236164570 | 5.927642822 | 0.000251114 | 55.41 |

这些 loss 仅证明数值有限、训练链路通畅与真实参数更新，不用于评价调度性能或长期收敛。

现有测试：
- GPU 环境：31 passed，18 条既有 Matplotlib/pyparsing 弃用警告。
- CPU 环境：31 passed，18 条同类警告；包括 Original 对上游 commit 的精确回归。
- CPU 原 verify_environment.py：张量/DGL 检查及两模式 × 三通信配置的 6 条 forward 路径通过。
- 两环境 pip check 通过。
- git diff --check 通过，参考仓库保持干净。

## CPU 与第一阶段成果保护

所有 CPU 验证使用 PYTHONDONTWRITEBYTECODE=1 和 python -B，pytest 关闭 cacheprovider，输出指定到新的 gpu_validation 目录。

安装前后逐文件 SHA-256 与符号链接目标校验：
- CPU 环境 33,218 个文件：无新增、删除或内容变化。
- CPU 包清单一致，原 environment.yml 内容一致。
- 第一阶段 results/stage1 的 60 个文件：无新增、删除或内容变化。
- 项目原有 54 个源文件/资源：无修改或删除，仅增加 scripts/verify_gpu.py。
- CPU 版本仍为 Python 3.9.23、PyTorch 1.12.1+cpu、DGL 0.9.0、Gym 0.21.0。

详见 preservation_report.json。未修改 base 中的包；Conda 安装仅更新自身共用包缓存及新环境。未覆盖 CPU 环境配置或阶段一文档/结果。

## 运行方法

以下命令在容器中执行：

    cd /workspace/uav_edgeric_stage1/uav_bs_ctrl
    export DGLBACKEND=pytorch MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
    /opt/conda/bin/conda run --no-capture-output -n uav_edgeric_gpu python -B scripts/verify_gpu.py
    /opt/conda/bin/conda run --no-capture-output -n uav_edgeric_gpu python -B -m pytest -q -p no:cacheprovider tests

只执行 CUDA 基础计算：

    /opt/conda/bin/conda run --no-capture-output -n uav_edgeric_gpu python -B scripts/verify_gpu.py --primitives-only

CPU 回归：

    /opt/conda/bin/conda run --no-capture-output -n uav_edgeric_py39 python -B -m pytest -q -p no:cacheprovider tests
    /opt/conda/bin/conda run --no-capture-output -n uav_edgeric_py39 python -B scripts/verify_environment.py ../results/gpu_validation/cpu_validation.json

Windows 非交互 SSH 示例：

    ssh -o BatchMode=yes -o ConnectTimeout=10 dell@100.64.193.15 "docker exec -w /workspace/uav_edgeric_stage1/uav_bs_ctrl -e DGLBACKEND=pytorch -e MPLBACKEND=Agg -e OMP_NUM_THREADS=1 -e PYTHONDONTWRITEBYTECODE=1 Liuyifei-env /opt/conda/bin/conda run --no-capture-output -n uav_edgeric_gpu python -B scripts/verify_gpu.py"

脚本默认输出 results/gpu_validation；保留多次验证请用 --output 指定其下新子目录。

原 check_args_sanity 仅把 device='cuda'、cuda_index=0 转成 cuda:0；不要向该原函数传 device='cuda:0'，否则它会退回 CPU。新脚本按原入口规范传值并断言结果为 cuda:0，未修改原函数。

## 环境配置与文件交付

- environment-gpu.yml：从实际通过验证的环境导出，固定 Linux x86_64 包版本/构建，添加 DGL 官方 CUDA wheel 源，移除绝对 prefix 和 defaults。
- README_GPU.md：本说明。
- uav_bs_ctrl/scripts/verify_gpu.py：唯一新增项目源码文件。
- results/gpu_validation/：安装日志、GPU/CPU 测试日志和 JUnit XML、每个训练配置的 JSON、gpu_validation.json、cuda_primitives.json、CPU 校验报告、显卡输出、包版本清单和 DGL wheel。
- environment-gpu-resolved.yml 保留原始导出；conda-explicit-linux-64.txt 记录确切 Conda 包 URL，pip-freeze.txt 记录 Python 包版本。

仅在新机器/目标环境尚不存在时：

    cd /workspace/uav_edgeric_stage1
    /opt/conda/bin/conda env create -f environment-gpu.yml

严格按已验证包来源复现，可先按 explicit 清单创建环境，再安装 GPU 配置文件 pip 段的包（pip-freeze 含 Conda 管理的包，不要用它覆盖整套 Conda 依赖）。没有额外创建第二份环境来完整重演安装，配置来自实际已验证环境。CPU 配置文件 environment.yml 未覆盖。

## 限制与未执行项

当前没有阻塞本次验收的未解决错误。只验证单 GPU、小批量、短训练，不证明长期收敛或完整大 batch 训练的显存需求。未执行百万步训练、MPI 多进程/多 GPU、QMIX 或 run_exp3 之外其他通信模块。没有修改调度、reward、动作空间或轨迹步长，没有双时间尺度扩展。未提交或推送 Git 修改。
