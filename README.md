# UAV-EdgeRIC：多无人机协同轨迹优化与轻量动态无线资源调度仿真

本仓库归档 UAV-EdgeRIC 第一、第二阶段的可复现实验代码、环境配置、测试和必要结果摘要。主体项目位于 `uav_bs_ctrl/`，参考实现以固定 commit 的 Git 子模块提供。

## 已完成内容

### 第一阶段：EdgeRIC-PF 动态调度

以原始 `uav_bs_ctrl` 为主体，参考 EdgeRIC 的动态权重思想实现轻量化 EdgeRIC-PF。调度器根据当前信道估计和历史平均速率计算 PF 权重，通过改变 GT 的服务顺序影响原有 RB 分配，同时保留原 RB 分配和约束机制。支持 `original` 与 `edgeric_pf` 两种模式，并包含单元测试、Original 回归测试及对比实验。详细说明见 [README_STAGE1.md](README_STAGE1.md)。

### 第二阶段：双时间尺度仿真

UAV 轨迹决策周期保持 40 秒；无线资源调度周期配置为 1 秒，因此每个轨迹动作执行 40 个调度子步。吞吐量、历史速率、用户速率和奖励在子步上累计并在轨迹步返回，Original 与 EdgeRIC-PF 均可运行。当前采用**动作后位置固定的 endpoint-hold** 分层近似，尚未实现连续轨迹插值。详细说明见 [README_STAGE2.md](README_STAGE2.md)。

GPU 环境已完成 PyTorch CUDA、DGL 图计算以及 GNN/MARL 前向、反向和参数更新验证，见 [README_GPU.md](README_GPU.md)。

## 环境与运行

- CPU：`uav_edgeric_py39`，配置见 [environment.yml](environment.yml)。
- GPU：`uav_edgeric_gpu`，配置见 [environment-gpu.yml](environment-gpu.yml)。

在 `uav_bs_ctrl` 目录中运行：

```bash
# CPU 环境验证和全部测试
conda run -n uav_edgeric_py39 python scripts/verify_environment.py
conda run -n uav_edgeric_py39 pytest -q tests

# GPU 环境验证
conda run -n uav_edgeric_gpu python scripts/verify_gpu.py

# 阶段一、阶段二短周期对比
conda run -n uav_edgeric_py39 python scripts/compare_stage1.py
conda run -n uav_edgeric_py39 python scripts/compare_stage2.py
```

实验摘要和图表保存在 [results/](results/)；可再生的数组、模型、日志和安装 wheel 按 `.gitignore` 排除。

## 研究进度

- 第一阶段：完成并归档。
- 第二阶段：完成并归档。
- 第三阶段：尚未开展。本仓库不包含连续轨迹插值、动态信道第三阶段对比、20 种子实验或正式长期 GNN/MARL 训练成果。

## 来源与致谢

主体代码源自 [zhangxiaochen95/uav_bs_ctrl](https://github.com/zhangxiaochen95/uav_bs_ctrl)；EdgeRIC 参考代码来自 [ucsdwcsng/EdgeRIC-on-5G](https://github.com/ucsdwcsng/EdgeRIC-on-5G)，并通过 `EdgeRIC-on-5G` 子模块固定到已验证的 `srsran` commit。请遵守上游项目的版权、许可证和作者署名要求。本仓库的第一、第二阶段集成代码与实验整理不代表对上游完整算法或源码的原创声明。

## 从原研究目录继续开发

原研究目录 `/workspace/uav_edgeric_stage1/uav_bs_ctrl` 保留独立 Git 历史。完成后可在发布工作区更新 subtree：

```bash
cd /workspace/uav_edgeric_stage1/uav_bs_ctrl
git add -A && git commit -m "stage3: ..."

cd /workspace/uav_edgeric_publish
git fetch uav_bs_ctrl_local feat/edgeric-dual-timescale
git subtree pull --prefix=uav_bs_ctrl uav_bs_ctrl_local feat/edgeric-dual-timescale
git add -A && git commit -m "sync: update uav_bs_ctrl"
git push origin main
```

发布工作区只负责 GitHub 汇总，原研究目录仍是主体开发位置。第三阶段内容需经过单独审查后再同步。
