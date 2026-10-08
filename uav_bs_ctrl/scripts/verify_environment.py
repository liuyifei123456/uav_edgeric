"""CPU import, tensor, graph, and existing MARL forward-path smoke checks."""
import contextlib
import io
import json
import platform
import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy
import pandas
import torch
import dgl
import dgl.function as fn
import gym
import mpi4py
from mpi4py import MPI
from algos.madrqn.run import train
from algos.madrqn.config import DEFAULT_CONFIG
from algos.madrqn.learner import MultiAgentQLearner
from algos.madrqn.utils.env_wrappers import make_env
from envs.mubs_cov.mubs_cov import MultiUbsCoverageEnv


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    x = torch.tensor([2., 3.], requires_grad=True)
    (x.square().sum()).backward()
    assert torch.equal(x.grad, torch.tensor([4., 6.]))
    g = dgl.graph(([0, 1], [1, 2]), num_nodes=3)
    g.ndata["x"] = torch.tensor([[1.], [2.], [3.]])
    g.update_all(fn.copy_u("x", "m"), fn.sum("m", "y"))
    assert torch.equal(g.ndata["y"], torch.tensor([[0.], [1.], [2.]]))

    forward_checks = []
    for mode in ("original", "edgeric_pf"):
        for protocol in (None, "tarmac", "disc"):
            config = dict(DEFAULT_CONFIG, o="gnn", c=protocol, device="cpu",
                          anneal_lr=False, norm_r=False)
            args = SimpleNamespace(**config)
            env = make_env(lambda: MultiUbsCoverageEnv("4ubs", record=False,
                                                       scheduler_mode=mode), args)
            with contextlib.redirect_stdout(io.StringIO()):
                learner = MultiAgentQLearner(env.get_env_info(), args)
            obs, state = env.reset()
            acts, hidden = learner.act(obs, learner.init_hidden(), 0.)
            assert len(acts) == 4 and torch.isfinite(hidden).all()
            obs, state, reward, done, info = env.step(acts)
            acts, hidden = learner.act(obs, hidden, 0.)
            assert len(acts) == 4 and torch.isfinite(hidden).all()
            forward_checks.append(dict(mode=mode, protocol=protocol, status="PASS"))
    versions = {m.__name__: m.__version__ for m in
                (np, scipy, pandas, torch, dgl, gym, mpi4py)}
    result = dict(python=platform.python_version(), versions=versions,
                  mpi_library=MPI.Get_library_version(), mpi_rank=MPI.COMM_WORLD.rank,
                  cuda_available=torch.cuda.is_available(), cuda_device_count=torch.cuda.device_count(),
                  torch_cuda_build=torch.version.cuda, cpu_tensor_and_dgl="PASS",
                  marl_gnn_forward_checks=forward_checks, imports="IMPORT_OK")
    print(json.dumps(result, indent=2))
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../results/stage1/environment_validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
