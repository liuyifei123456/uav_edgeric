"""CUDA primitives and short updates through the existing MARL learner.

Run in uav_edgeric_gpu. Does not change scheduler/model/training source.
Smoke-only reductions: batch_size=2, replay_size=8, two full 4ubs episodes
and two updates per case. Network/lr settings match run_exp3.py. Initial
positions form a synthetic covered layout to exercise nonempty GT graphs.
"""
import argparse
import contextlib
import copy
import gc
import io
import json
import platform
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import dgl
from dgl.nn.pytorch import GraphConv
from mpi4py import MPI
from algos.common import check_args_sanity
from algos.madrqn.config import DEFAULT_CONFIG
from algos.madrqn.learner import MultiAgentQLearner
from algos.madrqn.utils.env_wrappers import make_env
from envs.mubs_cov.mubs_cov import MultiUbsCoverageEnv


def cuda_primitives():
    assert torch.cuda.is_available(), "CUDA must be available; no CPU fallback"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(123)
    a = torch.randn(128, 128, device=device, requires_grad=True)
    b = torch.randn(128, 128, device=device, requires_grad=True)
    product = a @ b
    assert product.device == device
    torch.testing.assert_close(product.detach().cpu(), a.detach().cpu() @ b.detach().cpu(),
                               rtol=1e-3, atol=1e-3)
    loss = product.square().mean()
    loss.backward()
    for t in (a, b):
        assert t.grad.device == device and torch.isfinite(t.grad).all()
        assert torch.count_nonzero(t.grad) > 0
    graph = dgl.graph(([0, 1], [1, 2])).to(device)
    x = torch.randn(3, 8, device=device, requires_grad=True)
    conv = GraphConv(8, 4, allow_zero_in_degree=True).to(device)
    y = conv(graph, x)
    graph_loss = y.square().mean()
    graph_loss.backward()
    assert graph.device == device and y.device == device
    assert x.grad.device == device and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad) > 0
    assert conv.weight.grad.device == device and torch.isfinite(conv.weight.grad).all()
    assert torch.count_nonzero(conv.weight.grad) > 0
    torch.cuda.synchronize()
    return dict(status="PASS", torch_version=torch.__version__, dgl_version=dgl.__version__,
                python=platform.python_version(), numpy=np.__version__,
                cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                compute_capability=list(torch.cuda.get_device_capability(0)),
                torch_cuda_built=torch.backends.cuda.is_built(),
                matrix_loss=float(loss.item()), graphconv_loss=float(graph_loss.item()),
                tensor_device=str(product.device), graph_device=str(graph.device),
                graphconv_backward="PASS", matmul_backward="PASS",
                mpi_version=MPI.Get_library_version().splitlines()[0])


def covered_env(mode, fast_scheduling=False, dt_sched=1.0):
    env = MultiUbsCoverageEnv("4ubs", scheduler_mode=mode, record=False,
                              fast_scheduling=fast_scheduling, dt_sched=dt_sched)
    env.map = copy.copy(env.map)
    uavs = np.array([[1000., 1000.], [1120., 1000.],
                     [2000., 2000.], [2120., 2000.]], dtype=np.float32)
    rng = np.random.RandomState(123)
    gts = np.vstack([rng.uniform(-60., 60., (25, 2)) + [1060., 1000.],
                     rng.uniform(-60., 60., (25, 2)) + [2060., 2000.]]).astype(np.float32)
    env.map.set_positions = lambda: dict(ubs=uavs.copy(), gt=gts.copy())
    return env


def train_case(mode, protocol, output, fast_scheduling=False, dt_sched=1.0):
    started = time.monotonic()
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    # Match run_exp3 network/lr/algorithm settings. Only buffer/batch/run
    # duration are reduced. Default 50-step episode and 40s dt are unchanged.
    config = dict(DEFAULT_CONFIG, device="cuda", cuda_index=0, o="gnn", c=protocol,
                  hidden_size=256, n_layers=2, msg_size=64, lr=2.5e-4, polyak=0.999,
                  double_q=True, dueling=False, mixer=False, norm_r=True, anneal_lr=True,
                  batch_size=2, replay_size=8, max_seq_len=None)
    args = check_args_sanity(SimpleNamespace(**config))
    assert args.device == "cuda:0"
    env = make_env(lambda: covered_env(mode, fast_scheduling, dt_sched), args)
    with contextlib.redirect_stdout(io.StringIO()):
        learner = MultiAgentQLearner(env.get_env_info(), args)
    device = torch.device("cuda:0")
    assert all(p.device == device for p in learner.policy_net.parameters())
    assert all(p.device == device for p in learner.target_net.parameters())
    assert env.dt == 40 and env.n_actions == 9
    assert learner.max_seq_len == 50

    observed = dict(forward_calls=0, grad_enabled_forward_calls=0,
                    encoder_calls=0, cuda_losses=[], replay_sample_calls=0)
    def forward_hook(module, inputs, outputs):
        graph, hidden = inputs
        assert graph.device == device and hidden.device == device
        assert all(v.device == device for v in graph.ndata["feat"].values())
        assert all(v.device == device and torch.isfinite(v).all() for v in outputs)
        observed["forward_calls"] += 1
        observed["grad_enabled_forward_calls"] += int(torch.is_grad_enabled())

    def encoder_hook(module, inputs, result):
        assert result.device == device and torch.isfinite(result).all()
        observed["encoder_calls"] += 1

    def loss_hook(module, inputs, result):
        assert result.device == device and result.requires_grad and torch.isfinite(result)
        observed["cuda_losses"].append(float(result.detach().item()))

    # Observation hooks only; actual sampler, forward, loss, backward and
    # optimizer implementations remain the original project's methods.
    hooks = [learner.policy_net.register_forward_hook(forward_hook),
             learner.policy_net.enc.register_forward_hook(encoder_hook),
             learner.loss_fn.register_forward_hook(loss_hook)]
    original_sample = learner.buffer.sample
    def observed_sample(batch_size):
        samples = original_sample(batch_size)
        assert len(samples) == batch_size
        observed["replay_sample_calls"] += 1
        return samples
    learner.buffer.sample = observed_sample

    torch.cuda.reset_peak_memory_stats(device)
    interactions = 0
    initial_seen_edges = []
    episode_radio_stats = []
    for episode in range(2):
        (obs, state), hidden = env.reset(), learner.init_hidden()
        initial_seen_edges.append(obs.num_edges("seen"))
        assert initial_seen_edges[-1] > 0
        for _ in range(env.episode_limit):
            acts, next_hidden = learner.act(obs, hidden, 0.2)
            next_obs, next_state, reward, done, info = env.step(acts)
            assert np.isfinite(reward).all()
            learner.cache(obs, hidden, state, acts, reward,
                          next_obs, next_hidden, next_state, done, info["BadMask"])
            obs, state, hidden = next_obs, next_state, next_hidden
            interactions += 1
        assert done
        stats = env.get_scheduling_stats()
        if fast_scheduling:
            assert stats["scheduling_count"] == 50 * env.n_substeps
            assert stats["service_time_s"] == 2000
        else:
            assert stats["scheduling_count"] == 51
        episode_radio_stats.append(stats)
    assert len(learner.buffer) == 2

    updates = []
    for _ in range(2):
        before = {name: p.detach().clone() for name, p in learner.policy_net.named_parameters()}
        diagnostic = learner.update()  # Actual replay/BPTT/backward/AdamW path.
        torch.cuda.synchronize()
        assert np.isfinite(diagnostic["LossQ"])
        gradients = {name: p.grad for name, p in learner.policy_net.named_parameters() if p.grad is not None}
        assert gradients and all(g.device == device and torch.isfinite(g).all() for g in gradients.values())
        nonzero = [name for name, grad in gradients.items() if torch.count_nonzero(grad).item() > 0]
        deltas = {name: float((p.detach() - before[name]).abs().max().item())
                  for name, p in learner.policy_net.named_parameters()}
        changed = [name for name, delta in deltas.items() if delta > 0]
        assert changed and any(name.startswith("enc.") for name in changed)
        assert any(name.startswith("enc.") for name in nonzero)
        assert any(name.startswith("f_out.") for name in nonzero)
        recurrent_prefix = "rnn." if protocol is None else "f_comm."
        assert any(name.startswith(recurrent_prefix) for name in nonzero)
        # AdamW moment tensors must reside on CUDA. Scalar step counters may
        # legitimately be CPU tensors in PyTorch's non-capturable optimizer.
        for state_values in learner.optimizer.state.values():
            assert state_values["exp_avg"].device == device
            assert state_values["exp_avg_sq"].device == device
        updates.append(dict(loss=float(diagnostic["LossQ"]), gradient_device="cuda:0",
                            nonzero_gradient_parameters=nonzero,
                            changed_parameters=changed, max_parameter_delta=max(deltas.values()),
                            optimizer_moments_device="cuda:0"))
    assert observed["replay_sample_calls"] == 2
    assert observed["grad_enabled_forward_calls"] > 0
    assert len(observed["cuda_losses"]) == 2
    result = dict(status="PASS", mode=mode, protocol=protocol, map_id="4ubs",
                  fast_scheduling=fast_scheduling, dt_sched_s=dt_sched if fast_scheduling else env.dt,
                  episode_radio_stats=episode_radio_stats,
                  layout="synthetic covered positions; original map/radio/actions/dt",
                  config=vars(args), interactions=interactions, episodes=2,
                  replay_sequences=len(learner.buffer), initial_seen_edges=initial_seen_edges,
                  updates=updates, observed=observed, device=str(learner.device),
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                  elapsed_seconds=time.monotonic() - started)
    case_name = mode + "_" + (protocol or "none")
    (output / (case_name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "mode", "protocol", "device",
                     "interactions", "peak_allocated_mib", "elapsed_seconds")},
                     ensure_ascii=False), flush=True)
    for hook in hooks:
        hook.remove()
    learner.buffer.sample = original_sample
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("../results/gpu_validation"))
    parser.add_argument("--primitives-only", action="store_true")
    parser.add_argument("--fast-scheduling", action="store_true")
    parser.add_argument("--dt-sched", type=float, default=1.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    result = dict(primitives=cuda_primitives(), marl=[])
    (args.output / "cuda_primitives.json").write_text(json.dumps(result["primitives"], indent=2) + "\n")
    print(json.dumps(result["primitives"], indent=2), flush=True)
    if not args.primitives_only:
        for mode in ("original", "edgeric_pf"):
            for protocol in (None, "tarmac", "disc"):
                result["marl"].append(train_case(mode, protocol, args.output,
                                                args.fast_scheduling, args.dt_sched))
                gc.collect()
                torch.cuda.empty_cache()
        (args.output / "gpu_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("GPU_VALIDATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
