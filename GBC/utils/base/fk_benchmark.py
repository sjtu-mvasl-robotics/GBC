from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch
import yaml

from GBC.utils.base.base_fk import RobotKinematics
from GBC.utils.base.cached_fk import CachedTargetLinkKinematics


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(fn, device: torch.device, warmup: int, iters: int) -> tuple[float, float, float, float, float | None]:
    for _ in range(warmup):
        fn()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
    return statistics.mean(samples), statistics.median(samples), min(samples), max(samples), peak_mb


def _load_mujica_base(cfg_path: str | Path) -> dict:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)["base"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RobotKinematics with CachedTargetLinkKinematics.")
    parser.add_argument(
        "--cfg",
        default="/cpfs/user/yaoyifei/codespace/gbc/GBC_MUJICA/GBC_MUJICA/data_preparation/humanoid_cfg/unitree_g1.yaml",
        help="MUJICA humanoid yaml with base.urdf_path and base.foot_names.",
    )
    parser.add_argument("--batch", type=int, default=1280, help="Number of action rows to evaluate.")
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    base = _load_mujica_base(args.cfg)
    target_links = tuple(base["foot_names"])
    baseline = RobotKinematics(base["urdf_path"], device=str(device))
    baseline.set_target_links(list(target_links))
    cached = CachedTargetLinkKinematics(base["urdf_path"], target_links, device=device)

    actions = torch.randn(args.batch, baseline.num_dofs, device=device) * 0.15
    ref_actions = actions.detach().clone().requires_grad_(True)
    cached_actions = actions.detach().clone().requires_grad_(True)
    ref_out = baseline(ref_actions)
    cached_out = cached(cached_actions)
    max_pos_err = (ref_out - cached_out).abs().max()
    ref_out.square().mean().backward()
    cached_out.square().mean().backward()
    max_grad_err = (ref_actions.grad - cached_actions.grad).abs().max()

    def baseline_fwd():
        with torch.no_grad():
            return baseline(actions).square().mean()

    def cached_fwd():
        with torch.no_grad():
            return cached(actions).square().mean()

    def baseline_train():
        x = actions.detach().clone().requires_grad_(True)
        loss = baseline(x).square().mean()
        loss.backward()
        return loss

    def cached_train():
        x = actions.detach().clone().requires_grad_(True)
        loss = cached(x).square().mean()
        loss.backward()
        return loss

    baseline_fwd_stats = _time_call(baseline_fwd, device, args.warmup, args.iters)
    cached_fwd_stats = _time_call(cached_fwd, device, args.warmup, args.iters)
    baseline_train_stats = _time_call(baseline_train, device, args.warmup, args.iters)
    cached_train_stats = _time_call(cached_train, device, args.warmup, args.iters)

    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"cfg={args.cfg}")
    print(f"urdf={base['urdf_path']}")
    print(f"target_links={target_links}")
    print(f"batch={args.batch} dofs={baseline.num_dofs}")
    print(f"max_position_error={float(max_pos_err.detach().cpu()):.9g}")
    print(f"max_gradient_error={float(max_grad_err.detach().cpu()):.9g}")
    print(
        "baseline_forward "
        f"mean_ms={baseline_fwd_stats[0]:.3f} median_ms={baseline_fwd_stats[1]:.3f} "
        f"min_ms={baseline_fwd_stats[2]:.3f} max_ms={baseline_fwd_stats[3]:.3f}"
    )
    print(
        "cached_forward "
        f"mean_ms={cached_fwd_stats[0]:.3f} median_ms={cached_fwd_stats[1]:.3f} "
        f"min_ms={cached_fwd_stats[2]:.3f} max_ms={cached_fwd_stats[3]:.3f}"
    )
    print(
        "baseline_train "
        f"mean_ms={baseline_train_stats[0]:.3f} median_ms={baseline_train_stats[1]:.3f} "
        f"peak_mb={baseline_train_stats[4]:.1f}"
    )
    print(
        "cached_train "
        f"mean_ms={cached_train_stats[0]:.3f} median_ms={cached_train_stats[1]:.3f} "
        f"peak_mb={cached_train_stats[4]:.1f}"
    )


if __name__ == "__main__":
    main()
