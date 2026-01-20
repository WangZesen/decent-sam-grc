import os
import subprocess
import torch
import torch.nn as nn
import torch_xla.runtime as xr
from src.conf import Config, Env


def get_param_groups(module: nn.Module, weight_decay: float) -> list[dict]:
    decay = []
    no_decay = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def get_scheduler(
    optimizer: torch.optim.Optimizer, iters_per_epoch: int, cfg: Config
) -> torch.optim.lr_scheduler.LRScheduler:
    match cfg.scheduler.name:
        case "cosine":
            if cfg.scheduler.warmup_epochs > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=cfg.scheduler.warmup_decay,
                    total_iters=cfg.scheduler.warmup_epochs * iters_per_epoch,
                )
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=(cfg.epochs - cfg.scheduler.warmup_epochs) * iters_per_epoch,
                    eta_min=cfg.scheduler.eta_min,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[cfg.scheduler.warmup_epochs * iters_per_epoch],
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cfg.epochs * iters_per_epoch,
                    eta_min=cfg.scheduler.eta_min,
                )
        case _:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler.name}")
    return scheduler


def collect_env() -> Env:
    env = {
        "world_size": xr.world_size(),
        "gpu": "",
        "node_list": "",
    }
    return Env(**env)


def get_group_name(cfg: Config, env: Env) -> str:
    if cfg.trainer.name == "sync":
        return f"{cfg.trainer.name}_bs{cfg.batch_size}_ws{env.world_size}"
    elif cfg.trainer.name == "decent":
        if cfg.trainer.mix.name == "normal":
            return f"{cfg.trainer.name}_{cfg.trainer.topology.value}_bs{cfg.batch_size}_ws{env.world_size}"
        elif cfg.trainer.mix.name == "adaptive":
            return f"{cfg.trainer.name}_{cfg.trainer.topology.value}_amix{cfg.trainer.mix.p:.1f}@{cfg.trainer.mix.min_gamma:.2f}@{cfg.trainer.mix.start_epoch}_bs{cfg.batch_size}_ws{env.world_size}"
        else:
            raise ValueError(f"Unknown mix config: {cfg.trainer.mix.name}")
    else:
        raise ValueError(f"Unknown trainer: {cfg.trainer.name}")


def get_run_name(cfg: Config, env: Env) -> str:
    suffix = f"bs{cfg.batch_size}_ws{env.world_size}_seed{cfg.seed}_id{os.environ.get('SLURM_JOB_ID', 'local')}"
    if cfg.trainer.name == "sync":
        return f"{cfg.trainer.name}_{suffix}"
    elif cfg.trainer.name == "decent":
        if cfg.trainer.mix.name == "normal":
            return f"{cfg.trainer.name}_{cfg.trainer.topology.value}_{suffix}"
        elif cfg.trainer.mix.name == "adaptive":
            return f"{cfg.trainer.name}_{cfg.trainer.topology.value}_amix{cfg.trainer.mix.p:.1f}@{cfg.trainer.mix.min_gamma:.2f}@{cfg.trainer.mix.start_epoch}_{suffix}"
        else:
            raise ValueError(f"Unknown mix config: {cfg.trainer.mix.name}")
    else:
        raise ValueError(f"Unknown trainer: {cfg.trainer.name}")


def get_adaptive_gamma(cfg: Config, lr: float, max_lr: float, epoch: int) -> float:
    assert cfg.trainer.name == "decent", "Adaptive gamma is only for decent trainer"
    mix_cfg = cfg.trainer.mix
    match mix_cfg.name:
        case "normal":
            return 1.0
        case "adaptive":
            if epoch < mix_cfg.start_epoch:
                return 1.0
            gamma = (lr / max_lr) ** mix_cfg.p
            gamma = (1.0 - mix_cfg.min_gamma) * gamma + mix_cfg.min_gamma
            return gamma
        case _:
            raise ValueError(f"Unknown mix config: {mix_cfg.name}")
