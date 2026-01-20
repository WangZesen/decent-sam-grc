import os
import sys
import time
import wandb
import pandas as pd
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch.distributed as dist
from typing import Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from src.data import CifarLoader, Dataset
from src.conf import Config, load_all_configs, Topology
from src.model import get_model
from src.utils import get_param_groups, get_scheduler, collect_env, get_group_name, get_adaptive_gamma, get_run_name
import torch_xla.debug.metrics as met


@dataclass
class CommGroup:
    ranks: list[int]
    weight: float
    group: dist.ProcessGroup


class DecentDP(torch.nn.Module):
    def __init__(
        self,
        base_module: torch.nn.Module,
        topology: Topology,
        bucket_size: int = 13107200,
    ):
        super().__init__()
        
        device = torch_xla.device()
        self._device = device

        self._module = base_module.to(device)
        self._bucket_size = bucket_size
        self._topology = topology

        # acquire distributed info from env variables
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()
        self._local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        # sync parameters at start
        self._sync_params()

        # remap parameters into buckets
        self._create_buckets()

        # create communication groups based on topology
        self._create_comm_groups()

        # training step counter
        self._step = 0

        # comm op handle
        self._comm_op: Optional[dist.Work] = None

    @torch.no_grad()
    def _sync_params(self):
        data = []
        for param in self._module.parameters():
            data.append(param.data)
        xm.collective_broadcast(data, root_ordinal=0)

    @torch.no_grad()
    def _create_buckets(self):
        self._bucket_total_size = sum([self._align(param.numel()) for param in self._module.parameters()])
        self._param_bucket = torch.zeros((self._bucket_total_size), dtype=torch.float32, device=self._device)
        self._comm_bucket = torch.zeros((self._bucket_total_size), dtype=torch.float32, device=self._device)

        offset = 0
        for param in self._module.parameters():
            size = param.numel()
            aligned_size = self._align(size)

            assert param.is_contiguous(), "Parameters must be contiguous"

            # Copy parameter data into the bucket
            chunk = self._param_bucket[offset : offset + size]
            chunk.copy_(param.data.view(-1))

            # Store a view of the chunk back to the parameter for easy access
            param.data = chunk.view_as(param)
            offset += aligned_size
        self._comm_bucket.copy_(self._param_bucket)

    @torch.no_grad()
    def _create_comm_groups(self):
        match self._topology:
            case Topology.RING:
                self._comm_groups: list[list[list[int]]] = [[]]
                for i in range(0, self._world_size, 2):
                    ranks = sorted([i, (i + 1) % self._world_size])
                    self._comm_groups[-1].append(ranks)
                self._comm_groups.append([])
                for i in range(1, self._world_size, 2):
                    ranks = sorted([i, (i + 1) % self._world_size])
                    self._comm_groups[-1].append(ranks)
            case Topology.COMPLETE:
                ranks = list(range(self._world_size))
                self._comm_groups: list[list[list[int]]] = [[ranks]]
            case Topology.EXP:
                self._comm_groups: list[list[list[int]]] = []
                exp = 1
                while exp < self._world_size:
                    self._comm_groups.append([])
                    for i in range(self._world_size):
                        j = i ^ exp
                        if i < j and j < self._world_size:
                            ranks = sorted([i, j])
                            self._comm_groups[-1].append(ranks)
                    exp <<= 1
            case _:
                raise ValueError(f"Unsupported topology: {self._topology}")

    @torch.no_grad()
    def mix(self, gamma: float = 1.0):
        self._param_bucket.mul_(1 - gamma)
        self._param_bucket.add_(self._comm_bucket, alpha=gamma)

    @torch.no_grad()
    def start_comm(self):
        weight = 0.0
        for group_ranks in self._comm_groups[self._step % len(self._comm_groups)]:
            if self._rank in group_ranks:
                weight = 1.0 / len(group_ranks)
                break
        self._comm_bucket.copy_(self._param_bucket).mul_(weight)
        xm.all_reduce(
            xm.REDUCE_SUM,
            self._comm_bucket,
            groups=self._comm_groups[self._step % len(self._comm_groups)],
        )

    @torch.no_grad()
    def global_avg(self) -> float:
        self._backup = self._param_bucket.clone()
        xm.all_reduce(xm.REDUCE_SUM, self._param_bucket)
        self._param_bucket.div_(self._world_size)
        d2c = torch.norm(self._param_bucket - self._backup)
        xm.all_reduce(xm.REDUCE_SUM, d2c)
        d2c = d2c / self._world_size
        return d2c.item()

    @torch.no_grad()
    def restore(self):
        self._param_bucket.copy_(self._backup)
        del self._backup

    @torch.no_grad()
    def sync_buffers(self):
        for buffer in self._module.buffers():
            if buffer.dtype in [torch.float16, torch.float32, torch.float64]:
                xm.all_reduce(xm.REDUCE_SUM, buffer)
                buffer.div_(self._world_size)
            else:
                pass
        torch_xla.sync()

    def _align(self, size: int):
        return ((size + 31) // 32) * 32

    def forward(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    def parameters(self, recurse: bool = True):
        yield from self._module.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        yield from self._module.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)

    def train(self, mode: bool = True):
        self._module.train(mode)
        return self

    def eval(self):
        self._module.eval()
        return self


def train_epoch(
    model: DecentDP,
    train_loader: CifarLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    cfg: Config,
    max_lr: float,
) -> Tuple[float, float]:
    model.train()
    total_loss_tpu = torch.tensor(0.0, device=torch_xla.device(), requires_grad=False)
    num_samples = 0
    gamma = 1.0

    torch_xla.sync()
    start_time = time.time()

    for images, labels in train_loader:
        # gamma = get_adaptive_gamma(cfg, scheduler.get_last_lr()[0], max_lr, epoch)
        model.start_comm()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        model.mix(gamma)
        optimizer.step()
        scheduler.step()
        
        # batch_size = images.size(0)
        # total_loss_tpu += loss.clone().detach() * batch_size
        # num_samples += batch_size
        torch_xla.sync(wait=False)
    print(num_samples, flush=True)

    torch_xla.sync()
    xm.xla_rendezvous(b"check_loss_sync")
    end_time = time.time()

    data = torch.tensor([total_loss_tpu, num_samples], device=torch_xla.device(), requires_grad=False)
    xm.all_reduce(xm.REDUCE_SUM, data)
    avg_loss = (data[0] / data[1]).item()
    torch_xla.sync()
    if xm.is_master_ordinal(local=True):
        with open(f"./report_{epoch}.txt", "w") as f:
            print(met.short_metrics_report(), flush=True, file=f)
            print(met.metrics_report(), flush=True, file=f)
    return avg_loss, end_time - start_time


def eval_epoch(
    model: DecentDP,
    train_loader: CifarLoader,
    test_loader: CifarLoader,
    criterion: torch.nn.Module,
    cfg: Config,
):
    d2c = model.global_avg()
    with torch.no_grad():
        model.train()
        for images, _ in train_loader:
            model(images)
            torch_xla.sync()
    model.eval()
    model.sync_buffers()

    total_loss = torch.tensor(0.0, device=torch_xla.device(), requires_grad=False)
    total_correct = torch.tensor(0.0, device=torch_xla.device(), requires_grad=False)
    total_samples = 0
    torch_xla.sync()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            total_loss += loss.clone().detach() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum()
            total_samples += batch_size
            torch_xla.sync()
    data = torch.tensor([total_loss, total_correct, total_samples], device=torch_xla.device(), requires_grad=False)
    xm.all_reduce(xm.REDUCE_SUM, data)
    avg_loss = (data[0] / data[2]).item()
    accuracy = (data[1] / data[2]).item()

    model.restore()
    torch_xla.sync()
    return avg_loss, accuracy, d2c


def main(rank: int):
    cfg: Config = load_all_configs(sys.argv[1:])
    assert cfg.trainer.name == "decent", "This script only supports DecentDP trainer"

    dist.init_process_group(
        backend="xla",
        init_method="xla://",
    )

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    logger.remove()

    if xm.is_master_ordinal(local=True):
        logger.add(sys.stdout, level="TRACE")
        logger.info(cfg)

        cfg_obj = cfg.model_dump()
        env = collect_env()
        cfg_obj["env"] = env.model_dump()
        logger.info(env)
    if xm.is_master_ordinal(local=False):
        pass
        # wandb.init(
        #     project=cfg.log.project,
        #     config=cfg_obj,
        #     dir=os.environ.get("TMPDIR", "/tmp"),
        #     save_code=True,
        #     group=get_group_name(cfg, env),
        #     name=get_run_name(cfg, env),
        # )

    torch.manual_seed(cfg.seed)
    torch_xla.manual_seed(cfg.seed)

    train_ds = CifarLoader(
        ds_name=cfg.dataset,
        train=True,
        batch_size=cfg.batch_size,
        rank=rank,
        num_replicas=world_size,
        base_seed=cfg.seed,
    )
    test_ds = CifarLoader(
        ds_name=cfg.dataset,
        train=False,
        batch_size=cfg.batch_size,
        rank=rank,
        num_replicas=world_size,
    )

    model = get_model(cfg.model, num_classes=10 if cfg.dataset == Dataset.CIFAR10 else 100)
    # model.forward = torch.compile(model.forward)
    model = DecentDP(model, topology=cfg.trainer.topology)

    optimizer = torch.optim.SGD(
        get_param_groups(model, weight_decay=cfg.optimizer.weight_decay),
        lr=cfg.base_lr,
        momentum=cfg.optimizer.momentum,
    )
    criterion = torch.nn.CrossEntropyLoss().to(torch_xla.device())
    iters_per_epoch = len(train_ds)
    scheduler = get_scheduler(optimizer, iters_per_epoch, cfg)

    total_train_time = 0.0
    stats = pd.DataFrame(
        columns=[
            "epoch",
            "train_loss",
            "test_loss",
            "test_acc",
            "d2c",
            "epoch_time",
            "total_train_time",
            "lr",
            "gamma"
        ]
    )

    max_lr = 0.0

    for epoch in range(cfg.epochs):
        train_ds.set_epoch(epoch)

        if (cfg.trainer.mix.name == "adaptive") and (epoch == cfg.trainer.mix.start_epoch):
            max_lr = scheduler.get_last_lr()[0]
            if xm.is_master_ordinal(local=True):
                logger.info(f"Adaptive mixing starts at epoch {epoch}, max_lr={max_lr:.5f}")

        train_loss, train_time = train_epoch(model, train_ds, criterion, optimizer, scheduler, epoch, cfg, max_lr)
        if (epoch + 1) % cfg.log.eval_interval == 0 or epoch == cfg.epochs - 1:
            test_loss, test_acc, d2c = eval_epoch(
                model,
                train_ds,
                test_ds,
                criterion,
                cfg
            )
            if xm.is_master_ordinal(local=True):
                gamma = get_adaptive_gamma(cfg, scheduler.get_last_lr()[0], max_lr, epoch)
                logger.info(
                    f"Epoch [{epoch + 1:3d}/{cfg.epochs}], "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Test Loss: {test_loss:.4f}, "
                    f"Test Acc: {test_acc * 100:.3f}%, "
                    f"D2C: {d2c:.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.5f}, "
                    f"Epoch Time: {train_time:.2f}s",
                )
                total_train_time += train_time
                stats.loc[len(stats)] = [
                    epoch + 1,
                    train_loss,
                    test_loss,
                    test_acc,
                    d2c,
                    train_time,
                    total_train_time,
                    scheduler.get_last_lr()[0],
                    gamma
                ]
                if xm.is_master_ordinal(local=False):
                    pass
                    # data = {
                    #     "metric/train_loss": train_loss,
                    #     "metric/test_loss": test_loss,
                    #     "metric/test_acc": test_acc,
                    #     "metric/d2c": d2c,
                    #     "metric/epoch": epoch + 1,
                    #     "metric/epoch_time": train_time,
                    #     "metric/total_train_time": total_train_time,
                    #     "metric/lr": scheduler.get_last_lr()[0],
                    #     "metric/gamma": gamma
                    # }
                    # wandb.log(data, step=epoch + 1)
        else:
            if xm.is_master_ordinal(local=True):
                logger.info(f"Epoch [{epoch + 1}/{cfg.epochs}] Train Loss: {train_loss:.5f}")

    if xm.is_master_ordinal(local=True):
        stats.to_csv(f"./logs/{os.environ.get('SLURM_JOB_ID')}/stats.csv", index=False)
        # wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    torch_xla.launch(main)
