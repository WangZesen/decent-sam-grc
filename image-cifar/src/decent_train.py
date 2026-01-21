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
import torchvision as tv
from torch_xla.distributed.parallel_loader import MpDeviceLoader

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
        # self._local_world_size =
        self._local_rank = xm.get_local_ordinal()

        # sync parameters at start
        self._sync_params()

        # remap parameters into buckets
        self._create_buffers()

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
        torch_xla.sync()

    @torch.no_grad()
    def _create_buffers(self):
        self._comm_buffers = []
        self._backup_buffers = []
        self._params = [p.data for p in self._module.parameters()]
        for param in self._module.parameters():
            buf = torch.zeros_like(param.data)
            self._comm_buffers.append(buf)
            buf = torch.zeros_like(param.data)
            self._backup_buffers.append(buf)
        torch._foreach_copy_(self._comm_buffers, self._params)
        torch._foreach_copy_(self._backup_buffers, self._params)

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
                self._weight = 0.5
            case Topology.COMPLETE:
                ranks = list(range(self._world_size))
                self._comm_groups: list[list[list[int]]] = [[ranks]]
                self._weight = 1.0 / self._world_size
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
                self._weight = 0.5
            case _:
                raise ValueError(f"Unsupported topology: {self._topology}")

    @torch.no_grad()
    def mix(self, gamma: float = 1.0):
        # torch._foreach_mul_(self._params, 1 - gamma)
        # torch._foreach_add_(self._params, self._comm_buffers, alpha=gamma)
        xm.wait_device_ops()
        torch._foreach_copy_(self._params, self._comm_buffers)

    @torch.no_grad()
    def start_comm(self):
        torch._foreach_copy_(self._comm_buffers, self._params)
        xm.all_reduce(
            xm.REDUCE_SUM,
            self._comm_buffers,
            scale=self._weight,
            groups=self._comm_groups[self._step % len(self._comm_groups)],
        )
        self._step += 1

    @torch.no_grad()
    def global_avg(self) -> float:
        torch._foreach_copy_(self._backup_buffers, self._params)
        xm.all_reduce(
            xm.REDUCE_SUM,
            self._params,
            scale=1.0 / self._world_size,
        )
        xm.wait_device_ops()
        d2c = torch.norm(torch.stack([torch.norm(p1 - p2) for p1, p2 in zip(self._params, self._backup_buffers)]))
        d2c = xm.all_reduce(xm.REDUCE_SUM, d2c, scale=1.0 / self._world_size)
        xm.wait_device_ops()
        torch_xla.sync()
        return d2c.item()  # type: ignore

    @torch.no_grad()
    def restore(self):
        torch._foreach_copy_(self._params, self._backup_buffers)

    @torch.no_grad()
    def sync_buffers(self):
        data: list[torch.Tensor] = []
        for buffer in self._module.buffers():
            if buffer.dtype in [torch.float16, torch.float32, torch.float64]:
                data.append(buffer)
            else:
                pass
        xm.all_reduce(xm.REDUCE_SUM, data, scale=1.0 / self._world_size)
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
    train_loader: MpDeviceLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    cfg: Config,
    max_lr: float,
) -> Tuple[float, float]:

    model.train()
    total_loss_tpu = torch.tensor(0.0, device=torch_xla.device(), requires_grad=False)
    num_samples = torch.tensor(0.0, device=torch_xla.device(), requires_grad=False)
    gamma = 1.0
    start_time = time.time()

    torch_xla.sync(wait=True)

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

        batch_size = images.size(0)
        total_loss_tpu += loss * batch_size
        num_samples += batch_size

        torch_xla.sync()

    end_time = time.time()
    xm.wait_device_ops()
    data = torch.stack([total_loss_tpu, num_samples], dim=0)
    data = xm.all_reduce(xm.REDUCE_SUM, data)
    xm.wait_device_ops()
    torch_xla.sync()
    avg_loss = (data[0] / data[1]).item()

    return avg_loss, end_time - start_time


def eval_epoch(
    model: DecentDP,
    train_loader: MpDeviceLoader,
    test_loader: MpDeviceLoader,
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
    total_samples = torch.tensor(0.0, device=torch_xla.device(), requires_grad=False)
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
    data = torch.stack([total_loss, total_correct, total_samples])
    data = xm.all_reduce(xm.REDUCE_SUM, data)
    xm.wait_device_ops()
    torch_xla.sync()
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
        logger.info(f"world_size: {world_size}, rank: {rank}")
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
    xm.xla_rendezvous(b"setup_complete")

    if not xm.is_master_ordinal(local=True):
        xm.xla_rendezvous(b"data loading")


    # train_ds = CifarLoader(
    #     ds_name=cfg.dataset,
    #     train=True,
    #     batch_size=cfg.batch_size,
    #     rank=rank,
    #     num_replicas=world_size,
    #     base_seed=cfg.seed,
    # )
    # test_ds = CifarLoader(
    #     ds_name=cfg.dataset,
    #     train=False,
    #     batch_size=cfg.batch_size,
    #     rank=rank,
    #     num_replicas=world_size,
    # )

    train_dataset = tv.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=tv.transforms.Compose(
            [
                tv.transforms.RandomCrop(32, padding=4),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    )

    train_sampler = torch.utils.data.DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=cfg.seed,
                drop_last=True,
            )

    test_dataset = tv.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    )

    train_ds = MpDeviceLoader(
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.batch_size // world_size,
            sampler=train_sampler,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=16,
        ),
        device=torch_xla.device(),
    )
    test_ds = MpDeviceLoader(
        torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg.batch_size // world_size,
            sampler=torch.utils.data.DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=cfg.seed,
                drop_last=False,
            ),
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=16,
        ),
        device=torch_xla.device(),
    )


    if xm.is_master_ordinal(local=True):
        xm.xla_rendezvous(b"data loading")

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
        columns=["epoch", "train_loss", "test_loss", "test_acc", "d2c", "epoch_time", "total_train_time", "lr", "gamma"]
    )

    max_lr = 0.0

    for epoch in range(cfg.epochs):
        train_sampler.set_epoch(epoch)

        if (cfg.trainer.mix.name == "adaptive") and (epoch == cfg.trainer.mix.start_epoch):
            max_lr = scheduler.get_last_lr()[0]
            if xm.is_master_ordinal(local=True):
                logger.info(f"Adaptive mixing starts at epoch {epoch}, max_lr={max_lr:.5f}")

        train_loss, train_time = train_epoch(model, train_ds, criterion, optimizer, scheduler, epoch, cfg, max_lr)
        if (epoch + 1) % cfg.log.eval_interval == 0 or epoch == cfg.epochs - 1:
            test_loss, test_acc, d2c = eval_epoch(model, train_ds, test_ds, criterion, cfg)
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
                    gamma,
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
        os.makedirs(f"./logs/test", exist_ok=True)
        stats.to_csv(f"./logs/test/stats.csv", index=False)
        # wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    torch_xla.launch(main)
