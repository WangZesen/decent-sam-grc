import os
import torch
import torch.distributed as dist
import torch_xla
import torch_xla.runtime as xr


def run(pass_rank: int):
    rank = xr.global_ordinal()
    world_size = xr.world_size()
    print(f"Hello from rank {rank} of {world_size}, pass_rank={pass_rank}")
    dist.init_process_group("xla", "xla://")
    dist.barrier()
    print(f"Goodbye from rank {rank} of {world_size}, pass_rank={pass_rank}")
    dist.destroy_process_group()


if __name__ == "__main__":
    torch_xla.launch(run)
