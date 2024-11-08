import os

import torch
from torch import distributed as dist


def is_distributed():
    return "LOCAL_RANK" in os.environ


def is_main_process():
    if not is_distributed():
        return True

    return int(os.environ.get("LOCAL_RANK")) == 0


def init_dist():
    if is_distributed():
        dist_url = "env://"  # default
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)


def finish_dist():
    if "LOCAL_RANK" in os.environ:
        dist.destroy_process_group()


def get_rank():
    if is_distributed():
        return int(os.environ['LOCAL_RANK'])
    return 0


def get_world_size():
    if is_distributed():
        return int(os.environ['WORLD_SIZE'])
    return 1
