import os

import torch

from settings.directory import get_environment


def setup_and_get_device(config):
    env = get_environment()
    if env == "local":
        return torch.device("mps")

    if "LOCAL_RANK" in os.environ:  # distributed
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    elif config.device:
        device = torch.device(config.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    return device


def get_device_name():
    env = get_environment()
    if env == "local":
        return "mps"

    device_name = torch.cuda.get_device_name()
    if "V100" in device_name:
        return "v100"
    elif "A100" in device_name:
        return "a100"
    elif "H100" in device_name:
        return "h100"
    elif "8000" in device_name:
        return "rtx8000"
    elif "3090" in device_name:
        return "rtx3090"
    elif "4090" in device_name:
        return "rtx4090"
    elif "6000" in device_name:
        return "a6000"
    else:
        return device_name.replace(" ", "_")
