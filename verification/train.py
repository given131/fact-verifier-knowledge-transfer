import os

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig
import numpy as np
import random

from settings.device import setup_and_get_device
from settings.directory import (
    get_project_root_path,
    get_run_name,
    get_run_id,
    setup_dataset_dir, setup_run_dir,
)
from utils.distributed import (
    is_main_process,
    init_dist,
    finish_dist,
)
from utils.log import (
    init_logging,
    enable_log,
    json_log,
)
from utils.wandb import (
    enable_wandb,
    set_wandb,
    finish_wandb,
    login_wandb,
)
from verification.data.dataloader import get_data_loaders_for_train, get_data_loaders_for_llava
from verification.model.enums import EnumModelName
from verification.model.load_model import setup_model
from verification.model.vlm_based import load_llava
from verification.processor.load_processor import load_processor
from verification.run.training import train
from verification.run.training_llava import train_llava


@hydra.main(
    config_path=f"{get_project_root_path()}/conf",
    config_name="moc_fak",
)
def main(config: DictConfig):
    load_dotenv()
    init_dist()

    run_id = get_run_id()
    config.cur_run_id = run_id
    setup_dataset_dir(config)
    setup_run_dir(config)
    run_name = get_run_name(config)

    enable_log()
    init_logging(config.run_dir, run_name)

    if is_main_process():
        json_log("start", {"pid": os.getpid(), "config": dict(config)})

    if not config.debug:
        enable_wandb()
        login_wandb()
        set_wandb(config, run_name)

    # set seed
    torch.manual_seed(config.seed)

    device = setup_and_get_device(config)
    if config.model_name != EnumModelName.LLAVA_NEXT.value:
        model = setup_model(config, device)
        dataloaders, train_label_statistic = get_data_loaders_for_train(config)
        train(model, dataloaders, train_label_statistic, device, config)
    else:
        model = load_llava(config, device)
        processor = load_processor(config.model_name)
        dataloaders = get_data_loaders_for_llava(config, processor)
        train_llava(model, processor, dataloaders, device, config)

    # set down
    finish_wandb()
    finish_dist()


if __name__ == "__main__":
    main()
