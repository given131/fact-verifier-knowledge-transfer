import os

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig

from settings.device import setup_and_get_device
from settings.directory import (
    get_project_root_path,
    get_run_name,
    get_run_id,
    setup_run_dir, get_run_root_path,
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
from verification.data.dataloader import get_data_loaders_for_llava_eval
from verification.model.enums import EnumModelName
from verification.model.vlm_based import load_llava_eval
from verification.processor.load_processor import load_processor
from verification.run.inference import evaluate_single_llava_test


@hydra.main(
    config_path=f"{get_project_root_path()}/conf",
    config_name="infer",
)
def main(config: DictConfig):
    load_dotenv()
    init_dist()

    run_id = get_run_id()
    config.cur_run_id = run_id
    config.model_name = EnumModelName.LLAVA_NEXT.value
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

    torch.manual_seed(config.seed)
    device = setup_and_get_device(config)

    # create peft path
    peft_path = f"{get_run_root_path()}/{config.target_dir}/epoch_{config.target_epoch}"
    model = load_llava_eval(config, device, peft_path)
    processor = load_processor(config.model_name)
    dataloaders = get_data_loaders_for_llava_eval(config, processor)
    for dataset_type, dataloader in dataloaders.items():
        evaluate_single_llava_test(model, dataloader, device, processor, config, dataset_type)

    # set down
    finish_wandb()
    finish_dist()


if __name__ == "__main__":
    main()
