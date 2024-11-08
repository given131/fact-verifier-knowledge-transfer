import os
import re
from pathlib import Path

from utils.distributed import (
    get_world_size,
    is_main_process,
    is_distributed,
)


ENVIRONMENT_MAP = {
    "local": {
        "project": Path("/Users/given/projects/umd/universal-misinfo-detection"),
        "dataset": Path("/Users/given/projects/umd/dataset"),
        "run": Path("/Users/given/projects/umd/runs"),
    },
}


def get_environment():
    if os.path.exists("/Users/given"):
        return "local"
    else:
        raise ValueError("No valid path found")


def get_project_root_path():
    env = get_environment()
    return ENVIRONMENT_MAP[env]["project"]


def get_dataset_root_path():
    env = get_environment()
    return ENVIRONMENT_MAP[env]["dataset"]


def get_run_root_path():
    env = get_environment()
    return ENVIRONMENT_MAP[env]["run"]


def setup_dataset_dir(config):
    dataset_root_path = get_dataset_root_path()

    # dataset path
    if config.mode == "train":
        if "_" in config.dataset_type and "eg" not in config.dataset_type:
            config.train_file = dataset_root_path / "merged" / f"{config.dataset_type}.tsv"
        else:
            env = get_environment()
            config.train_file = dataset_root_path / config.dataset_type / "train.tsv"
            if env == "local":
                config.train_file = dataset_root_path / config.dataset_type / "val.tsv"
        config.val_file = dataset_root_path / config.val_file
        config.test_file = dataset_root_path / config.test_file


def setup_run_dir(config):
    run_root_path = get_run_root_path()
    run_dir_name = f"{config.cur_run_id:05d}"
    # verification
    if config.mode == "train":
        run_dir_name += f"-{config.dataset_type}"
    else:
        run_dir_name += f"-{config.mode}"

    config.run_dir = run_root_path / run_dir_name

    if is_main_process():
        os.makedirs(config.run_dir, exist_ok=True)


def get_run_id():
    run_root_path = get_run_root_path()
    prev_run_dirs = [x for x in os.listdir(run_root_path) if os.path.isdir(os.path.join(run_root_path, x))]
    prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1

    return cur_run_id


def get_run_name(config):
    from verification.model.enums import EnumModelName
    from settings.device import get_device_name
    env = get_environment()

    # verification
    if config.mode == "train":
        run_name = f"{config.dataset_type}-{config.reduction}"

        model_name = config.model_name
        if model_name == EnumModelName.CLIP_VIT_BASE_PATCH32:
            run_name += "-clip-32"
        elif model_name == EnumModelName.CLIP_VIT_LARGE_PATCH14:
            run_name += "-clip-large"
        elif model_name == EnumModelName.CLIP_VIT_LARGE_PATCH14_336:
            run_name += "-clip-large-336"

        if config.target_run_id:
            prev_run_name = get_prev_run_name(config.target_run_id)
            run_name += f"-target-{prev_run_name}"
        if config.seed != 42:
            run_name += f"-seed-{config.seed}"
        run_name += f"-{get_device_name()}"
        if is_distributed():
            run_name += f"-{get_world_size()}"
        run_name += f"-{env}-{config.cur_run_id}"

        return run_name

    elif config.mode == "infer":
        prev_run_name = get_prev_run_name(config.target_run_id)
        dataset_type = prev_run_name.split("-")[-1]
        config.dataset_type = dataset_type

        from verification.data.dataloader import is_two_class_dataset
        if is_two_class_dataset(dataset_type):
            config.class_num = 2

        return f"{dataset_type}-{config.target_run_id}-{env}-{config.cur_run_id}"

    # explanation generation
    elif config.mode == "eg-train":
        run_name = f"eg-train-{get_device_name()}-{get_world_size()}-{env}-{config.cur_run_id}"
        return run_name


def get_prev_run_name(run_id):
    run_root_path = get_run_root_path()
    dirs = os.listdir(run_root_path)
    run_id = f"{run_id:05d}"

    ret_value = []
    for dir in dirs:
        if run_id in dir:
            ret_value.append(dir)

    if len(ret_value) != 1:
        raise ValueError(f"run_id {run_id} is not found or ambiguous")

    return ret_value[0]
