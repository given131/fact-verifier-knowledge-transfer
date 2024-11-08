import os

import wandb

from utils.distributed import is_main_process

ENABLE_WANDB = False


def enable_wandb():
    global ENABLE_WANDB
    ENABLE_WANDB = True

def login_wandb():
    global ENABLE_WANDB
    if ENABLE_WANDB:
        wandb_key = os.getenv("WANDB_KEY")
        wandb.login(key=wandb_key)


def set_wandb(config, run_name):
    if ENABLE_WANDB and is_main_process():
        if not config.project_name:
            raise ValueError("project_name is not set")
        wandb.init(
            project=f'{config.project_name}', config=dict(config), tags=[config.model_name, config.dataset_type]
        )
        wandb.run.name = run_name
        wandb.alert(title=f"Start {config.mode}", text=f"{config.mode}-{run_name}")


def log_wandb(data):
    if ENABLE_WANDB and is_main_process():
        wandb.log(data)


def finish_wandb():
    if ENABLE_WANDB and is_main_process():
        wandb.finish()