import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from utils.distributed import is_main_process

ENABLE_LOG = False


def enable_log():
    global ENABLE_LOG
    ENABLE_LOG = True


def is_log_enabled():
    return ENABLE_LOG


def init_logging(run_dir, run_name):
    for root_handler in logging.root.handlers[:]:
        logging.root.removeHandler(root_handler)
    logger = logging.getLogger('transformers.image_utils')
    logger.disabled = True

    if not is_main_process():
        return
    now = datetime.now()
    now = now.strftime("%m_%d_%H_%M_%S")
    logging_file_path = run_dir / f"{run_name}_{now}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logging_file_path)
        ]
    )
    logging.info(f"logging file name: {logging_file_path}")

    # disable transformers logger


def json_log(action, data):
    if not is_log_enabled():
        return

    # only log from the first process
    if not is_main_process():
        return

    # make the value of log_dict to be json serializable
    log_object = {
        "time": datetime.now().strftime("%m/%d-%H:%M:%S"),
        "action": action,
    }
    # serialize data
    for key, value in data.items():
        if isinstance(value, list):
            value = str(value)
        elif isinstance(value, torch.Tensor) or isinstance(value, torch.nn.Parameter):
            value = str(value.tolist())
        elif isinstance(value, dict):
            value = value.copy()
            for k, v in value.items():
                if isinstance(v, torch.Tensor) or isinstance(v, torch.nn.Parameter):
                    v = v.tolist()
                value[k] = str(v)
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, Path):
            value = str(value)

        data[key] = value

    log_object["data"] = data
    log_str = json.dumps(log_object, indent=4)
    logging.info(log_str)


SEPERATE_LINE = "==========================================="
