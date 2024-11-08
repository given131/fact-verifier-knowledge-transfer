import logging

import torch

from settings.directory import get_run_root_path, get_prev_run_name


def save_model(best_f1, cur_f1, epoch, best_epoch, model, config):
    # save current model
    if config.save_every_epoch:
        logging.info(f"save model at epoch {epoch}")
        torch.save(model.state_dict(), config.run_dir / f"{epoch}.pt")

    # save best model
    if cur_f1 > best_f1:
        logging.info(f"new best f1: {cur_f1}")
        best_f1 = cur_f1
        best_epoch = epoch
        torch.save(model.state_dict(), config.run_dir / "best.pt")
        logging.info('Model Saved!')
    return best_f1, best_epoch


def load_checkpoint(model, target_run_id, device):
    run_path = get_run_root_path()
    pre_run_name = get_prev_run_name(target_run_id)
    prev_check_point = run_path / pre_run_name / 'best.pt'
    weight = torch.load(prev_check_point, map_location=device)
    model.load_state_dict(weight, strict=False)

    return model
