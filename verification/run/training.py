import logging
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from utils.distributed import (
    get_world_size,
    is_main_process,
)
from utils.log import (
    SEPERATE_LINE,
)
from utils.wandb import log_wandb
from verification.data.util import get_loss_weights
from verification.run.inference import (
    evaluate_single,
    validate,
)


def train(
        model,
        dataloaders,
        train_label_statistic,
        device,
        config,
):
    epochs = config.epochs
    best_f1 = 0
    best_epoch = 0

    # loss
    weights = None
    if config.use_weight:
        weights = get_loss_weights(device, train_label_statistic, config.loss_weight_power)
        logging.info(f"normed_weights: {weights}")
    criterion = nn.CrossEntropyLoss(reduction=config.reduction, weight=weights)

    # optimizer
    optimizer = Adam(model.parameters(), lr=config.lr, eps=1e-8)

    # run epoch
    epoch = 0
    logging.info('Epoch {}/{}'.format(epoch, epochs))

    for epoch in range(0, epochs):
        logging.info('-' * 50)
        logging.info('Epoch {}/{}'.format(epoch + 1, epochs))

        train_per_epoch(model, dataloaders["Train"], device, optimizer, criterion, config.batch_size, epoch)
        cur_f1, best_f1, best_epoch, is_early_stop = validate(
            model,
            dataloaders["Val"],
            device,
            best_f1,
            config,
            epoch,
            best_epoch,
        )

    # test
    if is_main_process():
        best_ckpt_path = os.path.join(config.run_dir, "best.pt")  # load best model
        logging.info(f"load best model from {best_ckpt_path}")
        best_ckpt = torch.load(best_ckpt_path)
        model.load_state_dict(best_ckpt, strict=False)
        evaluate_single(model, dataloaders["Test"], device, config)


def train_per_epoch(model, dataloader, device, optimizer, criterion, mini_batch_size, epoch):
    total_loss = 0
    step_count = 0
    model.train()

    accum_iter = 2048 / (mini_batch_size * get_world_size())
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch_idx, batch in enumerate(tepoch):
            labels = batch['label'].to(device)
            output = model(**batch, device=device)

            loss = criterion(output, labels)
            total_loss += loss
            loss = loss / 2048
            loss.backward()

            avg_loss = total_loss.item() / ((batch_idx + 1) * mini_batch_size)
            tepoch.set_postfix(loss=avg_loss)

            logging.info(SEPERATE_LINE)

            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()

                step_count += 1
                logging.info(SEPERATE_LINE * 2)

    log_wandb({
        "epoch": epoch,
        "avg_loss": avg_loss,
        "live_loss": loss,
    })

    return avg_loss
