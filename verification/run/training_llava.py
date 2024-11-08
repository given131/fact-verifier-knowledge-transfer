import logging
import os

import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from utils.distributed import (
    is_distributed,
    get_world_size,
)
from utils.log import (
    json_log,
    SEPERATE_LINE,
)
from utils.wandb import log_wandb
from verification.run.inference import evaluate_single_llava


def train_llava(model, processor, dataloaders, device, config):
    epochs = config.epochs
    tokenizer = processor.tokenizer

    optimizer = AdamW(model.parameters(), lr=config.lr, eps=config.lr)
    # cosine scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    accum_step = config.target_batch // (config.batch_size * get_world_size())
    running_loss = 0
    for epoch in range(epochs):
        model.train()
        logging.info("Epoch {}/{}".format(epoch, epochs))
        for i, batch in enumerate(tqdm(dataloaders["Train"])):
            # scheduler.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            image_sizes = batch["image_sizes"]
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            running_loss += loss.item()
            if i and i % accum_step == 0:
                json_log("optimizer.step", {
                    "step": i,
                    "loss": loss,
                })
                optimizer.step()
                optimizer.zero_grad()

            if i and i % 50:
                log_wandb({
                    "loss": running_loss / 100,
                })
                running_loss = 0
                torch.cuda.memory.empty_cache()

        optimizer.step()
        optimizer.zero_grad()
        running_loss = 0

        # eval with test
        try:
            model.eval()
            evaluate_single_llava(model, dataloaders["Val"], device, processor, config, epoch)
            evaluate_single_llava(model, dataloaders["Test"], device, processor, config, epoch, is_test=True)
        except:
            pass

        scheduler.step()
        model.module.save_pretrained(config.run_dir)

        save_path = config.run_dir / f"{epoch}.pt"
        torch.save({
            "optimizer": (optimizer.state_dict()),
            "scheduler": (scheduler.state_dict()),
            "epoch": epoch,
        }, save_path)


def train_per_epoch2(model, dataloader, device, optimizer, criterion, mini_batch_size, epoch):
    def _json_log(action, data):
        nonlocal batch_idx, epoch
        log_dict = {
            "step": step_count,
            "epoch": epoch,
            "batch_idx": batch_idx,
        }
        log_dict.update(data)
        json_log(action, log_dict)

        return

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

            stance_detect_layer = model.module.stance_detect_layer if is_distributed() else model.stance_detect_layer
            bias_params = stance_detect_layer.classifier_single.bias
            _json_log("backward", {
                "shape": {
                    "claim": batch["claim"].shape,
                    "claim_mask": batch["claim_mask"].shape,
                    "text_evidence": batch["text_evidence"].shape,
                    "text_evidence_mask": batch["text_evidence_mask"].shape,
                },
                "loss": loss,
                "params": bias_params,
                "grads": bias_params.grad,
                "output": output[:10],
                "id_list": batch['id_list'][:10],
                "length": len(batch['id_list']),
            })
            logging.info(SEPERATE_LINE)

            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()

                stance_detect_layer = model.module.stance_detect_layer if is_distributed() else model.stance_detect_layer
                bias_params = stance_detect_layer.classifier_single.bias
                _json_log("optimizer.step", {
                    "loss": loss,
                    "params": bias_params,
                    "id_list": batch['id_list'][:10],
                    "length": len(batch['id_list']),
                })
                step_count += 1
                logging.info(SEPERATE_LINE * 2)

    log_wandb({
        "epoch": epoch,
        "avg_loss": avg_loss,
        "live_loss": loss,
    })

    return avg_loss
