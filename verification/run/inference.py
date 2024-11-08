import itertools
import json
import logging

import torch
from sklearn.metrics import f1_score
from torch import distributed as dist
from tqdm import tqdm

from utils.distributed import (
    is_main_process,
    is_distributed,
)
from utils.log import json_log
from utils.wandb import log_wandb
from verification.data.dataloader import is_two_class_dataset
from verification.model.util import save_model
from verification.run.evaluation import calc_metric, save_prediction


def infer(model, dataloader, class_num, device):
    logging.info("infer")
    y_true = []
    y_pred = []
    claim_ids_list = []
    model.eval()

    with tqdm(dataloader, unit="batch") as tepoch:
        for batch_idx, batch in enumerate(tepoch):
            labels = batch['label'].to(device)
            with torch.no_grad():
                output = model(**batch, device=device)
            _, preds = output.data[:, :class_num].max(1)
            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())
            claim_ids = batch['id_list']
            claim_ids_list.extend(claim_ids)
            # tepoch.set_postfix()

    return y_true, y_pred, claim_ids_list


def validate(model, dataloader, device, best_f1, config, epoch, best_epoch):
    if is_main_process():
        json_log("validate", {"epoch": epoch})
    true, pred, claim_ids = infer(model, dataloader, config.class_num, device)

    global_true = [None for _ in range(dist.get_world_size())]
    global_pred = [None for _ in range(dist.get_world_size())]
    global_claim_ids = [None for _ in range(dist.get_world_size())]

    dist.all_gather_object(global_true, true)
    dist.all_gather_object(global_pred, pred)
    dist.all_gather_object(global_claim_ids, claim_ids)

    y_true = list(itertools.chain(*global_true))
    y_pred = list(itertools.chain(*global_pred))

    if not is_main_process():
        return None, None, None, None

    # get score
    json_log(
        "validate",
        {"epoch": epoch, "len.y_true": len(y_true), "len.y_pred": len(y_pred)}
    )

    cur_f1, c_report = calc_metric(y_true, y_pred, config, epoch)

    log_data = {
        "epoch": epoch,
        "val/f1": cur_f1,
        "val/refuted_f1": c_report["refutes"]["f1-score"],
        "val/supported_f1": c_report["supports"]["f1-score"],
    }

    if config.class_num == 3:
        log_data["val/nei_f1"] = c_report["nei"]["f1-score"]

    log_wandb(log_data)

    # save model
    best_f1, best_epoch = save_model(best_f1, cur_f1, epoch, best_epoch, model, config)

    # early stop
    if config.early_stop and (epoch - best_epoch) >= config.early_stop:
        logging.info('early stop at epc {}'.format(epoch))
        is_early_stop = True
    else:
        is_early_stop = False

    return cur_f1, best_f1, best_epoch, is_early_stop


def evaluate_single(model, dataloader, device, config, epoch=50):
    logging.info("test")
    y_true, y_pred, claim_ids_list = infer(model, dataloader, config.class_num, device)

    # get score
    f1, c_report = calc_metric(y_true, y_pred, config)
    log_wandb({
        "epoch": epoch,
        "test/f1": f1,
        "test/refuted_f1": c_report["refutes"]["f1-score"],
        f"test/supported_f1": c_report["supports"]["f1-score"],
    })
    if config.class_num == 3:
        log_wandb({"test/nei_f1": c_report["nei"]["f1-score"]})

    # save results
    save_prediction(claim_ids_list, y_pred, config)

    return f1


def evaluate_multiple(model, dataloader, device, config, dataset_type):
    model.eval()

    if is_two_class_dataset(dataset_type):
        class_num = 2
    else:
        class_num = 3

    true, pred, claim_ids = infer(model, dataloader, class_num, device)
    if is_distributed():
        global_true = [None for _ in range(dist.get_world_size())]
        global_pred = [None for _ in range(dist.get_world_size())]
        global_claim_ids = [None for _ in range(dist.get_world_size())]

        dist.all_gather_object(global_true, true)
        dist.all_gather_object(global_pred, pred)
        dist.all_gather_object(global_claim_ids, claim_ids)

        y_true = list(itertools.chain(*global_true))
        y_pred = list(itertools.chain(*global_pred))

        if not is_main_process():
            return None, None, None, None
    else:
        y_true = true
        y_pred = pred

    # get score
    json_log(
        "validate",
        {"dataset_type": dataset_type, "len.y_true": len(y_true), "len.y_pred": len(y_pred)}
    )

    f1, c_report = calc_metric(y_true, y_pred, config)
    log_wandb({
        dataset_type: f1,
    })


def infer_llava(model, dataloader, processor, device):
    model.eval()

    with torch.no_grad():
        y_pred = []
        y_true = []
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            input_ids = batch['input_ids'].to(device)
            max_new_tokens = 10
            if is_distributed():
                out = model.module.generate(input_ids, max_new_tokens=max_new_tokens)
            else:
                out = model.generate(input_ids, max_new_tokens=max_new_tokens)
            prompt = processor.tokenizer.decode(input_ids[0])
            output = processor.tokenizer.decode(out[0])

            try:
                json_log("infer.log", {"prompt": prompt, "output": output, "len.output": len(output)})
            except:
                pass

            y_pred.append(output[len(prompt):])
            print(output[len(prompt):])
            y_true.append(batch['answers'][0])

    return y_true, y_pred


def evaluate_single_llava(model, dataloader, device, processor, config, epoch=50, is_test=False):
    json_log("eval", {"epoch": epoch})
    model.eval()

    y_true, y_pred = infer_llava(model, dataloader, processor, device)
    global_true = [None for _ in range(dist.get_world_size())]
    global_pred = [None for _ in range(dist.get_world_size())]

    dist.all_gather_object(global_true, y_true)
    dist.all_gather_object(global_pred, y_pred)

    y_true = list(itertools.chain(*global_true))
    y_pred = list(itertools.chain(*global_pred))

    if not is_main_process():
        return None, None, None, None

    json_log("eval", {"len.y_true": len(y_true), "len.y_pred": len(y_pred)})
    json_log("eval", {"y_true": y_true, "y_pred": y_pred})

    split = "test" if is_test else "val"
    with open(config.run_dir / f"{config.dataset_type}_{split}_epoch_{epoch}.json", "w") as f:
        json.dump({"y_true": y_true, "y_pred": y_pred}, f)

    log_wandb({"epoch": epoch})

    # calculate f1 score
    y_pred_mapped = []
    for y_p in y_pred:
        y_p = y_p.lower()
        if "enough information" in y_p:
            y_pred_mapped.append(2)
        elif "refute" in y_p:
            y_pred_mapped.append(0)
        elif "support" in y_p:
            y_pred_mapped.append(1)

    f1 = f1_score(y_true, y_pred_mapped, average='micro')
    with open(config.run_dir / f"{config.dataset_type}_{split}_epoch_{epoch}.f1", "w") as f:
        f.write(str(f1))
        log_wandb({f"{split}/f1": f1})


def evaluate_single_llava_test(model, dataloader, device, processor, config, dataset_type):
    json_log(f"log.{dataset_type}", {"test": True})
    model.eval()

    y_true, y_pred = infer_llava(model, dataloader, processor, device)
    global_true = [None for _ in range(dist.get_world_size())]
    global_pred = [None for _ in range(dist.get_world_size())]

    dist.all_gather_object(global_true, y_true)
    dist.all_gather_object(global_pred, y_pred)

    y_true = list(itertools.chain(*global_true))
    y_pred = list(itertools.chain(*global_pred))

    if not is_main_process():
        return None, None, None, None

    json_log("eval", {"len.y_true": len(y_true), "len.y_pred": len(y_pred)})
    json_log("eval", {"y_true": y_true, "y_pred": y_pred})

    y_pred_mapped = []
    for y_p in y_pred:
        y_p = y_p.lower()
        if "enough information" in y_p:
            y_pred_mapped.append(2)
        elif "refute" in y_p:
            y_pred_mapped.append(0)
        elif "support" in y_p:
            y_pred_mapped.append(1)
    f1 = f1_score(y_true, y_pred_mapped, average='micro')
    with open(config.run_dir / f"{config.dataset_type}_{dataset_type}.f1", "w") as f:
        f.write(str(f1))
        log_wandb({f"{dataset_type}/f1": f1})
