import logging
import os

import nltk
import pandas as pd
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from settings.directory import get_dataset_root_path, get_environment
from utils.distributed import is_distributed
from utils.log import json_log
from verification.data.collator import CLIPCollator, LlavaCollator
from verification.data.dataset import EmbeddingDataset, LlavaDataset
from verification.data.sampler import DistributedEvalSampler
from verification.processor.load_processor import load_processor


def get_data(data_file_path, class_num):
    logging.info(f"data_file_path: {data_file_path}")

    # cumulative data
    claim_ids = []
    claims = []
    text_evidences = []
    image_evidences = []
    labels = []
    types = []
    splits = []

    # for stats
    refuted = 0
    supported = 0
    nei = 0

    df_data = pd.read_csv(data_file_path, sep='\t')
    df_data = df_data.fillna('')
    for _, row in tqdm(df_data.iterrows()):
        id = row['id']
        claim = row['claim']
        image_evidence = row['image_evidence']
        text_evidence = row['text_evidence']
        label = row['label']
        type = row['type']
        split = row['split']

        # stats
        if label == 0:
            refuted += 1
        elif label == 1:
            supported += 1
        elif label == 2:  # nei
            if class_num == 2:
                continue
            nei += 1
        else:
            raise ValueError(f"Unknown label: {label}, id: {id}")

        claim_ids.append(id)
        claims.append(claim)
        text_evidence = nltk.sent_tokenize(text_evidence)
        if not text_evidence:
            text_evidence = ['']
        text_evidences.append(text_evidence)
        image_evidences.append(image_evidence)
        labels.append(label)
        types.append(type)
        splits.append(split)
    stats = [refuted, supported]
    if nei:
        stats.append(nei)

    logging.info(f"stats: supported: {supported} / refuted: {refuted} / nei: {nei}")

    return claim_ids, claims, text_evidences, image_evidences, labels, types, splits, stats


def get_data_loader_and_statistics(data_dir, processor, batch_size, num_workers, class_num, sampler=None):
    ids, claims, evidences, images, labels, types, splits, label_statistic = get_data(data_dir, class_num)

    # create dataset
    dataset = EmbeddingDataset(claims, evidences, images, labels, processor, ids, types, splits)

    # sampler and shuffle
    shuffle = True
    if sampler:
        sampler = sampler(dataset=dataset, shuffle=True)
        shuffle = False

    # create data loader
    clip_collate = CLIPCollator(processor)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=clip_collate,
        sampler=sampler,
        shuffle=shuffle,
    )

    return data_loader, label_statistic


def get_data_loaders_for_eval(config):
    processor = load_processor(config.model_name)

    dataset_types = ["fak", "ham", "hax", "mmh", "mrf", "moc", "pos", "pre", "pst", "pub", "tox", "ukr", ]

    eval_data_loaders = {}
    for dataset_type in dataset_types:
        try:
            path = get_dataset_path(dataset_type)
            env = get_environment()

            sampler = DistributedEvalSampler if env != 'local' else None

            test_data_loader, _ = get_data_loader_and_statistics(path, processor, 1, 1, config.class_num, sampler)
            eval_data_loaders[dataset_type] = test_data_loader
        except Exception as e:
            json_log("get_data_loaders_for_eval", {"dataset_type": dataset_type, "error": str(e)})

    return eval_data_loaders


def get_data_loaders_for_llava_eval(config, processor):
    dataloaders = {}

    collator = LlavaCollator(processor)

    dataset_types = [
        "fak",
        "ham", "hax", "mmh", "mrf", "moc", "pos", "pre", "pst", "pub", "tox", "ukr",
    ]
    for dataset_type in dataset_types:
        test_file_path = get_dataset_path(dataset_type)

        class_num = 2 if is_two_class_dataset(dataset_type) else 3
        test = LlavaDataset(test_file_path, processor, class_num, is_test=True)

        dataloaders[dataset_type] = DataLoader(
            test,
            batch_size=config.batch_size,
            collate_fn=collator,
            drop_last=False,
            num_workers=config.num_workers
        )

    return dataloaders


def is_two_class_dataset(dataset_type):
    return dataset_type in ["fak", "ham", "hax", "mmh", "mrf", "pos", "pre", "pst", "tox", "ukr"]


def get_dataset_path(dataset_type):
    return get_dataset_root_path() / 'eval' / f'{dataset_type}.tsv'


def get_sampler_class():
    if is_distributed():
        return DistributedSampler
    else:
        return None


def get_sampler(dataset):
    if is_distributed():
        return DistributedSampler(dataset, shuffle=True)
    else:
        return None


def get_data_loaders_for_train(config):
    processor = load_processor(config.model_name)
    dataloaders = {}

    sampler = get_sampler_class()
    num_workers = len(os.sched_getaffinity(0)) if config.num_workers == 'max' else config.num_workers

    val_data_loader, _ = get_data_loader_and_statistics(
        config.val_file, processor, 1, 1, config.class_num, sampler,
    )
    test_data_loader, _ = get_data_loader_and_statistics(
        config.test_file, processor, 1, 1, config.class_num, None,
    )
    train_data_loader, train_label_statistic = get_data_loader_and_statistics(
        config.train_file, processor, config.batch_size, num_workers, config.class_num, sampler
    )

    assert len(train_label_statistic) == config.class_num, \
        f"config.class_num({config.class_num}) is not equal to the number of classes({len(train_label_statistic)}"

    dataloaders['Val'] = val_data_loader
    dataloaders['Test'] = test_data_loader
    dataloaders['Train'] = train_data_loader

    return dataloaders, train_label_statistic


def get_data_loaders_for_llava(config, processor):
    dataloaders = {}

    collator = LlavaCollator(processor)

    val = LlavaDataset(config.val_file, processor, config.class_num, is_test=True)
    test = LlavaDataset(config.test_file, processor, config.class_num, is_test=True)
    train = LlavaDataset(config.train_file, processor, config.class_num, is_test=False)

    shuffle = False if is_distributed() else True
    dataloaders["Val"] = DataLoader(
        val,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        drop_last=False,
        sampler=get_sampler(val),
        num_workers=config.num_workers
    )
    dataloaders["Test"] = DataLoader(
        test,
        batch_size=config.batch_size,
        collate_fn=collator,
        drop_last=False,
        num_workers=config.num_workers
    )
    dataloaders["Train"] = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        drop_last=True,
        sampler=get_sampler(train),
        num_workers=config.num_workers
    )

    return dataloaders
