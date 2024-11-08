import logging
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from settings.directory import get_dataset_root_path, get_project_root_path


def open_image(image_path_list, image_dir):
    images = []
    for image_path in image_path_list:
        image = Image.open(os.path.join(image_dir, image_path))
        image = image.convert("RGB")
        images.append(image)
    return images


class EmbeddingDataset(Dataset):
    def __init__(self, claim_data, evidence_data, images, labels, processor, claim_ids, types, splits):
        self.claim_data = claim_data  # one text
        self.evidence_data = evidence_data  # list of text
        self.images = images  # list of images
        self.labels = labels  # int
        self.claim_ids = claim_ids
        self.processor = processor
        self.types = types
        self.splits = splits
        # TODO
        self.image_dir = get_dataset_root_path() / "images"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}

        # claim and text evidence
        claim_text_evidence_list = []
        claim = self.claim_data[idx]
        text_evidence_list = self.evidence_data[idx]
        claim_text_evidence_list.append(claim)
        claim_text_evidence_list.extend(text_evidence_list)
        type = self.types[idx]

        # image evidence
        image_list_str = self.images[idx]
        image_evidence_list = []
        if image_list_str:
            image_dir = self._get_image_dir(idx, type)
            images_name_list = image_list_str.split(";")
            image_evidence_list = open_image(images_name_list, image_dir)

        try:
            # pixel values shape: (num_img_list, 3, 224, 224)
            # thus, when getting pixel values, they will be batched twice
            model_inputs = self.processor(
                text=claim_text_evidence_list,
                images=image_evidence_list if image_evidence_list else None,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                # max_length=77,
            )
            if "attention_mask" not in model_inputs:
                model_inputs["attention_mask"] = self._create_mask(model_inputs["input_ids"])

        except Exception as e:
            logging.info(self.claim_ids[idx])
            logging.info(e)
            raise Exception(e)

        # label
        label = self.labels[idx]

        sample["claim_id"] = self.claim_ids[idx]
        sample["label"] = torch.tensor(label)
        sample.update(model_inputs)  # key: input_ids, attention_mask, pixel_values

        return sample

    def _get_image_dir(self, idx, type):
        image_dir = self.image_dir / type
        if type == "moc":
            original_split = self.splits[idx]
            if original_split == "train":
                original_split = "total"
            image_dir = image_dir / original_split
        return image_dir

    # for debug
    def get_idx_from_id(self, id):
        return self.claim_ids.index(id)

    def _create_mask(self, input_ids):
        indices = (input_ids == 1).nonzero(as_tuple=True)
        first_indices = [indices[1][indices[0] == i][0].item() for i in range(input_ids.size(0))]

        mask = torch.ones_like(input_ids)

        for i, idx in enumerate(first_indices):
            if idx == 0:
                idx = 1
            mask[i, idx + 1:] = 0

        return mask


class LlavaDataset(Dataset):
    def __init__(
            self,
            data_path,
            processor,
            class_num,
            is_test=False,
    ):
        self.data_path = data_path

        df = pd.read_csv(data_path, sep="\t").fillna("")
        self.data = df.to_dict(orient="records")
        self.image_dir = get_dataset_root_path() / "images"
        self.processor = processor
        self.class_num = class_num
        self.is_test = is_test

        # load template
        template_path = get_project_root_path() / "verification/data/mistral-template.jinja"
        chat_template = open(template_path).read()
        chat_template = chat_template.replace("    ", "").replace("\n", "")
        self.processor.tokenizer.chat_template = chat_template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        # text : input (claim, evidence), label
        claim, evidence, label, type = example["claim"], example["text_evidence"], example["label"], example["type"]
        message = self._create_message(claim, evidence, label)

        # image
        image_list_str = example["image_evidence"]
        if image_list_str:
            image_dir = self._get_image_dir(example)
            images_name_list = image_list_str.split(";")
            image_evidence_list = open_image(images_name_list, image_dir)
            image_evidence = self._merge_images(image_evidence_list)
        else:
            height = width = 336
            empty_image = self._create_empty_image(height, width)
            image_evidence = empty_image

        return {
            "id": example["id"],
            "text": message,
            "images": image_evidence,
            "answer": label,
        }

    def _create_message(self, claim, evidence, label):
        message = self._create_message_from_template(claim, evidence, label)
        token_length = len(self.processor.tokenizer(message)["input_ids"])
        if token_length > 2048:
            evidence = self._truncate_evidence(claim, evidence)
            message = self._create_message_from_template(claim, evidence, label)

        return message

    def _create_message_from_template(self, claim, evidence, label):
        if label == 0:
            label = "refutes"
        elif label == 1:
            label = "supports"
        else:
            label = "not enough information"

        user_template = USER_PROMPT_TEMPLATE if self.class_num == 3 else USER_PROMPT_TEMPLATE2
        user_message = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_template.format(claim=claim, text_evidences=evidence)
            },
        ]
        if not self.is_test:
            assistant_message = {
                "role": "assistant",
                "content": f"{label}"
            }
            user_message.append(assistant_message)

        message = self.processor.tokenizer.apply_chat_template(user_message, tokenize=False)

        return message

    def _truncate_evidence(self, claim, evidence):
        claim_length = len(self.processor.tokenizer(claim)["input_ids"])
        evidence_max_size = 2048 - claim_length - 171
        evidence_token = self.processor.tokenizer(
            evidence, max_length=evidence_max_size, truncation=True, padding="max_length"
        )["input_ids"]
        evidence = self.processor.tokenizer.decode(evidence_token)

        return evidence

    def _get_image_dir(self, example):
        original_split = example["split"]
        type = example["type"]
        image_dir = self.image_dir / type
        if type == "moc":
            if original_split == "train":
                original_split = "total"
            image_dir = image_dir / original_split
        return image_dir

    def _create_empty_image(self, height, width):
        return Image.new("RGB", (height, width), (0, 0, 0))

    def _merge_images(self, image_files):
        grid_size = (4, 4)  # 4x4 grid
        sub_image_size = (256, 256)  # Each sub-image is 56x56 pixels

        merged_image = Image.new('RGB', (1024, 1024))

        for index, image_file in enumerate(image_files):
            if index >= 16:  # Only process the first 16 images
                break
            img = image_file.resize(sub_image_size, Image.LANCZOS)
            # Calculate the position to paste the image
            x = (index % grid_size[0]) * sub_image_size[0]
            y = (index // grid_size[0]) * sub_image_size[1]
            # Paste the image into the merged_image
            merged_image.paste(img, (x, y))

        return merged_image


SYSTEM_PROMPT = """You are a skilled AI assistant dedicated to fact-checking. When presented with a claim and corresponding evidence in two modalities—image and text—your role is to determine if the evidence 'supports', 'refutes', or indicates 'not enough information' (NEI) to decide on the claim."""

USER_PROMPT_TEMPLATE = """ You are presented with the following information in two modalities—image and text:

Claim: {claim}
Text Evidence: {text_evidences}
Image Evidence: <image>

# Task:
- Determine the correct label that describes the relationship between the claim and the evidence.
- There might be no image evidence or text evidence provided.

# Classes
- `supports`
- `refutes`
- `not enough information`

# Requirements
- You should simply select one of the classes, based on the provided evidence.
"""


USER_PROMPT_TEMPLATE2 = """ You are presented with the following information in two modalities—image and text:

Claim: {claim}
Text Evidence: {text_evidences}
Image Evidence: <image>

# Task:
- Determine the correct label that describes the relationship between the claim and the evidence.
- There might be no image evidence or text evidence provided.

# Classes
- `supports`
- `refutes`

# Requirements
- You should simply select one of the classes, based on the provided evidence.
"""
