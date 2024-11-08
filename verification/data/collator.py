import numpy as np
import torch
from torch.nn import functional as F


class CLIPCollator:
    def __init__(self, processor):
        self.processor = processor
        self.attention_mask_token = 1

    def __call__(self, batch):
        # ['claim_id', 'label', 'input_ids', 'attention_mask', 'pixel_values'][]
        id_list = []
        claim_list = []
        claim_mask_list = []
        text_evidence_list = []
        text_evidence_mask_list = []
        image_evidence_list = []
        label_list = []
        text_info_list = []
        image_info_list = []

        # pad tokens
        pad_token_id = self.processor.tokenizer.pad_token_id
        max_input_len = max([item['input_ids'].shape[-1] for item in batch])

        text_prev = 0
        image_prev = 0
        for batch_idx, item in enumerate(batch):
            # id
            id_list.append(item['claim_id'])

            # text input
            padding_size = max_input_len - item['input_ids'].shape[-1]

            padded_input_ids = F.pad(item['input_ids'], (0, padding_size), value=pad_token_id, )
            claim_list.append(padded_input_ids[0:1, :])
            text_evidence_list.append(padded_input_ids[1:, :])

            if text_length := item['input_ids'].shape[0] - 1:
                text_info_list.append((batch_idx, text_prev, text_length))
                text_prev += text_length

            # image input
            if 'pixel_values' in item:
                image_evidence_list.append(item['pixel_values'])
                image_length = item['pixel_values'].shape[0]
                image_info_list.append((batch_idx, image_prev, image_length))
                image_prev += image_length

            # mask
            # TODO check the padding value (maybe zero)
            padded_attention_mask = F.pad(item['attention_mask'], (0, padding_size), value=0)
            claim_mask_list.append(padded_attention_mask[0:1, :])
            text_evidence_mask_list.append(padded_attention_mask[1:, :])

            # label
            label_list.append(item['label'])

        return {
            'id_list': id_list,
            'claim': torch.concat(claim_list, dim=0),
            'claim_mask': torch.concat(claim_mask_list, dim=0),
            'text_evidence': torch.concat(text_evidence_list) if text_evidence_list else None,
            'text_evidence_mask': torch.concat(text_evidence_mask_list, dim=0),
            'image_evidence': torch.cat(image_evidence_list) if image_evidence_list else None,
            'text_info_list': text_info_list,
            'image_info_list': image_info_list,
            'label': torch.stack(label_list)
        }


class LlavaCollator:
    def __init__(self, processor):
        self.processor = processor
        self.response_template = "provided evidence. [/INST]"

        self.response_token_ids = self.processor.tokenizer.encode(
            self.response_template, add_special_tokens=False
        )
        self.ignore_index = -100

    def __call__(self, batch):
        messages = [example["text"] for example in batch]
        images = [example["images"] for example in batch]
        image_sizes = [image.size for image in images]
        answers = [example["answer"] for example in batch]

        inputs = self.processor(
            text=messages,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            # max_length=2048,
        )

        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_index

        for i in range(len(input_ids)):
            response_token_ids_start_idx = None
            for idx in np.where(labels[i] == self.response_token_ids[0])[0]:
                # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                if self.response_token_ids == labels[i][idx: idx + len(self.response_token_ids)].tolist():
                    response_token_ids_start_idx = idx

            response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)
            labels[i, :response_token_ids_end_idx] = self.ignore_index

        return {
            "input_ids": input_ids,
            "pixel_values": inputs["pixel_values"],
            "image_sizes": image_sizes,
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
            "answers": answers,
        }
