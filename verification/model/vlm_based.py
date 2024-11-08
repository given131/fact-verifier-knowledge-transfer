import logging

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    LlavaNextForConditionalGeneration,  # BitsAndBytesConfig,
)

from utils.distributed import is_distributed


def load_llava(config, device, load_path=""):
    model_name = config.model_name
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    llava = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        # quantization_config=quantization_config,
    ).to(device)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    llava = get_peft_model(llava, peft_config)
    if load_path:
        llava.load_adapter(load_path)

    llava.print_trainable_parameters()
    llava.config.gradient_checkpointing = True
    if is_distributed():
        llava = DDP(llava, device_ids=[device], find_unused_parameters=True)

    logging.info(f"model device: {llava.device}")

    return llava


def load_llava_eval(config, device, load_path=""):
    model_name = config.model_name
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    llava = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        # quantization_config=quantization_config,
    ).to(device)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    llava = get_peft_model(llava, peft_config)
    if load_path:
        llava.load_adapter(load_path, "default")

    if is_distributed():
        llava = DDP(llava, device_ids=[device], find_unused_parameters=True)

    logging.info(f"model device: {llava.device}")

    return llava



def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)