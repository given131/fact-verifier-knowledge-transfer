import numpy as np
from torch.utils.data import DataLoader
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from trl import DataCollatorForCompletionOnlyLM

from settings.directory import get_dataset_root_path, get_project_root_path
from verification.data.dataset import LlavaDataset

"""

dataset_path = get_dataset_root_path() / "moc" / "train.tsv"
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", padding_side="right")

ds = LlavaDataset(dataset_path, processor)

template_path = get_project_root_path() / "verification/data/mistral-template.jinja"
chat_template = open(template_path).read()
chat_template = chat_template.replace("    ", "").replace("\n", "")
processor.tokenizer.chat_template = chat_template

response_template = "(not enough evidence) [/INST]"


dataloader = DataLoader(ds, shuffle=False, drop_last=True)
for b in dataloader:
    pass
"""

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
print(model)