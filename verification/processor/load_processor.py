from transformers import CLIPProcessor, LlavaNextProcessor

from verification.model.enums import EnumModelName


def load_processor(model_name):
    if model_name == EnumModelName.LLAVA_NEXT.value:
        processor = LlavaNextProcessor.from_pretrained(model_name)
    else:
        processor = CLIPProcessor.from_pretrained(model_name)

    return processor
