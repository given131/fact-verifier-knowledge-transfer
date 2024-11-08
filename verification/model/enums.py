from enum import Enum


class EnumModelName(Enum):
    # embedding based
    CLIP_VIT_BASE_PATCH16 = "openai/clip-vit-base-patch16"
    CLIP_VIT_BASE_PATCH32 = "openai/clip-vit-base-patch32"
    CLIP_VIT_LARGE_PATCH14 = "openai/clip-vit-large-patch14"
    CLIP_VIT_LARGE_PATCH14_336 = "openai/clip-vit-large-patch14-336"
    # vlm
    LLAVA_NEXT = "llava-hf/llava-v1.6-mistral-7b-hf"
