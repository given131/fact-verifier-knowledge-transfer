from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import CLIPModel

from utils.distributed import is_distributed
from verification.model.embedding_based import MultiModal
from verification.model.enums import EnumModelName
from verification.model.util import load_checkpoint


def setup_model(config, device):
    if config.model_name in [
        EnumModelName.CLIP_VIT_BASE_PATCH16.value,
        EnumModelName.CLIP_VIT_BASE_PATCH32.value,
        EnumModelName.CLIP_VIT_LARGE_PATCH14.value,
        EnumModelName.CLIP_VIT_LARGE_PATCH14_336.value,
    ]:
        clip_model = CLIPModel.from_pretrained(config.model_name)
    else:
        raise ValueError(f"Unexpected model name: {config.model_name}")

    clip_model.requires_grad_(False)  # freeze
    model = MultiModal(clip_model, config.class_num, config).to(device)

    # load ddp
    if is_distributed():
        model = DDP(model, device_ids=[device], find_unused_parameters=True)

    # load checkpoint
    if config.target_run_id:
        model = load_checkpoint(model, config.target_run_id, device)

    return model
