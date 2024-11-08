import torch
from torch import nn as nn

from verification.model.enums import EnumModelName
from verification.model.stance_detect import BertMultiwayMatch


class MultiModal(nn.Module):

    def __init__(
            self,
            clip,
            class_num,
            config,
    ):
        super(MultiModal, self).__init__()

        self.clipmodel = clip
        self.stance_detect_layer = BertMultiwayMatch(class_num, config)
        self.config = config

    def forward(
            self, claim, claim_mask, text_evidence, text_evidence_mask, image_evidence, text_info_list,
            image_info_list, device, **kwargs
    ):
        clip_inputs = {}

        # text
        clip_inputs['input_ids'] = torch.cat([claim, text_evidence], dim=0).to(device)

        claim_mask = claim_mask.to(device)
        text_evidence_mask = text_evidence_mask.to(device)
        if self.config.model_name in [
            EnumModelName.CLIP_VIT_LARGE_PATCH14_336.value,
            EnumModelName.CLIP_VIT_BASE_PATCH32.value,
            EnumModelName.CLIP_VIT_BASE_PATCH16.value,
            EnumModelName.CLIP_VIT_LARGE_PATCH14.value,
        ]:
            clip_inputs['attention_mask'] = torch.cat([claim_mask, text_evidence_mask]).to(device)

        # run clip
        if image_info_list:
            image_evidence = image_evidence.to(device)
            clip_inputs['pixel_values'] = image_evidence

            clip_output = self.clipmodel(**clip_inputs)
            text_embeds = clip_output.text_model_output.last_hidden_state  # (batch_size, sequence_length, hidden_size) 12,28,512
            image_evidences_embed = clip_output.vision_model_output.last_hidden_state  # (batch_size, sequence_length, hidden_size) 1,50,768
            image_evidences_embed = image_evidences_embed[:, 1:, :]  # remove CLS token
            image_evidence_mask = torch.ones(
                image_evidences_embed.shape[0],
                image_evidences_embed.shape[1],
                dtype=image_evidences_embed.dtype,
                device=image_evidences_embed.device,
            )
        else:
            text_model_output = self.clipmodel.text_model(**clip_inputs)
            image_evidences_embed = None
            image_evidence_mask = None
            text_embeds = text_model_output.last_hidden_state
            # remove cls: why?
            text_embeds = text_embeds[:, 1:]
            claim_mask = claim_mask[:, 1:]
            text_evidence_mask = text_evidence_mask[:, 1:]

        # verifier
        batch_size = claim.shape[0]
        claim_embed = text_embeds[:batch_size]
        text_evidences_embed = text_embeds[batch_size:]  # n,sequence size, 512

        output = self.stance_detect_layer(
            claim_embed,
            text_evidences_embed,
            image_evidences_embed,
            claim_mask,
            text_evidence_mask,
            image_evidence_mask,
            text_info_list,
            image_info_list,
        )

        return output
