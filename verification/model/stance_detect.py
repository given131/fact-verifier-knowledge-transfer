import torch
import torch.nn as nn
import torch.nn.functional as F

from verification.model.enums import EnumModelName


class BertMultiwayMatch(nn.Module):
    def __init__(self, class_num, config):
        super().__init__()

        if config.model_name == EnumModelName.CLIP_VIT_BASE_PATCH32.value:
            image_embed_dim = 768
            text_embed_dim = 512
        elif config.model_name in [
            EnumModelName.CLIP_VIT_LARGE_PATCH14.value,
            EnumModelName.CLIP_VIT_LARGE_PATCH14_336.value,
        ]:
            image_embed_dim = 1024
            text_embed_dim = 768
        else:
            raise ValueError(f"model_name: {config.model_name} not supported")

        self.hidden_size = text_embed_dim

        # modal specific
        self.image_multihead_attn = nn.MultiheadAttention(
            text_embed_dim,
            1,
            kdim=image_embed_dim,
            vdim=image_embed_dim,
            batch_first=True,
        )
        self.text_multihead_attn = nn.MultiheadAttention(
            text_embed_dim,
            1,
            kdim=text_embed_dim,
            vdim=text_embed_dim,
            batch_first=True,
        )

        # fuse layer
        self.linear_fuse_p = nn.Linear(text_embed_dim * 2, text_embed_dim)

        # classifier head
        self.classifier_both = nn.Linear(self.hidden_size * 2, class_num)
        self.classifier_single = nn.Linear(self.hidden_size, class_num)

    def forward(
            self,
            # embed
            claim_embed,
            text_evidences_embed,
            image_evidences_embed,
            # mask
            claim_mask,
            text_evidences_mask,
            image_evidences_mask,
            # info
            text_info_list,
            image_info_list,
    ):
        device = claim_embed.device
        single_stance = torch.tensor([], device=device)

        text_id_list = [item[0] for item in text_info_list]
        image_id_list = [item[0] for item in image_info_list]
        both_id_list = sorted(list(set(text_id_list) & set(image_id_list)))  ## keep_order
        text_only_id_list = sorted(list(set(text_id_list) - set(image_id_list)))
        image_only_id_list = sorted(list(set(image_id_list) - set(text_id_list)))

        text_stance = self.gen_stance(
            claim_embed, text_evidences_embed, claim_mask, text_evidences_mask, "text", text_info_list,
        )
        both_text_stance = torch.tensor([], device=device)
        for idx, original_idx in enumerate(text_id_list):
            if original_idx in both_id_list:
                both_text_stance = torch.cat([both_text_stance, text_stance[idx].unsqueeze(0)], 0)
            else:
                single_stance = torch.cat([single_stance, text_stance[idx].unsqueeze(0)], 0)

        image_stance = self.gen_stance(
            claim_embed, image_evidences_embed, claim_mask, image_evidences_mask, "image", image_info_list,
        )
        both_image_stance = torch.tensor([], device=device)
        for idx, original_idx in enumerate(image_id_list):
            if original_idx in both_id_list:
                both_image_stance = torch.cat([both_image_stance, image_stance[idx].unsqueeze(0)], 0)
            else:
                single_stance = torch.cat([single_stance, image_stance[idx].unsqueeze(0)], 0)

        if text_only_id_list or image_only_id_list:
            single_logits = self.classifier_single(single_stance)
        if both_id_list:
            both_stance = torch.cat([both_text_stance, both_image_stance], -1)
            both_logits = self.classifier_both(both_stance)

        batch_size = claim_embed.shape[0]
        logits = [None for _ in range(batch_size)]
        for idx, original_idx in enumerate(text_only_id_list):
            logits[original_idx] = single_logits[idx].unsqueeze(0)
        for idx, original_idx in enumerate(image_only_id_list):
            logits[original_idx] = single_logits[idx].unsqueeze(0)
        for idx, original_idx in enumerate(both_id_list):
            logits[original_idx] = both_logits[idx].unsqueeze(0)

        return torch.cat(logits, 0)

    def gen_stance(self, claim_embed, evidence_embed, claim_mask, evidence_mask, modality_type, evidence_info_list):
        if not evidence_info_list:  # only for image
            return torch.tensor([], device=claim_embed.device)
        # tensors of (claim, evidences)
        repeated_claim_embed = self._repeat_claim_in_evidence_shape(claim_embed, evidence_info_list)

        # attention
        attention_mask = ~evidence_mask.to(dtype=torch.bool)
        if modality_type == "text":
            mc_t, _ = self.text_multihead_attn(
                query=repeated_claim_embed,
                key=evidence_embed,
                value=evidence_embed,
                key_padding_mask=attention_mask
            )
            # logging.info(f"mc_t: {mc_t.shape}")
        elif modality_type == "image":
            mc_t, _ = self.image_multihead_attn(
                repeated_claim_embed, evidence_embed, evidence_embed, key_padding_mask=attention_mask
            )
        else:
            raise ValueError(f"modality_type: {modality_type} not supported")

        # MLP fuse
        new_mp_q = torch.cat([mc_t - repeated_claim_embed, mc_t * repeated_claim_embed], 2)
        fused = self.linear_fuse_p(new_mp_q)
        fused = F.leaky_relu(fused)

        # pooling
        claim_mask_for_evidence = self._repeat_mask_in_evidence_shape(claim_mask, evidence_info_list)
        stance_embed = self._gen_stance_embed(
            fused, claim_mask_for_evidence, evidence_info_list,
        )
        return stance_embed

    def _gen_stance_embed(self, fused, claim_mask, evidence_info_list):
        # generate mask
        attn_mask = claim_mask.to(torch.bool)
        attn_mask = ~attn_mask
        attn_mask = attn_mask.unsqueeze(-1)

        new_mp_ = fused.masked_fill_(attn_mask, -10000)
        p_max, _ = torch.max(new_mp_, 1)  # shape: (sequence_length, dimension_size) ->

        pooled = []
        for claim_idx, evidence_start_idx, evidence_length in evidence_info_list:
            out = p_max[evidence_start_idx:evidence_start_idx + evidence_length]
            new_p_max = torch.mean(out, 0, True)

            pooled.append(new_p_max)

        return torch.cat(pooled, 0)

    def _repeat_claim_in_evidence_shape(self, claim_embed, evidence_info_list):
        claim_out = []
        for evidence_info in evidence_info_list:
            claim_idx, evidence_start_idx, evidence_length = evidence_info
            claim_out.append(claim_embed[claim_idx].repeat(evidence_length, 1, 1))

        return torch.cat(claim_out, 0)

    def _repeat_mask_in_evidence_shape(self, mask, evidence_info_list):
        claim_out = []
        for evidence_info in evidence_info_list:
            claim_idx, evidence_start_idx, evidence_length = evidence_info
            claim_out.append(mask[claim_idx].repeat(evidence_length, 1))

        return torch.cat(claim_out, 0)
