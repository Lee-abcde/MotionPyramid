import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def lengths_to_mask(lengths, max_len):
    steps = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return steps < lengths.unsqueeze(1)


def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.pose_embedding = nn.Linear(input_feats, latent_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        return self.pose_embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0]]
        return self.dropout(x)


class OutputProcessBert(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.final = nn.Linear(latent_dim, out_feats)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        output = self.final(hidden_states)
        return output.permute(1, 2, 0)


class Text2PhaseMoMask(nn.Module):
    def __init__(
        self,
        num_tokens,
        code_dim,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        text_dim=512,
        cond_drop_prob=0.1,
        aux_loss_weights=None,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.cond_drop_prob = cond_drop_prob
        self.mask_id = num_tokens
        self.pad_id = num_tokens + 1
        self.num_all_tokens = num_tokens + 2
        self.aux_loss_weights = aux_loss_weights or {
            "token": 1.0,
            "angle": 2.0,
            "style": 1.0,
            "traj": 10.0,
        }

        self.token_emb = nn.Embedding(self.num_all_tokens, code_dim)
        self.input_process = InputProcess(code_dim, latent_dim)
        self.position_enc = PositionalEncoding(latent_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cond_emb = nn.Linear(text_dim, latent_dim)

        self.token_head = OutputProcessBert(num_tokens, latent_dim)
        self.angle_head = OutputProcessBert(2, latent_dim)
        self.style_head = OutputProcessBert(1, latent_dim)
        self.traj_head = OutputProcessBert(12, latent_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def load_and_freeze_token_emb(self, codebook):
        with torch.no_grad():
            weight = torch.cat(
                [
                    codebook,
                    torch.zeros(2, codebook.shape[1], device=codebook.device, dtype=codebook.dtype),
                ],
                dim=0,
            )
            self.token_emb.weight = nn.Parameter(weight)
        self.token_emb.requires_grad_(False)

    def mask_cond(self, cond, force_mask=False):
        if force_mask:
            return torch.zeros_like(cond)
        if self.training and self.cond_drop_prob > 0.0:
            mask = torch.bernoulli(
                torch.full((cond.shape[0], 1), self.cond_drop_prob, device=cond.device)
            )
            return cond * (1.0 - mask)
        return cond

    def trans_forward(self, motion_ids, text_embed, padding_mask, force_mask=False):
        cond = self.mask_cond(text_embed, force_mask=force_mask)
        x = self.token_emb(motion_ids)
        x = self.input_process(x)
        cond = self.cond_emb(cond).unsqueeze(0)
        x = self.position_enc(x)
        xseq = torch.cat([cond, x], dim=0)
        src_padding_mask = torch.cat(
            [torch.zeros_like(padding_mask[:, :1]), padding_mask],
            dim=1,
        )
        hidden = self.seq_encoder(xseq, src_key_padding_mask=src_padding_mask)[1:]
        token_logits = self.token_head(hidden)
        angle = self.angle_head(hidden)
        angle = F.normalize(angle, dim=1)
        style = self.style_head(hidden)
        traj = self.traj_head(hidden)
        return {
            "token_logits": token_logits,
            "angle": angle,
            "style": style,
            "traj": traj,
            "hidden": hidden,
        }

    def _sample_masked_inputs(self, ids, valid_mask):
        batch_size, seq_len = ids.shape
        rand_time = torch.rand(batch_size, device=ids.device)
        rand_mask_probs = cosine_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((batch_size, seq_len), device=ids.device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        mask = mask & valid_mask

        x_ids = ids.clone()

        mask_rid = (torch.rand_like(x_ids.float()) < 0.1) & mask
        rand_id = torch.randint_like(x_ids, high=self.num_tokens)
        x_ids = torch.where(mask_rid, rand_id, x_ids)

        mask_mid = (torch.rand_like(x_ids.float()) < 0.88) & mask & (~mask_rid)
        x_ids = torch.where(mask_mid, torch.full_like(x_ids, self.mask_id), x_ids)
        labels = torch.where(mask, ids, torch.full_like(ids, -100))
        return x_ids, labels, mask

    def forward(self, ids, text_embed, lengths, angle_target, style_target, traj_target):
        batch_size, seq_len = ids.shape
        valid_mask = lengths_to_mask(lengths, seq_len)
        ids = torch.where(valid_mask, ids, torch.full_like(ids, self.pad_id))

        x_ids, labels, masked_positions = self._sample_masked_inputs(ids, valid_mask)
        outputs = self.trans_forward(x_ids, text_embed, padding_mask=~valid_mask)

        token_logits = outputs["token_logits"]
        angle = outputs["angle"]
        style = outputs["style"]
        traj = outputs["traj"]

        token_loss = F.cross_entropy(
            token_logits.permute(0, 2, 1).reshape(-1, self.num_tokens),
            labels.reshape(-1),
            ignore_index=-100,
        )

        mask_feat = valid_mask.unsqueeze(1).float()
        angle_loss = (F.mse_loss(angle, angle_target, reduction="none") * mask_feat).sum() / (
            mask_feat.sum() * 2 + 1e-8
        )
        style_loss = (F.mse_loss(style, style_target, reduction="none") * mask_feat).sum() / (
            mask_feat.sum() + 1e-8
        )
        traj_loss = (F.mse_loss(traj, traj_target, reduction="none") * mask_feat).sum() / (
            mask_feat.sum() * 12 + 1e-8
        )

        pred_ids = token_logits.argmax(dim=1)
        token_acc_mask = labels != -100
        if token_acc_mask.any():
            token_acc = (pred_ids[token_acc_mask] == ids[token_acc_mask]).float().mean()
        else:
            token_acc = torch.tensor(0.0, device=ids.device)

        total_loss = (
            self.aux_loss_weights["token"] * token_loss
            + self.aux_loss_weights["angle"] * angle_loss
            + self.aux_loss_weights["style"] * style_loss
            + self.aux_loss_weights["traj"] * traj_loss
        )
        return {
            "loss": total_loss,
            "token_loss": token_loss,
            "angle_mse": angle_loss,
            "style_mse": style_loss,
            "traj_mse": traj_loss,
            "token_acc": token_acc,
            "masked_ratio": masked_positions.float().mean(),
        }

    def forward_with_cond_scale(self, motion_ids, text_embed, padding_mask, cond_scale=1.0, force_mask=False):
        if force_mask:
            return self.trans_forward(motion_ids, text_embed, padding_mask, force_mask=True)
        outputs = self.trans_forward(motion_ids, text_embed, padding_mask, force_mask=False)
        if cond_scale == 1.0:
            return outputs
        aux = self.trans_forward(motion_ids, text_embed, padding_mask, force_mask=True)
        scaled = {}
        for key in ["token_logits", "angle", "style", "traj"]:
            scaled[key] = aux[key] + (outputs[key] - aux[key]) * cond_scale
        return scaled

    @torch.no_grad()
    def generate(
        self,
        text_embed,
        lengths,
        timesteps=10,
        cond_scale=1.0,
        temperature=1.0,
        topk_filter_thres=0.9,
        sample_mode="sample",
    ):
        device = text_embed.device
        seq_len = int(lengths.max().item())
        valid_mask = lengths_to_mask(lengths, seq_len)
        padding_mask = ~valid_mask

        ids = torch.where(
            padding_mask,
            torch.full((text_embed.shape[0], seq_len), self.pad_id, device=device, dtype=torch.long),
            torch.full((text_embed.shape[0], seq_len), self.mask_id, device=device, dtype=torch.long),
        )
        scores = torch.where(padding_mask, torch.full_like(ids.float(), 1e5), torch.zeros_like(ids.float()))
        remask_mask = valid_mask.clone()

        for timestep in torch.linspace(0, 1, timesteps, device=device):
            rand_mask_prob = cosine_schedule(timestep)
            current_mask = remask_mask & valid_mask
            if not current_mask.any():
                break

            ids = torch.where(current_mask, torch.full_like(ids, self.mask_id), ids)

            outputs = self.forward_with_cond_scale(ids, text_embed, padding_mask, cond_scale=cond_scale)
            logits = outputs["token_logits"].permute(0, 2, 1)

            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
            if sample_mode == "argmax":
                pred_ids = filtered_logits.argmax(dim=-1)
            elif sample_mode == "sample":
                probs = F.softmax(filtered_logits / max(temperature, 1e-6), dim=-1)
                pred_ids = torch.distributions.Categorical(probs).sample()
            else:
                raise ValueError(f"Unsupported sample_mode: {sample_mode}")
            ids = torch.where(current_mask, pred_ids, ids)

            probs_raw = logits.softmax(dim=-1)
            scores = probs_raw.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)
            scores = scores.masked_fill(~current_mask, 1e5)

            next_counts = torch.round(rand_mask_prob * lengths.float()).clamp(min=1).long()
            next_remask = torch.zeros_like(current_mask)
            for batch_idx in range(ids.shape[0]):
                active_idx = torch.nonzero(current_mask[batch_idx], as_tuple=False).squeeze(-1)
                if active_idx.numel() == 0:
                    continue
                keep_num = min(int(next_counts[batch_idx].item()), active_idx.numel())
                if keep_num <= 0:
                    continue
                active_scores = scores[batch_idx, active_idx]
                selected = active_idx[active_scores.argsort()[:keep_num]]
                next_remask[batch_idx, selected] = True
            remask_mask = next_remask

        final_outputs = self.trans_forward(ids, text_embed, padding_mask)
        ids = torch.where(padding_mask, torch.full_like(ids, -1), ids)
        return {
            "ids": ids,
            "angle": final_outputs["angle"],
            "style": final_outputs["style"],
            "traj": final_outputs["traj"],
        }

    def get_config(self):
        return {
            "num_tokens": self.num_tokens,
            "code_dim": self.code_dim,
            "latent_dim": self.latent_dim,
            "ff_size": self.latent_dim * 4,
            "num_layers": len(self.seq_encoder.layers),
            "num_heads": self.seq_encoder.layers[0].self_attn.num_heads,
            "dropout": self.seq_encoder.layers[0].dropout.p,
            "text_dim": self.text_dim,
            "cond_drop_prob": self.cond_drop_prob,
            "aux_loss_weights": self.aux_loss_weights,
        }


def top_k(logits, thres=0.9, dim=-1):
    k = max(1, int((1 - thres) * logits.shape[dim]))
    val, ind = torch.topk(logits, k, dim=dim)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(dim, ind, val)
    return probs
