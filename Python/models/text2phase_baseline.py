import torch
import torch.nn as nn
import torch.nn.functional as F


class Text2PhaseTransformerBaseline(nn.Module):
    def __init__(
        self,
        text_dim=512,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        max_seq_len=208,
        manifold_dim=128,
        style_dim=1,
        traj_dim=12,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.manifold_dim = manifold_dim
        self.style_dim = style_dim
        self.traj_dim = traj_dim
        self.output_dim = manifold_dim + style_dim + traj_dim

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.length_proj = nn.Linear(1, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        self.input_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, self.output_dim)

        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, text_embed, lengths=None, target_length=208):
        if target_length > self.max_seq_len:
            raise ValueError(
                f"target_length={target_length} exceeds max_seq_len={self.max_seq_len}"
            )

        batch_size = text_embed.shape[0]
        text_token = self.text_proj(text_embed).unsqueeze(1).expand(-1, target_length, -1)

        if lengths is None:
            length_ratio = torch.ones(batch_size, 1, device=text_embed.device, dtype=text_embed.dtype)
        else:
            length_ratio = lengths.float().unsqueeze(-1) / float(target_length)
        length_token = self.length_proj(length_ratio).unsqueeze(1)

        x = text_token + length_token + self.pos_embedding[:, :target_length]
        x = self.input_norm(x)
        x = self.encoder(x)
        x = self.output_norm(x)
        x = self.output_head(x)
        return x.permute(0, 2, 1)

    def get_config(self):
        return {
            "text_dim": self.text_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "max_seq_len": self.max_seq_len,
            "manifold_dim": self.manifold_dim,
            "style_dim": self.style_dim,
            "traj_dim": self.traj_dim,
        }


def compute_masked_text2phase_manifold_losses(target, prediction, mask):
    batch, feat, frame = target.shape
    assert prediction.shape == target.shape
    assert mask.shape == (batch, frame, 1)

    n_manifold = feat - 13
    mask_feat = mask.permute(0, 2, 1)

    pred_manifold = prediction[:, :n_manifold, :]
    pred_style = prediction[:, n_manifold:n_manifold + 1, :]
    pred_traj = prediction[:, n_manifold + 1:, :]

    target_manifold = target[:, :n_manifold, :]
    target_style = target[:, n_manifold:n_manifold + 1, :]
    target_traj = target[:, n_manifold + 1:, :]

    manifold_mse = F.mse_loss(pred_manifold, target_manifold, reduction="none")
    manifold_mse = (manifold_mse * mask_feat).sum() / (mask_feat.sum() * n_manifold + 1e-8)

    style_mse = F.mse_loss(pred_style, target_style, reduction="none")
    style_mse = (style_mse * mask_feat).sum() / (mask_feat.sum() + 1e-8)

    traj_mse = F.mse_loss(pred_traj, target_traj, reduction="none")
    traj_mse = (traj_mse * mask_feat).sum() / (mask_feat.sum() * 12 + 1e-8)

    total_loss = manifold_mse + style_mse + traj_mse * 10.0
    return {
        "loss": total_loss,
        "manifold_mse": manifold_mse,
        "style_mse": style_mse,
        "traj_mse": traj_mse,
    }
