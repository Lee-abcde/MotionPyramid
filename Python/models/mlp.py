import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from functools import partial
from copy import deepcopy
import random


class SiLU(torch.nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class PositionalEmbedding(nn.Module):
    __doc__ = r"""Computes a positional embedding of timesteps.
    Input:
        x: tensor of shape (N)
    Output:
        tensor of shape (N, dim)
    Args:
        dim (int): embedding dimension
        scale (float): linear scale to be applied to timesteps. Default: 1.0
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MotionMLP(nn.Module):
    def __init__(
            self,
            frame_size=512,
            hidden_size=1024,
            time_emb_size=512,
            layer_num=10,
            norm_type='layer_norm',
            act_type='SiLU'
    ):
        super().__init__()

        self.input_size = frame_size
        self.time_emb_size = time_emb_size
        layers = []
        for _ in range(layer_num):
            if act_type == 'ReLU':
                non_linear = torch.nn.ReLU()  ### v12 is ReLU
            elif act_type == 'SiLU':
                non_linear = SiLU()
            linear = nn.Linear(hidden_size + frame_size + time_emb_size, hidden_size)
            if norm_type == 'layer_norm':
                norm_layer = nn.LayerNorm(hidden_size)
            elif norm_type == 'group_norm':
                norm_layer = nn.GroupNorm(16, hidden_size)

            layers.append(norm_layer)
            layers.extend([non_linear, linear])

        self.net = nn.ModuleList(layers)
        self.fin = nn.Linear(frame_size + time_emb_size, hidden_size)
        self.fco = nn.Linear(hidden_size + frame_size + time_emb_size, frame_size)
        self.act = SiLU()

        self.time_mlp = torch.nn.Sequential(
            PositionalEmbedding(self.time_emb_size, 1.0),
            torch.nn.Linear(self.time_emb_size, self.time_emb_size),
            SiLU(),
            torch.nn.Linear(self.time_emb_size, self.time_emb_size),
        )

    def forward(self, x, timestep):
        x0 = x
        t = self.time_mlp(timestep).unsqueeze(1)
        time_embedding_broadcasted = t.expand(-1, x.size(1), -1)

        x = torch.cat([x, time_embedding_broadcasted], dim=-1)
        x = self.fin(x)

        for i, layer in enumerate(self.net):
            if i % 3 == 2:
                x = torch.cat([x, x0, time_embedding_broadcasted], dim=-1)
                x = layer(x)
            else:
                x = layer(x)

        x = torch.cat([x, x0, time_embedding_broadcasted], dim=-1)
        x = self.fco(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model and transfer to device
    model = MotionMLP().to(device)

    seq_len = 64
    batch = 64
    max_time = 1000  # maximum timestep
    feat_len = 512

    # Create input tensor and transfer to device
    signals = torch.randn(batch, feat_len, seq_len).to(device)
    timesteps = torch.randint(0, max_time, (batch,), device=device)  # random timestep

    # Use CUDA events to measure time
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # Start timing
    start_event.record()
    # Forward pass
    denoised_signals = model(signals, timesteps)
    print(denoised_signals.shape)
    # Stop timing
    end_event.record()
    # Wait for all CUDA operations to complete
    torch.cuda.synchronize()
    # Calculate time
    elapsed_time_ms = start_event.elapsed_time(end_event)  # Milliseconds
    print(f"Inference time: {elapsed_time_ms:.3f} ms")
