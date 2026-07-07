import torch
import torch.nn.functional as F
from torch import nn
import math

class ConvBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv1d(2 * in_ch, out_ch, 3, padding=1, padding_mode='reflect')
            self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        else:
            self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1, padding_mode='reflect')
            self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, padding_mode='reflect')
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):

        h = self.bn1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last dimension
        time_emb = time_emb[:, :, None]
        # Add time channel
        h = h + time_emb

        h = self.bn2(self.relu(self.conv2(h)))
        # Down or Upsample
        if hasattr(self, 'downsample'):
            h = self.downsample(h)
        elif hasattr(self, 'upsample'):
            h = self.upsample(h)
        return h


class PositionalEncoding1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeds = math.log(10000) / (half_dim - 1)
        embeds = torch.exp(torch.arange(half_dim, device=device) * -embeds)
        embeds = time[:, None] * embeds[None, :]
        embeds = torch.cat((embeds.sin(), embeds.cos()), dim=-1)
        return embeds


class Unet1d(nn.Module):
    """
    A simplified Unet architecture for 1D input.
    """

    def __init__(self,
                 signal_channels=512,
                 init_channels=128,
                 downsample_time=4,
                 time_emb_dim=512,
                 output_channel=512):
        super().__init__()
        self.signal_channels = signal_channels
        self.init_channels = init_channels
        self.downsample_time = downsample_time
        self.time_emb_dim = time_emb_dim
        self.output_channel = output_channel

        # Generate down and up channels
        down_channels = [self.init_channels * (2 ** i) for i in range(self.downsample_time)]
        up_channels = down_channels[::-1]

        # Time embedding
        self.time_mlp = nn.Sequential(
            PositionalEncoding1d(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv1d(self.signal_channels, down_channels[0], 3, padding=1, padding_mode='reflect')

        # Downsample
        self.downs = nn.ModuleList([ConvBlock1d(down_channels[i], down_channels[i + 1],
                                                self.time_emb_dim) for i in range(len(down_channels) - 1)])
        # Upsample
        self.ups = nn.ModuleList([ConvBlock1d(up_channels[i], up_channels[i + 1],
                                              self.time_emb_dim, up=True) for i in range(len(up_channels) - 1)])

        self.output = nn.Conv1d(up_channels[-1], self.output_channel, 1)

    def forward(self, x, timestep):

        # Embed time
        t = self.time_mlp(timestep)

        x = self.conv0(x)

        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        # Remove padding
        x = self.output(x)

        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model and transfer to device
    model = Unet1d().to(device)

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
