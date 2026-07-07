import torch
import torch.nn.functional as F
from torch import nn
import math

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()

        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t, ):

        h = self.bn1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb

        h = self.bn2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class PositionalEncoding(nn.Module):
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


class Unet(nn.Module):
    """
    A simplified Unet architecture.
    """

    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = (16, 32, 64, 128)
        up_channels = (128, 64, 32, 16)
        time_emb_dim = 512
        output_channel = 1

        # Time embedding
        self.time_mlp = nn.Sequential(
            PositionalEncoding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([ConvBlock(down_channels[i], down_channels[i + 1],
                                              time_emb_dim) for i in range(len(down_channels) - 1)])
        # Upsample
        self.ups = nn.ModuleList([ConvBlock(up_channels[i], up_channels[i + 1],
                                            time_emb_dim, up=True) for i in range(len(up_channels) - 1)])

        self.output = nn.Conv2d(up_channels[-1], output_channel, 1)

    def forward(self, x, timestep):

        # Embedd time
        t = self.time_mlp(timestep)
        # Compute padding size
        padding_size = (64 - x.shape[3] % 64) % 64
        original_length = x.shape[3]  # Record original length

        # Pad the last dimension
        x = F.pad(x, (0, padding_size))  # Pad the last dimension (left=0, right=padding_size)

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
        # Remove padding by slicing
        x = self.output(x)
        if padding_size > 0:
            x = x[..., :original_length]  # Keep the original-length segment

        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model and move it to the device
    model = Unet().to(device)

    channel = 512
    seq_len = 61
    batch = 64

    # Create input tensor and move it to the device
    images = torch.randn(batch, 1, channel, seq_len).to(device)
    timesteps = torch.ones(batch, ).to(device)


    # Use CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # Start timing
    start_event.record()
    # Run forward pass
    denoised_images = model(images, timesteps)
    print(denoised_images.shape)
    # Stop timing
    end_event.record()
    # Wait for all CUDA operation complete
    torch.cuda.synchronize()
    # Compute time
    elapsed_time_ms = start_event.elapsed_time(end_event)  # milliseconds
    print(f"Inference time: {elapsed_time_ms:.3f} ms")
