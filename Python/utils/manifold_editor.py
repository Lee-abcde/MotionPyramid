import torch
from models.VQ import get_phase_manifold
def get_manual_manifold(angle_range, phase_index, window_second, VQ, batch_size, time_range):
    n_channel_phase = 1  # Assume each sample has only one phase channel; adjust if needed
    state = VQ.get_weight()
    state = state[0][phase_index].reshape(1, -1)

    angles = torch.linspace(0, angle_range * window_second, time_range, device="cuda")
    angles = angles.unsqueeze(0).unsqueeze(0).expand(batch_size, n_channel_phase, time_range)  # adjust dimensions
    manual_manifold = get_phase_manifold(state, angles)[0]
    return manual_manifold, angles


def get_manual_manifold_autofrequency(base_angle_range, phase_index, window_second, VQ, batch_size, time_range,
                             min_factor=0.5, max_factor=2.0, device="cuda"):
    """
    Automatically generate an angle_range for each batch and compute the corresponding phase manifold

    Parameters:
    - base_angle_range: base angle range (float)
    - phase_index: selected phase index
    - window_second: window duration
    - VQ: vector-quantization model; must provide get_weight()
    - batch_size: batch size
    - time_range: number of temporal samples per sample
    - min_factor: minimum factor (default 0.5)
    - max_factor: maximum factor (default 2.0)
    - device: compute device
    """
    # Generate the angle_range for each batch
    angle_ranges = torch.linspace(min_factor * base_angle_range,
                                  max_factor * base_angle_range,
                                  batch_size, device=device).view(batch_size, 1)  # [B,1]
    ref_value = angle_ranges[8].clone()  # [1]

    # Expand to all batches
    angle_ranges = ref_value.repeat(batch_size, 1)  # [B,1]
    # codebook phase vector
    state = VQ.get_weight()[0][phase_index].reshape(1, -1)  # [1, D]

    # normalized temporal samples
    base_angles = torch.linspace(0, 1, time_range, device=device).view(1, -1)  # [1, T]

    # independent phase angles for each batch
    angles = base_angles * (angle_ranges * window_second)  # [B, T]
    # angles_debug = angles.cpu().numpy()
    angles = angles.unsqueeze(1)  # [B, 1, T]

    # Compute phase manifold
    manual_manifold = get_phase_manifold(state, angles)[0]
    return manual_manifold, angles