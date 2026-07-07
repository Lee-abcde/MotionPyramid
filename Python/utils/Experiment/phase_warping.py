import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import torch
from matplotlib.animation import FuncAnimation


def animate_skeleton(positions_reshaped, batch_idx=0, interval=0.05):
    """
    positions_reshaped: [D, B, T] (D = 3 * num_joints)
    batch_idx: which batch
    interval: interval between frames in milliseconds
    """
    D, B, T = positions_reshaped.shape
    num_joints = D // 3

    # [num_joints, 3, T]
    pos_batch = positions_reshaped[:, batch_idx, :].reshape(num_joints, 3, T)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set axis limits
    all_coords = pos_batch.numpy()
    xyz_min = all_coords.min()
    xyz_max = all_coords.max()
    ax.set_xlim([xyz_min, xyz_max])
    ax.set_ylim([xyz_min, xyz_max])
    ax.set_zlim([xyz_min, xyz_max])

    scat = ax.scatter([], [], [], c='b', s=30)

    def update(frame):
        joints = pos_batch[:, :, frame].numpy()  # [J, 3]
        scat._offsets3d = (joints[:, 0], joints[:, 1], joints[:, 2])
        ax.set_title(f"Frame {frame}")
        return scat,

    ani = FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
    plt.show()
    return ani

def reshape_to_dim_T_batch(array: np.ndarray, batch_size: int) -> torch.Tensor:
    """
    Convert an input numpy array of shape [feature_dim, num_frames]
    Convert to torch.Tensor with shape [feature_dim, T, batch]

    Args:
        array (np.ndarray): input with shape [feature_dim, num_frames]
        batch_size (int): temporal length T

    Returns:
        torch.Tensor: [feature_dim, T, batch]
    """
    feature_dim, num_frames = array.shape
    num_batches = num_frames // batch_size

    # Drop extra frames so the length is divisible
    trimmed = array[:, :num_batches * batch_size]  # [feature_dim, T*batch]

    # reshape to [feature_dim, T, batch] in column-major order
    reshaped = trimmed.reshape(feature_dim, batch_size, num_batches)

    # Convert to torch.Tensor
    return torch.from_numpy(reshaped).float()

def get_manual_manifold_autofrequency(base_angle_range, window_second, batch_size, time_range,
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

    # normalized temporal samples
    base_angles = torch.linspace(0, 1, time_range, device=device).view(1, -1)  # [1, T]

    # independent phase angles for each batch
    angles = base_angles * (angle_ranges * window_second)  # [B, T]
    angles = angles.unsqueeze(1)  # [B, 1, T]

    return angles, angle_ranges

def detect_foot_contact(positions, threshold=0.01):
    """
    positions: Tensor [81, B, T] (27 bones * 3 xyz)
    return: foot_contact [B, T], bool indicating whether each frame has foot contact
    """
    B, T = positions.shape[1], positions.shape[2]
    positions = positions.view(27, 3, B, T)  # [27, 3, B, T]

    # Take the left and right toe endpoints (ltoeSite=5, rtoeSite=10), z channel = 2
    ltoe_y = positions[5, 1]  # [B, T]
    rtoe_y = positions[10, 1] # [B, T]

    # Determine whether the foot contacts the ground
    l_contact = ltoe_y < threshold
    r_contact = rtoe_y < threshold

    # Count contact if either foot touches the ground
    foot_contact = l_contact | r_contact  # [B, T]

    return foot_contact, l_contact, r_contact

def angle_to_phase(angles):
    """
    angles: Tensor [B, 1, T] or any shape, in radians
    return: phase in [-0.5, 0.5)
    """
    # Convert to [0, 1)
    phase = (angles / (2 * torch.pi)) % 1.0
    # Map to [-0.5, 0.5)
    phase = phase - 0.5
    return phase

def phase_mean_range(phases, contact_mask_L, contact_mask_R):
    """
    phases: [T] tensor, values are in [-0.5, 0.5]
    contact_mask_L: [T] bool tensor, True means left-foot contact
    contact_mask_R: [T] bool tensor, True means right-foot contact

    return: dict containing mean +/- deviation for left and right feet
    """

    def circular_stats(phases_subset):
        if len(phases_subset) == 0:
            return 0.0, 0.0
        # Convert to angle
        theta = 2 * np.pi * phases_subset.cpu().numpy()
        z = np.exp(1j * theta)
        z_mean = np.mean(z)
        mean = np.angle(z_mean) / (2 * np.pi)  # [-0.5, 0.5]
        R = np.abs(z_mean)
        circ_std = np.sqrt(-2 * np.log(R)) / (2 * np.pi)
        return mean, circ_std

    mean_L, std_L = circular_stats(phases[contact_mask_L])
    mean_R, std_R = circular_stats(phases[contact_mask_R])

    return {
        'L': (mean_L, std_L),
        'R': (mean_R, std_R)
    }

if __name__ == '__main__':
    # Specify your file path
    npz_file = "../../results/difftest124/generate/df10_motion.npz"
    batch_size = 32
    motion_len = 208
    # Read the npz file
    data = np.load(npz_file)

    # Access each array
    positions = data['Positions']
    velocities = data['Velocities']
    rotations = data['Rotations']
    contact_label = data['ContactLabel']

    positions_reshaped = reshape_to_dim_T_batch(positions, batch_size)
    print(positions_reshaped.shape)
    # foot_contact, l_contact, r_contact = detect_foot_contact(positions_reshaped)

    foot_contact = reshape_to_dim_T_batch(contact_label, batch_size)
    # left-foot and right-foot probabilities
    r_contact_prob = foot_contact[0, :, :]
    l_contact_prob = foot_contact[1, :, :]


    # Convert to boolean values; True means ground contact
    l_contact = (l_contact_prob > 0.9)
    r_contact = (r_contact_prob > 0.9)

    angles, _ = get_manual_manifold_autofrequency(26 * torch.pi, 1, batch_size, motion_len)
    print("angles:", angles.shape)

    phases = angle_to_phase(angles).squeeze(1)
    print("phases range:", phases.min().item(), phases.max().item())

    # contact_phases = phases[r_contact]
    k = 15  # desired batch index
    phases_k = phases[k]              # [T]
    r_contact_k = l_contact[k]        # [T]

    contact_phases = phases_k[r_contact_k]
    plt.hist(contact_phases.cpu().numpy(), bins=40, range=(-0.5, 0.5), density=True)
    plt.xlabel("Phase")
    plt.ylabel("Density")
    plt.title("Phase distribution at foot contact")
    plt.show()

    # Example usage:
    # phases_k: [T], l_contact[k], r_contact[k] is a bool mask
    stats = phase_mean_range(phases_k, l_contact[k], r_contact[k])
    print(f"L foot: {stats['L'][0]:.3f}±{stats['L'][1]:.3f}")
    print(f"R foot: {stats['R'][0]:.3f}±{stats['R'][1]:.3f}")