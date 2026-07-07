import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
# Specify your file path
npz_file = "../../results/difftest124/generate/df10_motion.npz"
batch_size = 32
# Read the npz file
data = np.load(npz_file)

# Access each array
positions = data['Positions']
velocities = data['Velocities']
rotations = data['Rotations']

import numpy as np
import torch
import matplotlib.pyplot as plt
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



def plot_pairwise_distance_for_batch(positions_reshaped, batch_idx=100):
    """
    positions_reshaped: [D, B, T] (D is 3 * num_joints)
    batch_idx: select which batch
    """
    D, B, T = positions_reshaped.shape
    num_joints = D // 3  # each 3 dims are one joint xyz

    # Extract batch
    pos_batch = positions_reshaped[:, batch_idx, :]  # [D, T]
    pos_batch_numpy = pos_batch.numpy()
    pos_batch = pos_batch.reshape(num_joints, 3, T)  # [J, 3, T]

    # Compute pairwise average Euclidean distance
    dist_matrix = torch.zeros(T, T)
    for t1 in range(T):
        for t2 in range(T):
            joints1 = pos_batch[:, :, t1]  # [J, 3]
            joints2 = pos_batch[:, :, t2]  # [J, 3]
            dist_per_joint = torch.norm(joints1 - joints2, dim=1)  # [J]
            avg_dist = dist_per_joint.mean()
            dist_matrix[t1, t2] = avg_dist

    # debug
    dist_matrix_debug = dist_matrix.numpy()
    # Visualize
    plt.figure(figsize=(6, 6))
    plt.imshow(dist_matrix.numpy(), cmap='hot', origin='lower')
    plt.colorbar(label='Avg Joint Euclidean Distance')
    plt.xlabel('Frame t2')
    plt.ylabel('Frame t1')
    plt.title(f'Pairwise Avg Euclidean Distance Heatmap (Batch {batch_idx})')
    plt.show()

    return dist_matrix

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

positions_reshaped = reshape_to_dim_T_batch(positions, batch_size)
# velocities_reshaped = reshape_to_dim_T_batch(velocities, batch_size)
# rotations_reshaped = reshape_to_dim_T_batch(rotations, batch_size)

plot_pairwise_distance_for_batch(positions_reshaped, batch_idx=0)

# debug: Visualiza
# animate_skeleton(positions_reshaped, batch_idx=0, interval=100)

time_range = positions_reshaped.shape[2]

print("Positions reshaped:", positions_reshaped.shape)
from itertools import combinations


def phase_binned_mean_pairwise_distance(positions, num_bins=20):
    """
    Compute phase-binned maximum pairwise distance using true per-joint Euclidean distance

    positions: [D, T], each three dimensions represent one joint xyz
    angles: [T], phase corresponding to each frame
    num_bins: number of phase bins
    reduce_mode:
        "time_first"  -> compute pairwise distances over time for each joint, then average over joints
        "joint_first" -> average over joints first, then compute pairwise distances over time

    Returns:
        bin_score: average of maximum pairwise distances over all bins
        global_score: global maximum pairwise distance
    """
    D, T = positions.shape
    num_joints = D // 3  # each three dimensions represent one joint

    # Normalize phase to [0, 1]
    # phi = (angles % (2 * torch.pi)) / (2 * torch.pi)
    # bin_edges = torch.linspace(0, 1, num_bins + 1, device=positions.device)
    bin_max_list = []

    # Compute maximum pairwise distance inside each bin
    for i in range(num_bins):
        idx = torch.arange(i, T, step=num_bins, device=positions.device)
        if idx.numel() < 2:
            continue

        pos_bin = positions[:, idx]  # [D, len(idx)]
        num_frames_bin = pos_bin.shape[1]


        diffs = []
        for t1, t2 in combinations(range(num_frames_bin), 2):
            joints1 = pos_bin[:, t1].reshape(num_joints, 3)
            joints2 = pos_bin[:, t2].reshape(num_joints, 3)
            # Euclidean distance for each joint
            dist_per_joint = torch.norm(joints1 - joints2, dim=1)
            mse = dist_per_joint.mean()
            diffs.append(mse)
        bin_max_list.append(torch.stack(diffs).mean())

    bin_score = torch.stack(bin_max_list).mean() if bin_max_list else torch.tensor(0.0)

    # global maximum pairwise distance
    num_frames = positions.shape[1]

    diffs = []
    for t1, t2 in combinations(range(num_frames), 2):
        joints1 = positions[:, t1].reshape(num_joints, 3)
        joints2 = positions[:, t2].reshape(num_joints, 3)
        dist_per_joint = torch.norm(joints1 - joints2, dim=1)
        mse = dist_per_joint.mean()
        diffs.append(mse)
    global_score = torch.stack(diffs).mean()

    D, T = positions.shape
    num_joints = D // 3

    # Compute pairwise distance matrix
    dist_matrix = torch.zeros(T, T)

    for t1 in range(T):
        for t2 in range(T):
            joints1 = positions[:, t1].reshape(num_joints, 3)
            joints2 = positions[:, t2].reshape(num_joints, 3)
            dist_per_joint = torch.norm(joints1 - joints2, dim=1)
            dist_matrix[t1, t2] = dist_per_joint.mean()

    return bin_score, global_score


pos_batch = positions_reshaped[:, 0, :]  # [D, T]
pos_batch_numpy = pos_batch.numpy()
bin_score, global_score = phase_binned_mean_pairwise_distance(pos_batch, num_bins=16)

print("Mean Phase-binned RMSE over all batches:", bin_score)
print("Mean Global RMSE over all batches:", global_score)