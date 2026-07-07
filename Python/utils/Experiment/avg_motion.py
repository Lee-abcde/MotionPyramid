import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

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

    trimmed = array[:, :num_batches * batch_size]  # [feature_dim, T*batch]
    reshaped = trimmed.reshape(feature_dim, batch_size, num_batches)

    # Convert to torch.Tensor
    return torch.from_numpy(reshaped).float()


def compute_avg_joint_distance(X_style, X_ref):
    """
    X_style: (D, B, T)
    X_ref:   (D, T)

    Returns:
        distances: (B, T) average joint distance for each frame of each batch
    """
    D, B, T = X_style.shape
    num_joints = D // 3
    distances = np.zeros((B, T))

    for b in range(B):
        for t in range(T):
            joints_pred = X_style[:, b, t].reshape(num_joints, 3)
            joints_ref = X_ref[:, t].reshape(num_joints, 3)
            distances[b, t] = np.mean(np.linalg.norm(joints_pred - joints_ref, axis=1))

    return distances


def compute_avg_mpjpe_per_frame(X, phase_group=1, style_group=1, joint_indices=None, ref_batch="middle"):
    """
    Average input X (D, B, T) by phase/style groups and compute per-frame average joint distance between each style batch and the reference batch
    """
    D, B, T = X.shape

    # Select joints
    if joint_indices is not None:
        mask = []
        for j in joint_indices:
            mask.extend([3 * j, 3 * j + 1, 3 * j + 2])
        X = X[mask, :, :]

    # phase average
    new_T = T // phase_group
    X_phase = X[:, :, :new_T * phase_group].reshape(X.shape[0], B, new_T, phase_group).mean(axis=-1)

    # style average
    new_B = B // style_group
    X_style = X_phase[:, :new_B * style_group, :].reshape(X.shape[0], new_B, style_group, new_T).mean(axis=2)

    # Select reference batch
    if isinstance(ref_batch, str) and ref_batch == "middle":
        ref_idx = new_B // 2
    else:
        ref_idx = int(ref_batch)
    X_ref = X_style[:, ref_idx, :]  # (D, T)

    # Compute per-frame average joint distance
    distances = compute_avg_joint_distance(X_style, X_ref)  # shape (B, T)
    return distances

def readnpz_and_compute_mpjpe(npz_file, batch_size=64, phase_group=1, style_group=1):
    # Read the npz file
    data = np.load(npz_file)
    positions = data['Positions']
    positions_reshaped = reshape_to_dim_T_batch(positions, batch_size)
    print(f"Positions reshaped: {positions_reshaped.shape}")
    # Compute MPJPE for each style batch
    distances = compute_avg_mpjpe_per_frame(positions_reshaped, phase_group=phase_group, style_group=style_group)
    return distances


def plot_avg_joint_distance_heatmap(distance_list, labels, title="Avg Joint Distance Heatmap"):
    """
    distance_list: list of np.array, each element is the (B, T) average joint distance for one action
    labels: list of str, action name
    """
    num_actions = len(distance_list)
    max_B = max([d.shape[0] for d in distance_list])
    max_T = max([d.shape[1] for d in distance_list])

    # Build a common-size matrix and fill with NaN
    heatmap = np.full((num_actions, max_B, max_T), np.nan)
    for i, dist in enumerate(distance_list):
        B, T = dist.shape
        heatmap[i, :B, :T] = dist

    # one subplot per action
    fig, axes = plt.subplots(num_actions, 1, figsize=(12, 3 * num_actions), squeeze=False)
    for i, ax in enumerate(axes[:, 0]):
        im = ax.imshow(heatmap[i], aspect='auto', origin='lower', cmap='hot',
                       extent=[0, max_T, -0., 1.0])  # x-axis 0 to T, y-axis -0.5 to 0.5

        # map y-axis to style code [-0.5, 0.5]
        B = heatmap[i].shape[0]
        yticks = np.linspace(0., 1., min(B, 10))  # show at most 10 labels
        ax.set_yticks(yticks)
        ax.set_ylabel("Style Code")
        ax.set_xlabel("Frame")
        ax.set_title(f"{labels[i]} - {title}")
        fig.colorbar(im, ax=ax, label="Avg Joint Distance")

    plt.tight_layout()
    plt.show()

# ------------------- Example usage -------------------

# File path
jump_file = "../../results/difftest116/generate/avg2_motion_jump.npz"
run_file  = "../../results/difftest116/generate/avg2_motion_run.npz"

distances_jump = readnpz_and_compute_mpjpe(jump_file, batch_size=64, phase_group=1, style_group=1)
distances_run = readnpz_and_compute_mpjpe(run_file, batch_size=64, phase_group=1, style_group=1)

plot_avg_joint_distance_heatmap([distances_jump, distances_run], labels=["Jump", "Run"])