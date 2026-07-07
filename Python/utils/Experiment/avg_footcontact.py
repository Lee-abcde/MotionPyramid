import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation
def animate_skeleton(positions_reshaped, batch_idx=0, interval=50):
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

    # Add axis labels
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)

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
    """
    feature_dim, num_frames = array.shape
    num_batches = num_frames // batch_size

    trimmed = array[:, :num_batches * batch_size]  # [feature_dim, T*batch]
    reshaped = trimmed.reshape(feature_dim, batch_size, num_batches)

    return torch.from_numpy(reshaped).float()  # [D, T, B]

def compute_foot_contact_labels_percentile(X_style, percentile=65, ground_axis=1):
    """
    Mark foot-contact labels using the lowest percentile on the y axis

    Parameters:
        X_style: (D, B, T), joint sequence
        percentile: percentile threshold; for example, 20 treats the lowest 20% as contact
        ground_axis: which axis is vertical (default 1 = Y)

    Returns:
        contacts: dict
            {
                "left":  (B, T) left-foot contact labels (0 or 1)
                "right": (B, T) right-foot contact labels (0 or 1)
            }
    """
    D, B, T = X_style.shape
    num_joints = D // 3
    left_foot_idx = [5, 6]    # LeftFoot, LeftToeBase
    right_foot_idx = [11, 12] # RightFoot, RightToeBase

    contacts = {"left": np.zeros((B, T)), "right": np.zeros((B, T))}

    for b in range(B):
        joints = X_style[:, b, :].reshape(num_joints, 3, T)  # (J, 3, T)

        # average height of left and right feet (T,)
        left_heights = joints[left_foot_idx, ground_axis, :].mean(axis=0)
        right_heights = joints[right_foot_idx, ground_axis, :].mean(axis=0)

        # Compute threshold using the lowest percentile
        left_thresh = np.percentile(left_heights, percentile)
        right_thresh = np.percentile(right_heights, percentile)

        # Mark contact
        contacts["left"][b] = (left_heights <= left_thresh).astype(int)
        contacts["right"][b] = (right_heights <= right_thresh).astype(int)

    return contacts

def compute_foot_contact_labels(X_style, vel_threshold=0.15, height_threshold=.15, ground_axis=1):
    """
    Determine foot-contact labels from velocity and height

    Parameters:
        X_style: (D, B, T), joint sequence
        vel_threshold: velocity threshold; below this value is treated as contact
        height_threshold: height threshold, such as 0.05 m
        ground_axis: which axis is vertical (default 1 = Y)

    Returns:
        contacts: dict
            {
                "left":  (B, T-1) left-foot contact labels (0 or 1)
                "right": (B, T-1) right-foot contact labels (0 or 1)
            }
    """
    D, B, T = X_style.shape
    num_joints = D // 3
    # joint indices in the provided order
    left_foot_idx = [5, 6]    # LeftFoot, LeftToeBase
    right_foot_idx = [11, 12] # RightFoot, RightToeBase

    contacts = {"left": np.zeros((B, T - 1)), "right": np.zeros((B, T - 1))}

    for b in range(B):
        for t in range(T - 1):
            joints_curr = X_style[:, b, t].reshape(num_joints, 3)
            joints_next = X_style[:, b, t + 1].reshape(num_joints, 3)

            # left-foot and right-foot velocities
            left_vel = np.mean(np.linalg.norm(
                joints_next[left_foot_idx] - joints_curr[left_foot_idx],
                axis=1
            ))
            right_vel = np.mean(np.linalg.norm(
                joints_next[right_foot_idx] - joints_curr[right_foot_idx],
                axis=1
            ))

            # average height of left and right feet
            left_height = np.mean(joints_curr[left_foot_idx, ground_axis])
            right_height = np.mean(joints_curr[right_foot_idx, ground_axis])

            # Both velocity and height conditions must hold
            contacts["left"][b, t] = 1 if (left_vel < vel_threshold and left_height < height_threshold) else 0
            contacts["right"][b, t] = 1 if (right_vel < vel_threshold and right_height < height_threshold) else 0

    return contacts



def compute_contact_per_frame(X, phase_group=1, style_group=1):
    """
    Average input X (D, B, T) by phase/style groups and compute foot contact
    - Convert D to (num_joints, 3)
    - Translate each averaged motion along y so min(y)=0
    """
    D, B, T = X.shape
    num_joints = D // 3

    # reshape -> (num_joints, 3, B, T)
    X = X.reshape(num_joints, 3, B, T)

    # phase average
    new_T = T // phase_group
    X_phase = X[:, :, :, :new_T * phase_group].reshape(num_joints, 3, B, new_T, phase_group).mean(axis=-1)

    # style average
    new_B = B // style_group
    X_style = X_phase[:, :, :new_B * style_group, :].reshape(num_joints, 3, new_B, style_group, new_T).mean(axis=3)

    # (num_joints, 3, new_B, new_T) -> (num_joints, 3, new_B, new_T)
    # Y-axis translation: subtract the minimum y coordinate across all frames for each batch
    for b in range(new_B):
        min_y = X_style[:, 1, b, :].min()   # the y axis is at index 1
        X_style[:, 1, b, :] -= min_y

    # reshape back to the original input format (D, new_B, new_T)
    X_style_flat = X_style.reshape(D, new_B, new_T)
    # animate_skeleton(torch.from_numpy(X_style_flat), 0, 0.5)
    # Compute foot contact
    contacts = compute_foot_contact_labels_percentile(X_style_flat)
    return contacts, X_style_flat


def readnpz_and_compute_contact(npz_file, batch_size=64, phase_group=1, style_group=1):
    """
    Read the npz file and compute foot contact
    """
    data = np.load(npz_file)
    positions = data['Positions']  # [D, num_frames]
    positions_reshaped = reshape_to_dim_T_batch(positions, batch_size)  # [D, T, B]
    print(f"Positions reshaped: {positions_reshaped.shape}")


    positions_perm = positions_reshaped.numpy()

    contacts, _ = compute_contact_per_frame(positions_perm,
                                         phase_group=phase_group,
                                         style_group=style_group)
    return contacts


def plot_contact_heatmap(left_contacts, right_contacts, labels, title="Foot Contact Heatmap", save_path=None):
    import math
    num_actions = len(left_contacts)
    ncols = 2
    nrows = math.ceil(num_actions / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 6 * nrows))
    axes = axes.flatten()

    # Set font size
    label_fontsize = 28
    title_fontsize = 32
    tick_fontsize = 24

    for i in range(num_actions):
        ax = axes[i]
        B, T = left_contacts[i].shape
        heatmap_rgb = np.ones((B, T, 3))

        both_contact = (left_contacts[i] == 1) & (right_contacts[i] == 1)
        left_only = (left_contacts[i] == 1) & (~both_contact)
        right_only = (right_contacts[i] == 1) & (~both_contact)

        # soft color palette
        heatmap_rgb[left_only] = [0.99, 0.75, 0.53]  # pastel yellow
        heatmap_rgb[right_only] = [0.65, 0.81, 0.89]  # pastel blue
        heatmap_rgb[both_contact] =  [0.70, 0.87, 0.54]  # pastel green

        phase = np.linspace(-0.5, 0.5, T)
        ax.imshow(heatmap_rgb, aspect='auto', origin='lower', extent=[phase[0], phase[-1], 0, 1])
        ax.set_ylabel("Style Code", fontsize=label_fontsize)
        ax.set_xlabel("Phase", fontsize=label_fontsize)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([0, 1], fontsize=tick_fontsize)
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_xticklabels([-0.5, 0, 0.5], fontsize=tick_fontsize)
        ax.set_title(f"{labels[i]}", fontsize=title_fontsize, pad=30)

    for j in range(num_actions, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')

    plt.show()




# ------------------- Example usage -------------------

files = [
    "../../results/difftest116/generate/avg2_motion_jump.npz",
    "../../results/difftest116/generate/avg2_motion_run.npz",
    "../../results/difftest120/generate/avg2_motion_jump.npz",
    "../../results/difftest120/generate/avg2_motion_run.npz"
]
labels = ["Jump (Gram Matrix)", "Run (Gram Matrix)", "Jump (Avg Pooling)", "Run (Avg Pooling)"]

left_list = []
right_list = []

for f in files:
    contacts = readnpz_and_compute_contact(f, batch_size=64, phase_group=1, style_group=1)
    left_list.append(contacts["left"])
    right_list.append(contacts["right"])

# Plot heatmap
plot_contact_heatmap(left_list, right_list, labels, save_path="5-2 avg_footcontact_heatmaps.pdf")

