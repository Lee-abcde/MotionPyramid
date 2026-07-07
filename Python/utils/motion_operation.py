import torch
def generate_y_axis_rotations(num_matrices: int):
    # Generate equally spaced angles from 0 to 2π radians
    angles = torch.linspace(0, 2 * torch.pi, num_matrices)

    # Initialize a tensor to store rotation matrices of shape (num_matrices, 3, 3)
    rotation_matrices = torch.zeros((num_matrices, 3, 3))

    for i, theta in enumerate(angles):
        # Compute the rotation matrix around the Y-axis for the current angle
        rotation_matrices[i] = torch.tensor([
            [torch.cos(theta), 0, torch.sin(theta)],
            [0, 1, 0],
            [-torch.sin(theta), 0, torch.cos(theta)],
        ])

    return rotation_matrices


def create_rotation_matrix_y(theta_degrees):
    # Convert the input angle from degrees to radians
    theta = torch.deg2rad(torch.tensor(theta_degrees, dtype=torch.float32))
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)


    # Build rotation matrix
    R = torch.zeros((3, 3), dtype=torch.float32)
    R[0, 0] = cos_theta
    R[0, 2] = sin_theta
    R[1, 1] = 1.0
    R[2, 0] = -sin_theta
    R[2, 2] = cos_theta
    return R

def trajecotry_signal_editor(gt_motion, init_pos, init_rot, relative_root, model_kwargs, diffusion):
    B, _, F = relative_root.shape

    # Generate interpolation coefficients from 0.5 to 2, one per batch
    scales = torch.linspace(0.1, 3.0, B, device=relative_root.device).view(B, 1, 1)  # (B, 1, 1)

    # Create a mask used to scale only the first three velocity channels
    velocity = relative_root[:, :3, :]  # (B, 3, F)
    rest = relative_root[:, 3:, :]  # (B, 9, F)

    # Scale velocity
    velocity_scaled = velocity * scales

    # Concatenate back to a full relative_root
    relative_root_scaled = torch.cat([velocity_scaled, rest], dim=1)

    # Assign to model_kwargs
    model_kwargs['y']['relative2start_rootpos'] = relative_root_scaled

    B, D = model_kwargs['y']['relative2start_rootpos'].shape[0], model_kwargs['y']['relative2start_rootpos'].shape[2]
    velocities = torch.linspace(0.02, 0.2, steps=B, device=model_kwargs['y']['relative2start_rootpos'].device)  # (B,)
    velocities_expanded = velocities.unsqueeze(1).expand(-1, D)  # (B, D)
    model_kwargs['y']['relative2start_rootpos'][:, -12, :] = 0  # set the 12th value from the end to 1
    model_kwargs['y']['relative2start_rootpos'][:, -11, :] = 0  # set the 11th value from the end to 0
    model_kwargs['y']['relative2start_rootpos'][:, -10, :] = velocities_expanded

    model_kwargs['y']['relative2start_rootpos'][:, -9:, :] = 0
    R = create_rotation_matrix_y(2).view(9)
    expanded_R = R.unsqueeze(0).unsqueeze(-1)
    model_kwargs['y']['relative2start_rootpos'][:, -9:, :] = expanded_R

    gt_motion[:, :, -12] = 0  # set the 12th value from the end to 1
    gt_motion[:, :, -11] = 0  # set the 11th value from the end to 0
    gt_motion[:, :, -10] = velocities_expanded

    model_kwargs['y']['relative2start_rootpos'][:, -9:, :] = 0

    R = create_rotation_matrix_y(0).view(9)
    expanded_R = R.unsqueeze(0).unsqueeze(0)
    gt_motion[:, :, -9:] = expanded_R
    gt_absolute_motion = diffusion.transfer2absolute_batch(gt_motion, init_pos, init_rot)
    gt_absolute_motion = gt_absolute_motion.reshape(-1, gt_absolute_motion.shape[2]).numpy()

    R = create_rotation_matrix_y(0).view(9)
    expanded_R = R.unsqueeze(0).unsqueeze(-1).expand(8, 9, 448)
    R1 = create_rotation_matrix_y(2).view(9)
    expanded_R1 = R1.unsqueeze(0).unsqueeze(-1).expand(8, 9, 448)
    R2 = create_rotation_matrix_y(-2).view(9)
    expanded_R2 = R2.unsqueeze(0).unsqueeze(-1).expand(8, 9, 384)
    model_kwargs['y']['relative2start_rootpos'][:, -9:, :] = torch.concat((expanded_R, expanded_R1, expanded_R2), dim=2)
