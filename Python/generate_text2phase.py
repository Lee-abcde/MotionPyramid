# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
# from utils.fixseed import fixseed
import time
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import argparse
import random
from argparse import ArgumentParser
from train_diff import add_base_options,get_cond_mode,add_data_options,add_model_options, add_diffusion_options,add_pretrained_VQ_options
import json
from option import TrainVQOptionParser, TestOptionParser
from dataset import create_dataset_from_args, create_mdm_dataset_from_args, create_txt2phase_dataset_from_args
from models import VQ as VQ_model
from models.VQ import get_phase_manifold
from models import phase_decoder as phase_decoder_model
from train_diff import train_args, create_model_and_diffusion
from torch.utils.data.dataloader import DataLoader
from copy import deepcopy
import Library.Utility as utility
from utils.npz_writer import write_motion2npz
from utils.training_loop import TrainLoop
import functools
from diffusion.resample import create_named_schedule_sampler
from torch.utils.data import Subset
import torch.nn.functional as F
from train_txt2phase import collate_fn

def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", type=str, required=True,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", type=str, required=True,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=32, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=1, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=1, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--random_sample", default=False, type=bool,
                       help="Shuffle dataset")


def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--inject_target_noise", action='store_true',
                        help="Inject the forward-diffused ground truth as the initial noise to verify low reconstruction MSE.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--manual_phase", default=False, type=bool,
                        help="Use the phase encoded by PAE or generate phase info base on manually selection")
    group.add_argument("--write_progressive_motion", default=False, type=bool,
                        help="write generation motion")
    group.add_argument("--input_set", default='train', type=str,
                       help="Use Test or Train set to check results")

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')

def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args
def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    add_pretrained_VQ_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    elif (args.action_file or args.action_name) and cond_mode != 'action':
        raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    return args

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    assert len(unexpected_keys) == 0
    # assert all([k.startswith('clip_model.') for k in missing_keys])

def clean_vq_state_dict(state_dict):
    for key in list(state_dict.keys()):
        if not (key.startswith("embedding") or key.startswith("vqs.")):
            state_dict.pop(key)
    return state_dict

class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        # self.rot2xyz = self.model.rot2xyz
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = self.model.cond_mode
        self.encode_text = self.model.encode_text
        self.nrootfeats = self.model.nrootfeats

    def forward(self, x, timesteps, y=None):
        cond_mode = self.model.cond_mode
        # assert cond_mode in ['text', 'action']
        # y_uncond = deepcopy(y)
        out = self.model(x, timesteps, y)
        y_uncond = {key: value.clone() if isinstance(value, torch.Tensor) else deepcopy(value) for key, value in
                    y.items()}
        y_uncond['uncond'] = True
        if torch.rand(1).item() < self.model.cond_mask_prob:
            y_uncond['text_embed'] = torch.zeros_like(y_uncond['text_embed'])
        if torch.rand(1).item() < self.model.cond_mask_prob:
            y_uncond['relative2start_rootpos'][:, :3, :] = 0
        if torch.rand(1).item() < self.model.cond_mask_prob:
            y_uncond['relative2start_rootpos'][:, 3:, :] = 0
        x[:, -10:, :] = 0
        x_uncond = x.clone()
        out_uncond = self.model(x_uncond, timesteps, y_uncond)
        return out_uncond + (y['scale'].view(-1, 1, 1) * (out - out_uncond))

def get_manual_manifold(angle_range, phase_index, window_second, VQ, batch_size, time_range):
    n_channel_phase = 1  # Assume each sample has only one phase channel; adjust if needed
    state = VQ.get_weight()
    state = state[0][phase_index].reshape(1, -1)

    angles = torch.linspace(0, angle_range * window_second, time_range, device="cuda")
    angles = angles.unsqueeze(0).unsqueeze(0).expand(batch_size, n_channel_phase, time_range)  # adjust dimensions
    manual_manifold = get_phase_manifold(state, angles)[0]
    return manual_manifold
def generate_y_axis_rotations(num_matrices: int):
    # Generate uniformly distributed angles from 0 to 360 degrees
    angles = torch.linspace(0, 2 * torch.pi, num_matrices)

    # Initialize the tensor that stores rotation matrices (16, 3, 3)
    rotation_matrices = torch.zeros((num_matrices, 3, 3))

    for i, theta in enumerate(angles):
        # Compute a rotation matrix around the y axis
        rotation_matrices[i] = torch.tensor([
            [torch.cos(theta), 0, torch.sin(theta)],
            [0, 1, 0],
            [-torch.sin(theta), 0, torch.cos(theta)],
        ])

    return rotation_matrices


def create_rotation_matrix_y(theta_degrees):
    # Convert angle to a tensor and then to radians
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

def project_manifold_to_codebook(manifold_output, VQ):
    """
    Project manifold_output [B, T, n_latent] to the geometrically exact nearest point on
    the VQ codebook ellipses, searching over 360 discretized angles.
    """
    vq_weights = VQ.get_weight().squeeze(0).to(manifold_output.device) # [class_num, 256]
    B, T, n_latent = manifold_output.shape # n_latent is 128
    class_num = vq_weights.shape[0]

    manifold_flat = manifold_output.reshape(B * T, n_latent) # [B*T, 128]

    # Extract A and B bases defined by VQ.py
    vq_weights = vq_weights.reshape(class_num, n_latent, 2)
    A_vec = vq_weights[:, :, 0] # [class_num, 128]
    B_vec = vq_weights[:, :, 1] # [class_num, 128]

    # Discretize theta to 360 points to find nearest projection point
    num_theta = 360
    theta = torch.linspace(0, 2 * torch.pi, num_theta, device=manifold_flat.device) # [num_theta]
    cos_theta = torch.cos(theta).unsqueeze(1) # [num_theta, 1]
    sin_theta = torch.sin(theta).unsqueeze(1) # [num_theta, 1]

    best_distances = torch.full((B * T,), float('inf'), device=manifold_flat.device)
    best_projected = torch.zeros_like(manifold_flat)

    for c in range(class_num):
        # M_c is the ellipse points for class c: A cos(theta) + B sin(theta)
        M_c = A_vec[c].unsqueeze(0) * cos_theta + B_vec[c].unsqueeze(0) * sin_theta # [num_theta, 128]
        
        dist = torch.cdist(manifold_flat, M_c) # [B*T, num_theta]
        min_dist_for_class, min_idx_for_class = torch.min(dist, dim=1) # [B*T]
        
        mask = min_dist_for_class < best_distances
        best_distances[mask] = min_dist_for_class[mask]
        best_projected[mask] = M_c[min_idx_for_class[mask]]

    projected_code = best_projected.reshape(B, T, n_latent) # [B, T, 128]
    return projected_code


def measure_text2phase(target, model_output, mask, use_manifold=False, VQ=None):
    import torch.nn.functional as F
    batch, feat, frame = target.shape
    assert target.shape == model_output.shape
    assert mask.shape == (batch, frame, 1)

    valid_mask = mask.squeeze(-1)

    if use_manifold:
        # [manifold(n_latent) | style(1) | traj(12)]
        n_latent = feat - 13
        manifold_output = model_output[:, :n_latent, :]
        style_output = model_output[:, n_latent:n_latent + 1, :]
        traj_output = model_output[:, n_latent + 1:, :]

        manifold_target = target[:, :n_latent, :]
        style_target = target[:, n_latent:n_latent + 1, :]
        traj_target = target[:, n_latent + 1:, :]
        
        # Project manifold before calculating MSE if VQ is provided
        if VQ is not None:
            # We must permute to [B, T, C] for the projection func, then permute back
            projected_manifold = project_manifold_to_codebook(manifold_output.permute(0, 2, 1), VQ).permute(0, 2, 1)
        else:
            projected_manifold = manifold_output

        # Manifold MSE on Projected Code
        manifold_mse = F.mse_loss(projected_manifold, manifold_target, reduction='none')
        manifold_mse = (manifold_mse * mask.permute(0, 2, 1)).sum() / (mask.sum() * n_latent + 1e-8)

        # Style & Traj MSE
        style_loss = F.mse_loss(style_output, style_target, reduction='none')
        style_loss = (style_loss * mask.permute(0, 2, 1)).sum() / (mask.sum() + 1e-8)

        traj_mse = F.mse_loss(traj_output, traj_target, reduction='none')
        traj_mse = (traj_mse * mask.permute(0, 2, 1)).sum() / (mask.sum() * 12 + 1e-8)

        terms = {
            "manifold_mse": manifold_mse,
            "style_mse": style_loss,
            "traj_mse": traj_mse,
        }
    else:
        # Legacy [one_hot(512) | angle(2) | style(1) | traj(12)]
        phase_dim = 512
        phase_logits = model_output[:, :phase_dim, :].permute(0, 2, 1)
        angle_output = model_output[:, phase_dim:phase_dim + 2, :]
        style_output = model_output[:, phase_dim + 2:phase_dim + 3, :]
        traj_output = model_output[:, phase_dim + 3:, :]

        phase_target = target[:, :phase_dim, :].permute(0, 2, 1)
        angle_target = target[:, phase_dim:phase_dim + 2, :]
        style_target = target[:, phase_dim + 2:phase_dim + 3, :]
        traj_target = target[:, phase_dim + 3:, :]

        # 1. Phaseclassification loss
        phase_class = torch.argmax(phase_target, dim=-1)
        ce_loss = F.cross_entropy(
            phase_logits.reshape(-1, 512),
            phase_class.reshape(-1),
            reduction='none'
        ).view(batch, frame)
        ce_loss = (ce_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        pred_class = torch.argmax(model_output[:, :512, :], dim=1)
        true_class = torch.argmax(target[:, :512, :], dim=1)

        # Compute accuracy
        correct = (pred_class == true_class).float()
        accuracy = (correct * valid_mask).sum() / valid_mask.sum()

        # 2. angle loss
        angle_mse = F.mse_loss(angle_output, angle_target, reduction='none')
        angle_mse = (angle_mse * mask.permute(0, 2, 1)).sum() / (mask.sum() * 2 + 1e-8)

        # 3. Style Codeloss
        style_loss = F.mse_loss(style_output, style_target, reduction='none')
        style_loss = (style_loss * mask.permute(0, 2, 1)).sum() / (mask.sum() + 1e-8)

        traj_mse = F.mse_loss(traj_output, traj_target, reduction='none')
        traj_mse = (traj_mse * mask.permute(0, 2, 1)).sum() / (mask.sum() * 12 + 1e-8)

        terms = {
            "phase_ce": ce_loss,
            "angle_mse": angle_mse,
            "style_mse": style_loss,
            "traj_mse": traj_mse,
            'accuracy': accuracy
        }
    return terms

def custom_interpolate_phase(phase):
    B, C, T = phase.shape
    assert T == 208, "Expected input length of 208 frames"

    result = []
    i = 0
    while i + 36 <= T:
        part_18_compress = F.interpolate(
            phase[:, :, i:i+18], size=3, mode='linear', align_corners=True
        )  # [B, C, 6]

        part_18_expand = F.interpolate(
            phase[:, :, i+18:i+36], size=33, mode='linear', align_corners=True
        )  # [B, C, 30]

        result.append(part_18_compress)
        result.append(part_18_expand)

        i += 36

    # Concatenate the remaining segment shorter than 36 directly, without interpolation
    if i < T:
        result.append(phase[:, :, i:])  # [B, C, <36]

    phase_interp = torch.cat(result, dim=2)
    return phase_interp

def save_generated_phase_style_editFrequencey(phase, class_num, m_length, VQ, args, text, name):
    # ---------- 1. Temporal interpolation ----------
    B, C, T = phase.shape
    phase_interp = F.interpolate(phase, scale_factor=2, mode='linear', align_corners=True)

    # ---------- 2. Split components ----------
    phase_logits = phase_interp[:, :class_num, :]                      # [B, class_num, 2T]
    angle_output = phase_interp[:, class_num:class_num + 2, :]        # [B, 2, 2T]
    angle_output = angle_output.permute((0, 2, 1)).unsqueeze(-1)      # [B, 2T, 2, 1]
    style_output = phase_interp[:, class_num + 2:, :]                 # [B, C', 2T]

    # ---------- 3. manifold computation ----------
    pred_class = torch.argmax(phase_logits, dim=1)                    # [B, 2T]
    state = VQ.get_weight().cpu()                                     # [1, codebook_size, dim]
    state_reshaped = state.squeeze(0)                                 # [codebook_size, dim]
    output = state_reshaped[pred_class]                               # [B, 2T, dim]
    output = output.reshape((output.shape[0], output.shape[1], -1, 2))# [B, 2T, D//2, 2]
    mainfold = (output @ angle_output).squeeze(-1).permute((0, 2, 1)) # [B, D//2, 2T]

    # ---------- 4. Save ----------
    output_dir = args.output_dir + args.input_set
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, name)
    torch.save({
        'manifold': mainfold,
        'stylecode': style_output,
        'length': m_length * 2,  # remember to update length
        'text': text
    }, output_path)

def save_generated_phase_style(phase, class_num, m_length, VQ, args, text, name):
    output_dir = args.output_dir+args.input_set
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, name)

    if getattr(args, 'use_manifold', False):
        n_latent = phase.shape[1] - 13
        manifold_output = phase[:, :n_latent, :].permute(0, 2, 1) # [B, T, n_latent]
        style_output = phase[:, n_latent:n_latent+1, :]
        trajectory_output = phase[:, n_latent+1:, :]

        # Project manifold outputs to nearest codebook ellipse
        projected_code = project_manifold_to_codebook(manifold_output, VQ)
        
        projected_mainfold = projected_code.permute(0, 2, 1) # [B, n_latent, T]

        torch.save({
            'manifold': projected_mainfold,
            'stylecode': style_output,
            'length': m_length,
            'relative_trajectory': trajectory_output,
            'text': text
        }, output_path)

    else:
        phase_logits = phase[:, :class_num, :]
        angle_output = phase[:, class_num:class_num + 2, :].permute((0, 2, 1)).unsqueeze(-1)
        style_output = phase[:, class_num + 2:class_num + 3, :]
        trajectory_output = phase[:, class_num + 3:, :]

        pred_class = torch.argmax(phase_logits, dim=1)
        state = VQ.get_weight().cpu()
        state_reshaped = state.squeeze(0)
        output = state_reshaped[pred_class]
        output = output.reshape((output.shape[0], output.shape[1], -1, 2))
        mainfold = (output @ angle_output).squeeze(-1).permute((0, 2, 1))

        torch.save({
            'manifold': mainfold,
            'stylecode': style_output,
            'length': m_length,
            'relative_trajectory': trajectory_output,
            'text': text
        }, output_path)


def process_motion2original_point(motion_list, data_std, data_mean, args):
    processed_motions = []
    for idx, motion in enumerate(motion_list):
        motion_t = motion.copy()

        # 1. Zero root position
        initial_root = motion_t[0, -12:-9]
        motion_t[:, -12:-9] -= initial_root

        # 2. Adjust root rotation to the target direction, such as +X
        initial_rot = motion_t[0, -9:].reshape(3, 3)
        target_rot = np.eye(3)  # Target direction is the identity matrix, assuming the original facing direction is +X
        inv_initial_rot = initial_rot.T @ target_rot  # Align initial rotation to the target direction

        for t in range(motion_t.shape[0]):
            # 2.1 Rotate root position
            current_pos = motion_t[t, -12:-9]
            rotated_pos = inv_initial_rot @ current_pos.reshape(3, 1)
            motion_t[t, -12:-9] = rotated_pos.flatten()

            # 2.2 Adjust root rotation
            current_rot = motion_t[t, -9:].reshape(3, 3)
            adjusted_rot = current_rot @ inv_initial_rot
            motion_t[t, -9:] = adjusted_rot.reshape(-1)

        processed_motions.append(motion_t)
        output_file = f"{args.output_dir}{args.input_set}/gt_motion_rot_{idx}.npz"
        write_motion2npz(motion_t, data_std, data_mean, motion_t.shape[0],
                         output_file, True)

    return np.concatenate(processed_motions, axis=0)
def main():
    option_parser = TrainVQOptionParser()
    args = generate_args()

    file_path = osp.join(args.pretrained_save, "args.txt")
    with open(file_path, "r") as f:
        args_dict = json.load(f)  # read as a dictionary
        vq_args = argparse.Namespace(**args_dict)  # convert the dictionary to Namespace
        vq_args = option_parser.post_process(vq_args)

    if args.random_sample:
        dynamic_seed = int(time.time())
    else:
        dynamic_seed = args.seed
    fixseed(dynamic_seed)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print('Loading dataset...')
    # create motion_datas to load VQ models (enable manually phase)
    repeated_times = 1
    gen_window = 2
    walkthedog_motion_datas = create_dataset_from_args(vq_args)
    args.window = gen_window
    # mdm_motion_datas = create_mdm_dataset_from_args(vq_args, args)
    mdm_motion_datas = create_txt2phase_dataset_from_args(vq_args, args, args.input_set)
    # data_frame_length = mdm_motion_datas[0].frames_per_window
    # max_frames = int(repeated_times * data_frame_length)
    # Build network model
    networks, VQ = VQ_model.create_model_from_args(vq_args, walkthedog_motion_datas)

    Save = args.pretrained_save
    ref_files = [f for f in os.listdir(Save) if f.endswith("Channels_VQ.pt")]
    ref_files.sort(key=lambda x: int(x.split('_')[0]))
    largest_epoch = ref_files[-1].split('_')[0]

    if vq_args.train_phase_decoder:
        phase_decoders = []
        for i, data in enumerate(walkthedog_motion_datas):
            phase_decoders.append(phase_decoder_model.create_model_from_args2(vq_args, data))
            state_file_name = f'{data.name}_{i}_{int(largest_epoch) - 1:04d}_Channels.pt'
            state_dict = torch.load(osp.join(Save, state_file_name), map_location='cpu')
            phase_decoders[-1].load_state_dict(state_dict)
    else:
        phase_decoders = []

    VQ_target_file = f'{largest_epoch}_{vq_args.phase_channels}Channels_VQ.pt'
    state_dict = torch.load(osp.join(Save, VQ_target_file), map_location='cpu')
    state_dict = clean_vq_state_dict(state_dict)
    VQ.load_state_dict(state_dict, strict=False)

    for i in range(len(phase_decoders)):
        phase_decoders[i] = utility.ToDevice(phase_decoders[i])
        phase_decoders[i].train()

    VQ = utility.ToDevice(VQ)
    VQ.eval()

    print("Creating model and diffusion...")
    motion_data = mdm_motion_datas[0]
    generator = torch.Generator().manual_seed(dynamic_seed)
    data_loader = DataLoader(
        motion_data,
        batch_size=args.batch_size,
        shuffle=True,  # keep shuffle=True but use a fixed generator
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        generator=generator,
        collate_fn=lambda batch: collate_fn(batch,
        num_embed_vq=motion_data.num_embed_vq
    ))
    model, diffusion = create_model_and_diffusion(args, vq_args, data_loader)
    model = model.cuda()

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler

    model.eval()  # disable random masking

    iterator = iter(data_loader)
    data_batch = next(iterator)
    max_frames = 208

    angle_xy = data_batch['angle_xy'].cuda()
    stylecode = data_batch['stylecode'].cuda()
    manifold_onehot = data_batch['manifold_onehot'].cuda()
    text_embed = data_batch['text_embed'].cuda()
    mask = data_batch['mask'].cuda()
    relative_traj = data_batch['relative_trajectory'].cuda()
    m_length = data_batch['lengths'].cuda()
    m_text = data_batch['text']
    for i, text in enumerate(m_text):
        print(f"[{i}] {text}")
    gt_motion = process_motion2original_point(data_batch['motion'], motion_data.data_std, motion_data.data_mean, args)
    write_motion2npz(gt_motion, motion_data.data_std, motion_data.data_mean, gt_motion.shape[0], args.output_dir + args.input_set + "/gt_motion_rot.npz",
                     True)
    # write_motion2npz(data_batch['motion'][7], motion_data.data_std, motion_data.data_mean, data_batch['motion'][7].shape[0], args.output_dir + "gt_motion.npz",
    #                  True)
    # manifold = manifold[:, :, :int(max_frames)]
    # manifold = torch.zeros_like(manifold)

    # code for test text2phase2motion diversity
    # text_embed = text_embed[0].unsqueeze(0).repeat(32, 1)
    # stylecode = stylecode[0].unsqueeze(0).repeat(32, 1, 1)
    # m_length = m_length[0].unsqueeze(0).repeat(32)
    model_kwargs = {
        'y': {
            'text_embed': text_embed,
            'lengths': m_length,
            'masks': mask,
        }
    }

    if args.use_manifold:
        manifold_cont = data_batch['manifold_continuous'].cuda()
        target_batch = torch.cat([manifold_cont, stylecode, relative_traj], dim=2).permute((0, 2, 1))
    else:
        target_batch = torch.cat([manifold_onehot, angle_xy, stylecode, relative_traj], dim=2).permute((0, 2, 1))

    # Initialize from noisy ground truth if flag is provided
    init_image = None
    skip_timesteps = 0
    if args.inject_target_noise:
        # Pass the clean target_batch; gaussian_diffusion will q_sample it internally
        init_image = target_batch
        skip_timesteps = 0  # Skip the top 10% most destructive noise steps so the initial condition survives

    all_motions = []
    all_text = []

    # t = torch.full((motion.shape[0],), args.diffusion_steps-1, device=torch.device("cuda"))
    # noise = torch.randn_like(motion)
    # init_image = diffusion.q_sample(motion, t, noise=noise)
    for rep_i in range(args.num_repetitions):
        print(f'### Sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device='cuda') * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        model_kwargs['y']['skip_step'] = list(range(1000, 1000))
        sample, motion_record, df_contact_label = sample_fn(
            model,
            (args.batch_size, model.nstylefeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        if args.unconstrained:
            all_text += ['unconstrained'] * args.num_samples
        else:
            pass
            
        sample_cuda = sample.cuda()
        if target_batch.shape == sample_cuda.shape:
            # Measure Without Projection
            loss_raw = measure_text2phase(target_batch, sample_cuda, mask, use_manifold=args.use_manifold)
            print("=== RAW MANIFOLD OUTPUT LOSS ===")
            print(loss_raw)
            
            # Measure With Projection
            loss_proj = measure_text2phase(target_batch, sample_cuda, mask, use_manifold=args.use_manifold, VQ=VQ)
            print("=== PROJECTED MANIFOLD OUTPUT LOSS ===")
            print(loss_proj)

        all_motions.append(sample.cpu())
        # all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

        # save mainfold and phase
        class_num = motion_data.num_embed_vq
        save_generated_phase_style(sample.cpu(), class_num, m_length, VQ, args, m_text, 'df_phase.pt')
        save_generated_phase_style(target_batch.cpu(), class_num, m_length, VQ, args, m_text, 'gt_phase.pt')
        # state = state[0][phase_index].reshape(1, -1)
        print(f"created {len(all_motions) * args.batch_size} samples")
if __name__ == "__main__":
    main()
