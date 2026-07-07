# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
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
from dataset import create_dataset_from_args, create_mdm_dataset_from_args, CachedText2MotionDataset, text2motion_collate_fn
from models import VQ as VQ_model
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
from utils.diffusion_onnx import export_diffusion_to_onnx
from utils.motion_operation import *
from utils.manifold_editor import *

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
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--manual_phase", default=False, type=bool,
                        help="Use the phase encoded by PAE or generate phase info base on manually selection")
    group.add_argument("--write_progressive_motion", default=False, type=bool,
                        help="write generation motion")
    group.add_argument("--use_gt_phase", action='store_true',
                       help="Use ground-truth phase.")
    group.add_argument("--generated_phase_path", default='./results/text2phase4/generate/test/', type=str,
                        help="Use the phase encoded by PAE or generate phase info base on manually selection")
    group.add_argument("--save_onnx", default=False, type=bool,
                        help="save diffusion as onnx")


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


def resolve_phase_result_paths(generated_phase_path):
    if osp.isdir(generated_phase_path):
        gen_phase_path = osp.join(generated_phase_path, 'generated_phase_results.pt')
        gt_phase_path = osp.join(generated_phase_path, 'gt_phase_results.pt')
    else:
        gen_phase_path = generated_phase_path
        gt_phase_path = osp.join(osp.dirname(generated_phase_path), 'gt_phase_results.pt')

    if not osp.exists(gen_phase_path):
        raise FileNotFoundError(f"Generated phase results not found: {gen_phase_path}")
    if not osp.exists(gt_phase_path):
        raise FileNotFoundError(f"Ground-truth phase results not found: {gt_phase_path}")
    return gen_phase_path, gt_phase_path


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
def draw_pic(predictions, angles):
    import matplotlib.pyplot as plt
    import torch
    import numpy as np

    batch_size = angles.shape[0]
    # angle shape: (batch_size, 1, 208) or similar; take index 0 in the second dimension as the angle channel
    angles_ = angles[:, 0, :]  # shape (batch_size, 208)
    predictions_ = predictions  # shape (batch_size, 208, 2)

    predictions_tensor = torch.from_numpy(predictions_)

    # Collect all contact angles; first create two lists
    left_angles_list = []
    right_angles_list = []

    for b in range(batch_size):
        # Find contact indices
        left_touch_idx = torch.nonzero(predictions_tensor[b, :, 0] == 1).squeeze()
        right_touch_idx = torch.nonzero(predictions_tensor[b, :, 1] == 1).squeeze()

        # Take the corresponding angles; use row b here
        left_angles_list.append(angles_[b, left_touch_idx])
        right_angles_list.append(angles_[b, right_touch_idx])

    # Concatenate contact angles from all batches
    left_angles_all = torch.cat(left_angles_list)
    right_angles_all = torch.cat(right_angles_list)

    # Normalization function
    def normalize_angle(angle_tensor):
        two_pi = 2 * np.pi
        angle_mod = torch.fmod(angle_tensor, two_pi)
        angle_mod = torch.where(angle_mod < 0, angle_mod + two_pi, angle_mod)
        return angle_mod

    def map_to_minus_point5_to_point5(angle_tensor):
        two_pi = 2 * np.pi
        normalized = angle_tensor / two_pi
        mapped = normalized - 0.5
        return mapped

    left_angles_norm = normalize_angle(left_angles_all)
    left_angles_mapped = map_to_minus_point5_to_point5(left_angles_norm).cpu().numpy()

    right_angles_norm = normalize_angle(right_angles_all)
    right_angles_mapped = map_to_minus_point5_to_point5(right_angles_norm).cpu().numpy()

    # Compute interval percentages

    # left-foot interval [-0.5, 0.2]
    left_in_interval = (left_angles_mapped >= -0.4) & (left_angles_mapped <= 0.)
    left_percentage = np.sum(left_in_interval) / len(left_angles_mapped) * 100

    # right-foot interval [0.0, 0.5] ∪ [-0.5, -0.35]
    right_in_interval = ((right_angles_mapped >= 0.1) & (right_angles_mapped <= 0.5))
    right_percentage = np.sum(right_in_interval) / len(right_angles_mapped) * 100

    print(f'Left foot angles in [-0.4, 0.0]: {left_percentage:.2f}%')
    print(f'Right foot angles in [0.1, 0.5]: {right_percentage:.2f}%')

    plt.figure(figsize=(10, 5))
    plt.hist(left_angles_mapped, bins=30, alpha=0.6, label='Left Foot Contact')
    plt.hist(right_angles_mapped, bins=30, alpha=0.6, label='Right Foot Contact')
    plt.xlabel('Normalized Angle [-0.5, 0.5]')
    plt.ylabel('Frequency')
    plt.title('Angle Distribution at Foot Contact (All Batches)')
    plt.legend()
    plt.grid(True)
    plt.savefig('output_all_batches.png')

def remove_root_from_single_sequence(global_pos):
        """
        Remove root influence from a single denormalized motion sequence with a 27-joint skeleton

        :param motion_sequence: input motion sequence, shape is(frames, 417)
                              layout:[velocity(81) + position(81) + rotation(243) + root(12)]
        :return: motion sequence after root removal, shape is(frames, 405)
                 layout:[velocity(81) + root_relative_position(81) + root_relative_rotation(243)]
        """
        if not isinstance(global_pos, torch.Tensor):
            global_pos = torch.from_numpy(global_pos)  # convert to tensor(if it is not already)
        frames, _ = global_pos.shape

        # Split input
        velocities = global_pos[:, :81]  # first 84 velocity dimensions, unchanged
        positions = global_pos[:, 81:162]  # 27 * 3 = 81
        rotations = global_pos[:, 162:405]  # 27 * 9 = 243
        root_pos_rot = global_pos[:, 405:]  # last 12 dimensions

        # root pose
        root_positions = root_pos_rot[:, :3]  # (frames, 3)
        root_rotations = root_pos_rot[:, 3:].view(frames, 3, 3)  # (frames, 3, 3)

        # position transform
        positions = positions.view(frames, 27, 3)  # (frames, 27, 3)
        root_positions = root_positions.unsqueeze(1)  # (frames, 1, 3)

        positions_from_root = torch.matmul(
            positions, root_rotations.permute(0, 2, 1)
        ) + root_positions  # (frames, 27, 3)

        # rotation transform
        rotations = rotations.view(frames, 27, 3, 3)  # (frames, 27, 3, 3)
        rotations_from_root = torch.matmul(
            root_rotations.unsqueeze(1), rotations
        )  # (frames, 27, 3, 3)

        # flatten and concatenate
        adjusted_positions = positions_from_root.view(frames, -1)  # (frames, 81)
        adjusted_rotations = rotations_from_root.view(frames, -1)  # (frames, 243)

        updated_motion = torch.cat((velocities, adjusted_positions, adjusted_rotations), dim=-1)  # (frames, 408)
        # restored motion without root is right, don't do the unnormalization again in write_motionnpz
        # write_motion2npz(updated_motion, self.data_std, self.data_mean, updated_motion.shape[0],
        #                  "results/visualnpz/model_motion_gt.npz")
        return updated_motion.numpy()

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

    print('='*20)
    print('Loading dataset...')
    print('=' * 20)
    # create motion_datas to load VQ models (enable manually phase)
    repeated_times = 1
    gen_window = 10
    local_motion_datas = create_dataset_from_args(vq_args)
    args.window = gen_window
    mdm_motion_datas = create_mdm_dataset_from_args(vq_args, args)[0]
    data_frame_length = mdm_motion_datas.frames_per_window
    max_frames = int(repeated_times * data_frame_length)
    # Build network model
    networks, VQ = VQ_model.create_model_from_args(vq_args, local_motion_datas)

    Save = args.pretrained_save
    ref_files = [f for f in os.listdir(Save) if f.endswith("Channels_VQ.pt")]
    ref_files.sort(key=lambda x: int(x.split('_')[0]))
    largest_epoch = ref_files[-1].split('_')[0]

    if vq_args.train_phase_decoder:
        phase_decoders = []
        for i, data in enumerate(local_motion_datas):
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
    motion_data = mdm_motion_datas
    test_dataset = Subset(motion_data, motion_data.test_set_index)
    test_dataset.data_std, test_dataset.data_mean = motion_data.data_std, motion_data.data_mean
    generator = torch.Generator().manual_seed(dynamic_seed)
    data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # keep shuffle=True but use a fixed generator
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        generator=generator  # add a fixed-seed generator
    )
    model, diffusion = create_model_and_diffusion(args, vq_args, data_loader)
    model = model.cuda()

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler

    model.eval()  # disable random masking

    gen_phase_path, gt_phase_path = resolve_phase_result_paths(args.generated_phase_path)
    t2m_dataset = CachedText2MotionDataset('./Datasets/HumanML3DwithRoot_Text/preprocessed_data/test_cache.pt',
                                           gen_phase_path,
                                           gt_phase_path)
    t2m_data_loader = DataLoader(
        t2m_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
        generator=generator,
        collate_fn=lambda batch: text2motion_collate_fn(batch,
        num_embed_vq=vq_args.num_embed_vq
    ))

    # sample = dataset[0]
    # print(sample['text'])  # print the text of the first sample
    # print(sample['motion'].shape)  # print the motion shape
    # write_motion2npz(sample['motion'], motion_data.data_std, motion_data.data_mean, sample['m_length'],
    #                  args.output_dir + f'test{gen_window}_motion.npz', True, sample['contact_label'].cpu().numpy())

    final_motion_list = []
    final_gt_motion_list = []
    for batch in t2m_data_loader:
        # motion, manifold, init_pos, init_rot, foot_contact, _, stylecode = next(iter(data_loader))

        gt_absolute_motion = batch['motion']
        manifold = batch['manifold'].permute(0, 2, 1)
        stylecode = batch['stylecode'].permute(0, 2, 1)
        relative_trajectory = batch['relative_trajectory'].permute(0, 2, 1)
        first_frame = batch['motion'][:, 0, :]  # [32, 417]
        init_pos = first_frame[:, -12:-9]
        init_rot = first_frame[:, -9:].reshape(gt_absolute_motion.size(0), 3, 3)

        motion = gt_absolute_motion.permute(0, 2, 1).clone()
        gt_relative_trajectory = batch['gt_relative_trajectory'].permute(0, 2, 1)
        motion[:,-12:,:] = gt_relative_trajectory
        # debug
        # gt_absolute_motion_test = diffusion.transfer2absolute_batch(motion.permute(0, 2, 1), init_pos, init_rot)
        # diff = (gt_absolute_motion - gt_absolute_motion_test).detach().cpu().numpy()
        manifold, stylecode = manifold.repeat(1, 1, repeated_times), stylecode.repeat(1, 1, repeated_times)
        # gt_motion_np = gt_absolute_motion * motion_data.data_std + motion_data.data_mean
        # gt_motion_np = diffusion.integrate_root_motion(gt_motion_np)
        # gt_motion_np = gt_motion_np.cpu().numpy()
        # gt_motion_data = {
        #     f'motion_{i}': gt_motion_np[i]  # each has shape (208, 405)
        #     for i in range(gt_motion_np.shape[0])
        # }
        # np.savez_compressed('gt_motion.npz', **gt_motion_data)
        lengths = batch['lengths']
        gt_segments = []

        for i in range(args.batch_size):
            seg = gt_absolute_motion[i, :lengths[i], :]  # [length_i, D]
            final_gt_motion_list.append(seg)

            gt_segments.append(seg.cpu())

        gt_concatenated_motion = np.concatenate(gt_segments, axis=0)  # [total_frames, D]

        # Debug
        # frame_num = gt_concatenated_motion.shape[0]
        # write_motion2npz(
        #     gt_concatenated_motion,
        #     motion_data.data_std,
        #     motion_data.data_mean,
        #     frame_num,
        #     args.output_dir + f'gt{gen_window}_gtmotionclipped.npz',
        #     True,
        #     None
        # )
        # gt_absolute_motion = gt_absolute_motion.reshape(-1, gt_absolute_motion.shape[2]).numpy()

        manifold, stylecode, relative_trajectory, gt_relative_trajectory = map(lambda x: x.cuda(), (manifold, stylecode, relative_trajectory, gt_relative_trajectory))
        model_kwargs = {
            'y': {
                'text_embed': manifold,
                'stylecode': stylecode,
                'relative2start_rootpos': gt_relative_trajectory
            }
        }
        if args.manual_phase:
            motion_count_second = 1
            motion_len = int(motion_count_second * 208)
            manual_manifold, angles = get_manual_manifold(40 * torch.pi, 14, motion_count_second, VQ, args.batch_size, motion_len)  # 3 means 3s

            # (style 1)
            # motion_count_second = 2
            # motion_len = int(motion_count_second * 128)
            # manual_manifold = get_manual_manifold(2 * torch.pi, 2, motion_count_second, VQ, args.batch_size, motion_len)  # 3 means 3s

            # manual_manifold = get_manual_manifold(2 * torch.pi, 11, 3, VQ, args.batch_size, 60)  # 3 means 3s
            # manual_manifold1 = get_manual_manifold(3 * torch.pi, 117, 3, VQ, args.batch_size, 60)
            # manual_manifold2 = get_manual_manifold(3 * torch.pi, 14, 4, VQ, args.batch_size,
            #                                        88)
            # manual_manifold = torch.cat((manual_manifold, manual_manifold1, manual_manifold2), dim=2)
            # manual_manifold = torch.zeros_like(manual_manifold)

            # Change style for a sequence
            # interp = torch.linspace(0.2, 0.2, steps=motion_len, device='cuda')  # shape: (motion_len,)
            # manual_stylecode = interp.view(1, 1, -1).expand(args.batch_size, -1, -1)
            # different batch use different style code
            const_vals = torch.linspace(0.5, 0.5, steps=args.batch_size, device='cuda')
            manual_stylecode = const_vals.view(-1, 1, 1).expand(-1, 1, motion_len)
            model_kwargs = {
                'y': {
                    'text_embed': manual_manifold,  # random embedding replacing text
                    'stylecode': manual_stylecode
                }
            }  # [32, 610 (10 * 61)]
        # model_kwargs['y']['relative2start_rootpos'][:, -12, :] = 0  # set the 12th value from the end to 1
        # model_kwargs['y']['relative2start_rootpos'][:, -11, :] = 0  # set the 11th value from the end to 0
        # model_kwargs['y']['relative2start_rootpos'][:, -10, :] = 0.12
        # trajecotry_signal_editor(gt_motion, init_pos, init_rot, relative_root, model_kwargs, diffusion)
        # R = create_rotation_matrix_y(0).view(9)
        # expanded_R = R.unsqueeze(0).unsqueeze(-1)
        # model_kwargs['y']['relative2start_rootpos'][:, -9:, :] = expanded_R
        # For generation pipeline, we choose manifold as [batch, all_feature] to enable progressive generation
        phase_decoder = phase_decoders[0] if len(phase_decoders) > 0 else None
        # if phase_decoder is not None:
        #     decoder_input = manual_manifold if args.manual_phase else manifold
        #     decoder_input = decoder_input.permute(0, 2, 1)
        #     decoder_input = decoder_input.reshape(-1, decoder_input.shape[-1])  # [32 * 61, 10]
        #     style_input = manual_stylecode if args.manual_phase else stylecode
        #     style_input = style_input.permute(0, 2, 1)
        #     style_input = style_input.reshape(-1, style_input.shape[-1])  # [32 * 61, 10]
        #     avg_motion = phase_decoder(decoder_input, style_input).cpu().detach()
        #     motion_featdim = avg_motion.shape[1]
        #     write_motion2npz(avg_motion, motion_data.data_std[:motion_featdim], motion_data.data_mean[:motion_featdim], args.batch_size * max_frames,
        #                      args.output_dir + f'avg{gen_window}_motion.npz')

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
                (args.batch_size, model.njoints * model.nfeats + model.nrootfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            if args.write_progressive_motion == True:
                all_samples = []
                frame_num = args.batch_size * max_frames
                window_size = 4  # previously each generated motion process counted n frames
                for index, value in enumerate(motion_record):
                    r_sample = value['sample']
                    all_samples.append(r_sample[:, :, index:index + window_size])
                    r_sample = r_sample.permute(0, 2, 1).cpu()
                    r_sample = r_sample.reshape(-1, r_sample.shape[-1])
                    frame_num = args.batch_size * max_frames
                    write_motion2npz(r_sample, motion_data.data_std, motion_data.data_mean, frame_num,
                                     args.output_dir + f'dfstep{index}_motion.npz')
                all_samples.append(sample[:, :, len(motion_record) * window_size:max_frames])
                final_result = torch.cat(all_samples, dim=2)
                final_result = final_result.permute(0, 2, 1).cpu()
                final_result = final_result.reshape(-1, final_result.shape[-1])
                write_motion2npz(final_result, motion_data.data_std, motion_data.data_mean, frame_num,
                                 args.output_dir + f'dfchange_motion.npz')

            if args.unconstrained:
                all_text += ['unconstrained'] * args.num_samples
            else:
                pass
            # if (not args.manual_phase) and (sample.shape == motion.shape):
            #     sample_cuda = sample.cuda()
            #     loss = nn.MSELoss()(sample_cuda, motion)
            #     print(loss)
            #     squared_diff = (sample_cuda - motion) ** 2
            #     mse_per_batch = squared_diff.mean(dim=[1, 2])
            #     print(mse_per_batch)

            sample = sample.permute(0, 2, 1)
            all_motions.append(sample.cpu())
            # all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

            print(f"created {len(all_motions) * args.batch_size} samples")

        all_motions = torch.concatenate(all_motions, axis=0)
        frame_num = args.batch_size * max_frames
        df_motion = diffusion.motion_postprocess_gen(all_motions)
        if not args.manual_phase:

            def mse_per_item(df_motion, motion, lengths):
                """
                df_motion: (B, T, D)
                motion:    (B, T, D)
                lengths:   (B,) valid length of each batch clip
                return:    (B,) MSE for each sample
                """
                B, T, D = df_motion.shape

                # mask: (B, T, 1)
                mask = torch.arange(T, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
                mask = mask.unsqueeze(-1)  # (B, T, 1)

                # squared difference
                diff = (df_motion - motion) ** 2 * mask

                # MSE for each sample
                mse_per_sample = diff.sum(dim=(1, 2)) / (lengths * D)
                return mse_per_sample  # shape: (B,)
            mse_each = mse_per_item(df_motion, motion.permute(0, 2, 1), lengths)
            print(mse_each)
            df_motion[:, :, -12:] = motion[:, -12:, :].permute(0, 2, 1)
            df_contact_label = df_contact_label.permute(0, 2, 1)
            probabilities = torch.sigmoid(df_contact_label)
        else:
            df_contact_label = df_contact_label.permute(0, 2, 1)
            probabilities = torch.sigmoid(df_contact_label)
            predictions = (probabilities > 0.5).float()
            p_np = predictions.cpu().numpy()
            # init_pos = torch.zeros([args.batch_size, 3])
            # init_rot = torch.eye(3).repeat(args.batch_size, 1, 1)
            # init_rot = generate_y_axis_rotations(args.batch_size)
            # first_tensor = df_motion[0]
            # df_motion = first_tensor.unsqueeze(0).repeat(16, 1, 1)

            # draw_pic(p_np, angles)
        # For motion based on generated phase, we set init pos/init rot to 0 or unit matrix to match with ground truth motion
        # df_motion[:, :, -10] = velocities_expanded
        # df_motion[:, :, -9:] = R
        # df_motion[:, :, -12:] = relative_root_scaled.permute((0, 2, 1))
        # init_rot = create_rotation_matrix_y(0).repeat(args.batch_size, 1, 1)
        df_absolute_motion = diffusion.transfer2absolute_batch(df_motion, init_pos, init_rot)
        # df_motion_np = df_absolute_motion * motion_data.data_std + motion_data.data_mean
        # df_motion_np = diffusion.integrate_root_motion(df_motion_np)
        # df_motion_np = df_motion_np.cpu().numpy()
        # df_motion_data = {
        #     f'motion_{i}': df_motion_np[i]  # each has shape (208, 405)
        #     for i in range(df_motion_np.shape[0])
        # }
        # np.savez_compressed('df_motion.npz', **df_motion_data)

        # set the starting frame contact as zero, to avoid Untiy to connect two different motions
        probabilities[:, 0, :] = 0.
        # df_absolute_motion = df_absolute_motion.reshape(-1, df_absolute_motion.shape[2]).numpy()
        # df_contact_label = probabilities.reshape(-1, 2).cpu().numpy()
        # write_motion2npz(df_absolute_motion, motion_data.data_std, motion_data.data_mean, frame_num,
        #                  args.output_dir + f'df{gen_window}_motion.npz', True, df_contact_label)
        lengths = batch['lengths']
        segments = []
        contactlabel_segments = []
        for i in range(args.batch_size):
            seg = df_absolute_motion[i, :lengths[i], :]
            final_motion_list.append(seg)
            contact_label = probabilities[i, :lengths[i], :]
            segments.append(seg.cpu())
            contactlabel_segments.append(contact_label.cpu())
            # write_motion2npz(seg.cpu().numpy(), motion_data.data_std, motion_data.data_mean, seg.shape[0],
            #                  args.output_dir + (
            #                      f'df{gen_window}_motion_gtphaseclipped_{i}.npz' if args.use_gt_phase else f'df{gen_window}_motion_dfphaseclipped_{i}.npz'),
            #                  True, contact_label.cpu().numpy())

        concatenated_motion = np.concatenate(segments, axis=0)
        concatenated_label = np.concatenate(contactlabel_segments, axis=0)
        frame_num = concatenated_motion.shape[0]
        # debug
        # write_motion2npz(concatenated_motion, motion_data.data_std, motion_data.data_mean, frame_num,
        #                  args.output_dir + (f'df{gen_window}_motion_gtphaseclipped.npz' if args.use_gt_phase else f'df{gen_window}_motion_dfphaseclipped.npz'), True, concatenated_label)
        df_absolute_motion = df_absolute_motion.reshape(-1, df_absolute_motion.shape[2]).numpy()

        # write motion for each batch, turn off (default)
        # for i in range(args.batch_size):
        #     frame_num =[i * max_frames, i * max_frames + max_frames]
        #     # all_motions_cpu = all_motions.cpu().detach().numpy()
        #     write_motion2npz(df_absolute_motion, motion_data.data_std, motion_data.data_mean, frame_num,
        #                      args.output_dir + f'df{gen_window}_motion_{i}.npz', True)
        #     if (not args.manual_phase) and (not args.generated_phase):
        #         write_motion2npz(gt_absolute_motion, motion_data.data_std, motion_data.data_mean, frame_num,
        #                          args.output_dir + f'gt{gen_window}_motion_{i}.npz', True)
    os.makedirs(args.output_dir, exist_ok=True)
    final_npz_path = args.output_dir + "/final_df_motion_list.npz"
    df_motion_data = {
        f'motion_{i}': remove_root_from_single_sequence(final_motion_list[i] * motion_data.data_std + motion_data.data_mean)
        for i in range(len(final_motion_list))
    }
    np.savez_compressed(final_npz_path, **df_motion_data)
    print(f"Saved all generated motions to {final_npz_path}")

    final_npz_path = args.output_dir + "/final_gt_motion_list.npz"
    gt_motion_data = {
        f'motion_{i}': remove_root_from_single_sequence(final_gt_motion_list[i] * motion_data.data_std + motion_data.data_mean)
        for i in range(len(final_gt_motion_list))
    }
    np.savez_compressed(final_npz_path, **gt_motion_data)
    print(f"Saved all generated motions to {final_npz_path}")


if __name__ == "__main__":
    main()
