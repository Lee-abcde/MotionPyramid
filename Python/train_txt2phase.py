import os
import os.path as osp

from tqdm import tqdm

import Library.Utility as utility
from models import VQ as model
import json
import argparse

import pickle

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from option import TrainVQOptionParser, TestOptionParser

from dataset import create_dataset_from_args, create_txt2phase_dataset_from_args
from modules import save_manifold
from utils.criteria_test import get_usage, get_dataset_usage, get_combinatorial_usage, get_combinatorial_dataset_usage
from train_diff import train_args, create_model_and_diffusion
from utils.training_loop import TrainLoop
from generate import load_model_wo_clip, ClassifierFreeSampleModel
from torch.utils.data import Subset
class PositionLoss:
    def __init__(self, feature_name, feature_dims, std, mean):
        self.idx = feature_name.index('Positions')
        self.feature_dims = feature_dims
        self.std = torch.from_numpy(std).cuda()
        self.mean = torch.from_numpy(mean).cuda()

    def get(self, v):
        v = v * self.std + self.mean
        for i in range(self.idx):
            v = v[..., self.feature_dims[i]:]
        return v[..., :self.feature_dims[self.idx]].reshape(-1, 3)

    def __call__(self, x, y):
        x = self.get(x)
        y = self.get(y)
        diff = ((x - y) ** 2).sum(-1).mean()
        return diff

def accumulate_usage(network, VQ, motion_data, args):
    E = np.arange(len(motion_data))
    batch_size = args.batch_size * 16
    loop = tqdm(range(0, len(motion_data), batch_size))
    usage = np.zeros(VQ.num_embed, dtype=np.int32)

    for i in loop:
        eval_indices = E[i:i + batch_size]
        eval_batch = motion_data.load_batches(eval_indices)[..., :args.frames]
        eval_batch = utility.ToDevice(eval_batch)
        output = network(eval_batch)
        vq_info = output[4]
        index = vq_info[3].detach().cpu().numpy()
        np.add.at(usage, index, 1)

    return usage


def clean_vq_state_dict(state_dict):
    for key in list(state_dict.keys()):
        if not (key.startswith("embedding") or key.startswith("vqs.")):
            state_dict.pop(key)
    return state_dict

def collate_fn(batch, num_embed_vq, target_length=208):
    # Unpack each item in the batch
    angle_xy_list = [item['angle_xy'] for item in batch]
    stylecode_list = [item['stylecode'] for item in batch]
    manifold_index_list = [item['manifold_index'] for item in batch]
    text_list = [item['text'] for item in batch]
    text_embed_list = [item['text_embed'] for item in batch]
    lengths = [item['m_length'] for item in batch]
    motion = [item['motion'] for item in batch]
    relative_trajectory = [item['relative_trajectory'] for item in batch]

    # Check if manifold_continuous is available
    has_manifold_cont = batch[0].get('manifold_continuous', None) is not None
    if has_manifold_cont:
        manifold_cont_list = [item['manifold_continuous'] for item in batch]

    # Ensure no sequence exceeds the target length
    assert all(l <= target_length for l in lengths), "Some sequences exceed the target length."

    # Pad angle_xy to target_length with zeros
    padded_angle_xy = torch.stack([
        F.pad(xy, (0, 0, 0, target_length - xy.size(0)))  # pad rows
        for xy in angle_xy_list
    ])  # Shape: (B, target_length, 2)

    # Pad stylecode to target_length
    padded_stylecode = torch.stack([
        F.pad(sc, (0, 0, 0, target_length - sc.size(0)))
        for sc in stylecode_list
    ])  # Shape: (B, target_length, D_style)

    # Pad manifold index with -1
    padded_manifold_idx = torch.stack([
        F.pad(mi, (0, 0, 0, target_length - mi.size(0)), value=-1)
        for mi in manifold_index_list
    ])  # Shape: (B, target_length, 1) or (B, target_length)

    # Create mask (1 for valid positions, 0 for padding)
    mask = (padded_manifold_idx != -1).float()

    # Replace -1 with 0 to avoid indexing errors in one-hot
    safe_manifold_idx = padded_manifold_idx.clone()
    safe_manifold_idx[safe_manifold_idx == -1] = 0

    # If manifold_index is shape (B, T, 1), squeeze it to (B, T)
    if safe_manifold_idx.ndim == 3:
        safe_manifold_idx = safe_manifold_idx.squeeze(-1)

    # One-hot encode manifold indices
    manifold_onehot = F.one_hot(safe_manifold_idx.long(), num_classes=num_embed_vq).float()
    # Shape: (B, target_length, num_embed_vq)

    # Stack text embeddings
    text_embed = torch.stack(text_embed_list)  # Shape: (B, D_text)

    padded_rela_trajectory = torch.stack([
        F.pad(traj, (0, 0, 0, target_length - traj.size(0)))  # pad rows
        for traj in relative_trajectory
    ])

    result = {
        'angle_xy': padded_angle_xy,
        'stylecode': padded_stylecode,
        'manifold_onehot': manifold_onehot,
        'text_embed': text_embed,
        'mask': mask,
        'lengths': torch.tensor(lengths),
        'text': text_list,
        'motion': motion,
        'relative_trajectory': padded_rela_trajectory
    }

    # Pad manifold_continuous if available
    if has_manifold_cont:
        padded_manifold_cont = torch.stack([
            F.pad(mc, (0, 0, 0, target_length - mc.size(0)))
            for mc in manifold_cont_list
        ])  # Shape: (B, target_length, n_latent_channel)
        result['manifold_continuous'] = padded_manifold_cont

    return result

def main():
    option_parser = TrainVQOptionParser()
    # set diffusion model, args contains info to load the pretrained model
    diff_args = train_args()

    file_path = osp.join(diff_args.pretrained_save, "args.txt")
    with open(file_path, "r") as f:
        args_dict = json.load(f)  # read as a dictionary
        vq_args = argparse.Namespace(**args_dict)  # convert the dictionary to Namespace
        vq_args = option_parser.post_process(vq_args)

    txt2phase_data = create_txt2phase_dataset_from_args(vq_args, diff_args, dataset_mode='train')[0]

    data_loader = DataLoader(txt2phase_data, batch_size=diff_args.batch_size, shuffle=True, num_workers=0, drop_last=True,
                             pin_memory=True, collate_fn=lambda batch: collate_fn(batch, num_embed_vq=txt2phase_data.num_embed_vq))
    mdm_model, diffusion = create_model_and_diffusion(diff_args, vq_args, data_loader)
    mdm_model = mdm_model.cuda()

    TrainLoop(diff_args, None, mdm_model, diffusion, data_loader).run_text2phase_loop()

if __name__ == '__main__':
    main()