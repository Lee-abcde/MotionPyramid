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

from option import TrainVQOptionParser, TestOptionParser

from dataset import create_dataset_from_args,create_mdm_dataset_from_args
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


def test_forward():
    option_parser = TrainVQOptionParser()
    # set diffusion model, args contains info to load the pretrained model
    diff_args = train_args()

    file_path = osp.join(diff_args.pretrained_save, "args.txt")
    with open(file_path, "r") as f:
        args = option_parser.text_deserialize(f.read().split())
        args = option_parser.post_process(args)

    # Load = args.load
    Save = diff_args.pretrained_save

    motion_datas = create_dataset_from_args(args)
    # Build network model
    networks, VQ = model.create_model_from_args(args, motion_datas)

    # Find the latest epoch file after locating files
    ref_files = [f for f in os.listdir(Save) if f.endswith("Channels_VQ.pt")]
    ref_files.sort(key=lambda x: int(x.split('_')[0]))
    largest_epoch = ref_files[-1].split('_')[0]

    phase_decoders = []
    # Load model weights
    for i in range(len(networks)):
        network = networks[i]
        target_file = f'{largest_epoch}_{i}_{args.phase_channels}Channels.pt'
        state_dict = torch.load(osp.join(Save, target_file), map_location='cpu')
        network.load_state_dict(state_dict)

    VQ_target_file = f'{largest_epoch}_{args.phase_channels}Channels_VQ.pt'
    state_dict = torch.load(osp.join(Save, VQ_target_file), map_location='cpu')
    state_dict = clean_vq_state_dict(state_dict)
    VQ.load_state_dict(state_dict, strict=False)

    # Move models to the device and set inference mode
    for i in range(len(networks)):
        networks[i] = utility.ToDevice(networks[i])
        networks[i].eval()

    for i in range(len(phase_decoders)):
        phase_decoders[i] = utility.ToDevice(phase_decoders[i])
        phase_decoders[i].train()  # So it won't perform an extra normalization

    # VQ is part of the pretrained networks
    VQ = utility.ToDevice(VQ)
    VQ.eval()

    motion_data = motion_datas[0]
    data_loader = DataLoader(motion_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    mdm_model, diffusion = create_model_and_diffusion(diff_args, data_loader)
    mdm_model = mdm_model.cuda()
    # diffusion = diffusion.cuda()

    batch_size = 32
    njoints = 28
    nfeats = 15
    phase_info = 10
    max_frames = 61
    x = torch.randn((batch_size, njoints * nfeats + phase_info, max_frames)).cuda()
    # Initialize random input
    # x = 0
    # for batch_data in data_loader:
    #     # batch_data is one batch of data here
    #     # You can directly use batch_data for training, printing, or other operations
    #     x = batch_data.cuda()
    #     break  # If only one batch is needed, use break to exit the loop
    # x = torch.randn((2* batch_size, njoints * nfeats, max_frames)).cuda()

    # Initialize timestep tensor with shape [batch_size]
    timesteps = torch.randint(0, 1000, (batch_size,)).cuda()

    # pae_input = motion_data.get_feature_by_names(x, args.needed_channel_names)
    # human_network = networks[0]
    # pae_input = utility.ToDevice(pae_input)
    # yPred, latent, signal, params, vq_info = human_network(pae_input)
    # manifold = params[4]
    # manifold = manifold.permute(0, 2, 1)
    # A, B, C = manifold.shape
    # manifold = manifold.reshape(A, B * C)
    # Initialize the condition dictionary y; include keys such as text or action as required by the model
    manifold = torch.randn((batch_size, max_frames*phase_info)).cuda()
    y = {
        # 'text': ["random text"] * batch_size,  # if using text embedding
        # 'action': torch.randint(0, 5, (batch_size,)),  # assuming action ranges from 0 to num_actions
        'text_embed': manifold  # random embedding replacing text
    }

    output = mdm_model(x, timesteps, y)
    print("Output shape:", output.shape)
    # train_platform.close()


def main():
    option_parser = TrainVQOptionParser()
    # set diffusion model, args contains info to load the pretrained model
    diff_args = train_args()

    file_path = osp.join(diff_args.pretrained_save, "args.txt")
    with open(file_path, "r") as f:
        args_dict = json.load(f)  # read as a dictionary
        vq_args = argparse.Namespace(**args_dict)  # convert the dictionary to Namespace
        vq_args = option_parser.post_process(vq_args)

    motion_datas = create_mdm_dataset_from_args(vq_args, diff_args)
    motion_data = motion_datas[0]
    train_dataset = Subset(motion_data, motion_data.train_set_index)
    train_dataset.data_std, train_dataset.data_mean = motion_data.data_std, motion_data.data_mean
    data_loader = DataLoader(train_dataset, batch_size=diff_args.batch_size, shuffle=True, num_workers=0, drop_last=True,
                             pin_memory=True)
    mdm_model, diffusion = create_model_and_diffusion(diff_args, vq_args, data_loader)
    # if diff_args.load_pretrained_diffusion_model == True:
    #     state_dict = torch.load('./results/difftest/model000052008.pt', map_location='cpu')
    #     load_model_wo_clip(mdm_model, state_dict)
    #     # if args.guidance_param != 1:
    #     #     mdm_model = ClassifierFreeSampleModel(mdm_model)   # wrapping model with the classifier-free sampler
    #     # model = model.cuda()
    #     mdm_model.eval()  # disable random masking
    mdm_model = mdm_model.cuda()

    TrainLoop(diff_args, None, mdm_model, diffusion, data_loader).run_loop()

if __name__ == '__main__':
    # test_forward()
    main()
