import argparse
import json
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader

import Library.Utility as utility
from dataset import create_dataset_from_args, create_txt2phase_dataset_from_args
from generate_text2phase import measure_text2phase, project_manifold_to_codebook
from models import VQ as VQ_model
from models.text2phase_baseline import Text2PhaseTransformerBaseline
from option import TrainVQOptionParser
from train_txt2phase import collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="text2phase")
    parser.add_argument("--pretrained_save", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--input_set", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--target_length", type=int, default=208)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_vq_args(pretrained_save):
    option_parser = TrainVQOptionParser()
    file_path = osp.join(pretrained_save, "args.txt")
    with open(file_path, "r") as f:
        args_dict = json.load(f)
    vq_args = argparse.Namespace(**args_dict)
    return option_parser.post_process(vq_args)


def clean_vq_state_dict(state_dict):
    for key in list(state_dict.keys()):
        if not (key.startswith("embedding") or key.startswith("vqs.")):
            state_dict.pop(key)
    return state_dict


def build_loader(dataset, batch_size, num_workers, target_length):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(
            batch,
            num_embed_vq=dataset.num_embed_vq,
            target_length=target_length,
        ),
    )


def build_target(batch):
    return torch.cat(
        [batch["manifold_continuous"], batch["stylecode"], batch["relative_trajectory"]],
        dim=2,
    ).permute(0, 2, 1)


def load_vq(vq_args, pretrained_save):
    motion_datas = create_dataset_from_args(vq_args)
    _, vq_model = VQ_model.create_model_from_args(vq_args, motion_datas)

    ref_files = [f for f in os.listdir(pretrained_save) if f.endswith("Channels_VQ.pt")]
    ref_files.sort(key=lambda x: int(x.split("_")[0]))
    largest_epoch = ref_files[-1].split("_")[0]
    vq_target_file = f"{largest_epoch}_{vq_args.phase_channels}Channels_VQ.pt"
    state_dict = torch.load(osp.join(pretrained_save, vq_target_file), map_location="cpu")
    state_dict = clean_vq_state_dict(state_dict)
    vq_model.load_state_dict(state_dict, strict=False)
    vq_model = utility.ToDevice(vq_model)
    vq_model.eval()
    return vq_model


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    save_dir = osp.join(args.output_dir, "eval_text2phase_baseline")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq_args = build_vq_args(args.pretrained_save)
    diff_args = argparse.Namespace(
        load=args.load,
        pretrained_save=args.pretrained_save,
        std_cap=vq_args.std_cap,
        num_embed_vq=vq_args.num_embed_vq,
    )

    dataset = create_txt2phase_dataset_from_args(vq_args, diff_args, dataset_mode=args.input_set)[0]
    data_loader = build_loader(dataset, args.batch_size, args.num_workers, args.target_length)

    checkpoint = torch.load(args.model_path, map_location="cpu")
    model = Text2PhaseTransformerBaseline(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    vq_model = load_vq(vq_args, args.pretrained_save)

    all_gen_manifold = []
    all_gen_style = []
    all_gen_traj = []
    all_gen_length = []
    all_gt_manifold = []
    all_gt_style = []
    all_gt_traj = []
    all_gt_length = []

    aggregate_loss_raw = {}
    aggregate_loss_proj = {}
    total_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            print(f"batch[{batch_idx + 1}/{len(data_loader)}]")
            text_embed = batch["text_embed"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)

            target = build_target({k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})
            prediction = model(text_embed, lengths=lengths, target_length=target.shape[-1])
            prediction = prediction * mask.permute(0, 2, 1)

            loss_raw = measure_text2phase(target, prediction, mask, use_manifold=True)
            loss_proj = measure_text2phase(target, prediction, mask, use_manifold=True, VQ=vq_model)

            for key, value in loss_raw.items():
                aggregate_loss_raw[key] = aggregate_loss_raw.get(key, 0.0) + value.item()
            for key, value in loss_proj.items():
                aggregate_loss_proj[key] = aggregate_loss_proj.get(key, 0.0) + value.item()
            total_batches += 1

            n_latent = prediction.shape[1] - 13
            pred_manifold = prediction[:, :n_latent, :].permute(0, 2, 1)
            pred_style = prediction[:, n_latent:n_latent + 1, :].cpu()
            pred_traj = prediction[:, n_latent + 1:, :].cpu()
            proj_manifold = project_manifold_to_codebook(pred_manifold, vq_model).cpu()

            gt_manifold = target[:, :n_latent, :].permute(0, 2, 1).cpu()
            gt_style = target[:, n_latent:n_latent + 1, :].cpu()
            gt_traj = target[:, n_latent + 1:, :].cpu()

            all_gen_manifold.append(proj_manifold)
            all_gen_style.append(pred_style)
            all_gen_traj.append(pred_traj)
            all_gen_length.extend(lengths.cpu().tolist())

            all_gt_manifold.append(gt_manifold)
            all_gt_style.append(gt_style)
            all_gt_traj.append(gt_traj)
            all_gt_length.extend(lengths.cpu().tolist())

    torch.save(
        {
            "manifold": torch.cat(all_gen_manifold, dim=0),
            "stylecode": torch.cat(all_gen_style, dim=0),
            "length": all_gen_length,
            "relative_traj": torch.cat(all_gen_traj, dim=0),
        },
        osp.join(save_dir, "generated_phase_results.pt"),
    )
    torch.save(
        {
            "manifold": torch.cat(all_gt_manifold, dim=0),
            "stylecode": torch.cat(all_gt_style, dim=0),
            "length": all_gt_length,
            "relative_traj": torch.cat(all_gt_traj, dim=0),
        },
        osp.join(save_dir, "gt_phase_results.pt"),
    )

    if total_batches > 0:
        print("Average RAW metrics:")
        for key, value in aggregate_loss_raw.items():
            print(f"  {key}: {value / total_batches:.6f}")
        print("Average PROJECTED metrics:")
        for key, value in aggregate_loss_proj.items():
            print(f"  {key}: {value / total_batches:.6f}")


if __name__ == "__main__":
    main()
