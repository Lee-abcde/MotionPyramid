import argparse
import json
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader

from dataset import create_txt2phase_dataset_from_args
from generate_text2phase import measure_text2phase
from models.text2phase_momask import Text2PhaseMoMask
from option import TrainVQOptionParser
from train_text2phase_momask import collate_momask_text2phase


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, required=True)
    parser.add_argument("--pretrained_save", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--input_set", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--target_length", type=int, default=208)
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--cond_scale", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk_filter_thres", type=float, default=0.9)
    parser.add_argument("--sample_mode", type=str, default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--smooth_min_run", type=int, default=0)
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
        collate_fn=lambda batch: collate_momask_text2phase(batch, target_length=target_length),
    )


def load_codebook(pretrained_save):
    ref_files = [f for f in os.listdir(pretrained_save) if f.endswith("Channels_VQ.pt")]
    ref_files.sort(key=lambda x: int(x.split("_")[0]))
    if not ref_files:
        raise FileNotFoundError(f"No VQ checkpoint found under {pretrained_save}")
    vq_target_file = ref_files[-1]
    state_dict = torch.load(osp.join(pretrained_save, vq_target_file), map_location="cpu")
    state_dict = clean_vq_state_dict(state_dict)
    embedding_keys = [k for k in state_dict.keys() if k.endswith("embedding.weight")]
    if not embedding_keys:
        raise KeyError(f"No embedding.weight found in {vq_target_file}")
    embedding_keys.sort()
    return state_dict[embedding_keys[0]].detach().float()


def ids_to_manifold(ids, angle, codebook):
    angle_output = angle.permute(0, 2, 1).unsqueeze(-1)
    safe_ids = ids.clamp(min=0).cpu()
    code = codebook.cpu()[safe_ids]
    code = code.reshape(code.shape[0], code.shape[1], -1, 2)
    manifold = (code @ angle_output.cpu()).squeeze(-1)
    return manifold


def pad_sequence_feature(x, target_length, pad_value=0.0):
    if x.shape[-1] == target_length:
        return x
    pad_len = target_length - x.shape[-1]
    if pad_len < 0:
        raise ValueError(f"sequence length {x.shape[-1]} exceeds target_length {target_length}")
    return torch.nn.functional.pad(x, (0, pad_len), value=pad_value)


def smooth_short_runs(ids, lengths, min_run):
    if min_run <= 1:
        return ids
    smoothed = ids.clone()
    for b in range(ids.shape[0]):
        length = int(lengths[b].item())
        seq = smoothed[b, :length].clone()
        start = 0
        while start < length:
            end = start + 1
            while end < length and seq[end] == seq[start]:
                end += 1
            run_len = end - start
            if run_len < min_run:
                left_val = seq[start - 1] if start > 0 else None
                right_val = seq[end] if end < length else None
                fill_val = None
                if left_val is not None and right_val is not None:
                    fill_val = left_val if run_len <= min_run else right_val
                    if left_val != right_val:
                        fill_val = left_val
                elif left_val is not None:
                    fill_val = left_val
                elif right_val is not None:
                    fill_val = right_val
                if fill_val is not None:
                    seq[start:end] = fill_val
            start = end
        smoothed[b, :length] = seq
    return smoothed


def save_phase_bundle(save_path, manifold_list, style_list, traj_list, angle_list, phase_id_list, length_list):
    torch.save(
        {
            "manifold": torch.cat(manifold_list, dim=0),
            "stylecode": torch.cat(style_list, dim=0),
            "length": length_list,
            "relative_traj": torch.cat(traj_list, dim=0),
            "angle_xy": torch.cat(angle_list, dim=0),
            "phase_ids": torch.cat(phase_id_list, dim=0),
        },
        save_path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    save_dir = osp.join(args.output_dir, "eval_text2phase_momask")
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
    loader = build_loader(dataset, args.batch_size, args.num_workers, args.target_length)
    checkpoint = torch.load(args.model_path, map_location="cpu")
    model = Text2PhaseMoMask(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    codebook = load_codebook(args.pretrained_save)

    all_gen_manifold = []
    all_gen_style = []
    all_gen_traj = []
    all_gen_angle = []
    all_gen_phase_ids = []
    all_gen_length = []
    all_pred_idx_gt_angle_manifold = []
    all_pred_idx_gt_angle_style = []
    all_pred_idx_gt_angle_traj = []
    all_pred_idx_gt_angle_angle = []
    all_pred_idx_gt_angle_phase_ids = []
    all_gt_idx_pred_angle_manifold = []
    all_gt_idx_pred_angle_style = []
    all_gt_idx_pred_angle_traj = []
    all_gt_idx_pred_angle_angle = []
    all_gt_idx_pred_angle_phase_ids = []
    all_gt_manifold = []
    all_gt_style = []
    all_gt_traj = []
    all_gt_angle = []
    all_gt_phase_ids = []
    all_gt_length = []
    aggregate_loss = {}
    aggregate_token_correct = 0.0
    aggregate_token_total = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            print(f"batch[{batch_idx + 1}/{len(loader)}]")
            outputs = model.generate(
                batch["text_embed"].to(device),
                batch["lengths"].to(device),
                timesteps=args.timesteps,
                cond_scale=args.cond_scale,
                temperature=args.temperature,
                topk_filter_thres=args.topk_filter_thres,
                sample_mode=args.sample_mode,
            )

            ids = outputs["ids"]
            angle = outputs["angle"]
            style = outputs["style"].cpu()
            traj = outputs["traj"].cpu()
            angle = pad_sequence_feature(angle.cpu(), args.target_length, pad_value=0.0)
            style = pad_sequence_feature(style, args.target_length, pad_value=0.0)
            traj = pad_sequence_feature(traj, args.target_length, pad_value=0.0)
            ids = pad_sequence_feature(ids.cpu(), args.target_length, pad_value=-1).long()
            ids = smooth_short_runs(ids, batch["lengths"], args.smooth_min_run)
            gen_manifold = ids_to_manifold(ids, angle, codebook).permute(0, 2, 1)

            gt_ids = batch["manifold_index"]
            gt_angle = batch["angle_xy"].permute(0, 2, 1)
            gt_style = batch["stylecode"].permute(0, 2, 1)
            gt_traj = batch["relative_trajectory"].permute(0, 2, 1)
            gt_manifold = ids_to_manifold(gt_ids, gt_angle, codebook).permute(0, 2, 1)
            pred_idx_gt_angle_manifold = ids_to_manifold(ids, gt_angle, codebook).permute(0, 2, 1)
            gt_idx_pred_angle_manifold = ids_to_manifold(gt_ids, angle, codebook).permute(0, 2, 1)

            valid_token_mask = batch["mask"].squeeze(-1).bool()
            aggregate_token_correct += (ids[valid_token_mask] == gt_ids[valid_token_mask]).float().sum().item()
            aggregate_token_total += valid_token_mask.float().sum().item()

            pred_full = torch.cat([gen_manifold, style, traj], dim=1).to(device)
            gt_full = torch.cat([gt_manifold, gt_style, gt_traj], dim=1).to(device)
            batch_loss = measure_text2phase(gt_full, pred_full, batch["mask"].to(device), use_manifold=True)
            for key, value in batch_loss.items():
                aggregate_loss[key] = aggregate_loss.get(key, 0.0) + value.item()
            total_batches += 1

            all_gen_manifold.append(gen_manifold)
            all_gen_style.append(style)
            all_gen_traj.append(traj)
            all_gen_angle.append(angle.cpu())
            all_gen_phase_ids.append(ids.cpu())
            all_gen_length.extend(batch["lengths"].tolist())
            all_pred_idx_gt_angle_manifold.append(pred_idx_gt_angle_manifold)
            all_pred_idx_gt_angle_style.append(gt_style.cpu())
            all_pred_idx_gt_angle_traj.append(gt_traj.cpu())
            all_pred_idx_gt_angle_angle.append(gt_angle.cpu())
            all_pred_idx_gt_angle_phase_ids.append(ids.cpu())
            all_gt_idx_pred_angle_manifold.append(gt_idx_pred_angle_manifold)
            all_gt_idx_pred_angle_style.append(style)
            all_gt_idx_pred_angle_traj.append(traj)
            all_gt_idx_pred_angle_angle.append(angle.cpu())
            all_gt_idx_pred_angle_phase_ids.append(gt_ids.cpu())

            all_gt_manifold.append(gt_manifold)
            all_gt_style.append(gt_style)
            all_gt_traj.append(gt_traj)
            all_gt_angle.append(gt_angle.cpu())
            all_gt_phase_ids.append(gt_ids.cpu())
            all_gt_length.extend(batch["lengths"].tolist())

    save_phase_bundle(
        osp.join(save_dir, "generated_phase_results.pt"),
        all_gen_manifold,
        all_gen_style,
        all_gen_traj,
        all_gen_angle,
        all_gen_phase_ids,
        all_gen_length,
    )
    save_phase_bundle(
        osp.join(save_dir, "gt_phase_results.pt"),
        all_gt_manifold,
        all_gt_style,
        all_gt_traj,
        all_gt_angle,
        all_gt_phase_ids,
        all_gt_length,
    )
    save_phase_bundle(
        osp.join(save_dir, "pred_index_gt_angle_traj_style.pt"),
        all_pred_idx_gt_angle_manifold,
        all_pred_idx_gt_angle_style,
        all_pred_idx_gt_angle_traj,
        all_pred_idx_gt_angle_angle,
        all_pred_idx_gt_angle_phase_ids,
        all_gen_length,
    )
    save_phase_bundle(
        osp.join(save_dir, "gt_index_pred_angle_traj_style.pt"),
        all_gt_idx_pred_angle_manifold,
        all_gt_idx_pred_angle_style,
        all_gt_idx_pred_angle_traj,
        all_gt_idx_pred_angle_angle,
        all_gt_idx_pred_angle_phase_ids,
        all_gen_length,
    )

    if total_batches > 0:
        print("Average metrics:")
        for key, value in aggregate_loss.items():
            print(f"  {key}: {value / total_batches:.6f}")
        if aggregate_token_total > 0:
            print(f"  token_acc: {aggregate_token_correct / aggregate_token_total:.6f}")


if __name__ == "__main__":
    main()
