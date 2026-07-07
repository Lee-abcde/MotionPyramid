import argparse
import json
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader

from dataset import create_txt2phase_dataset_from_args
from models.text2phase_baseline import (
    Text2PhaseTransformerBaseline,
    compute_masked_text2phase_manifold_losses,
)
from option import TrainVQOptionParser
from train_txt2phase import collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="text2phase")
    parser.add_argument("--load", type=str, required=True)
    parser.add_argument("--pretrained_save", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--target_length", type=int, default=208)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
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


def build_loader(dataset, batch_size, num_workers, shuffle, target_length):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
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
    manifold_cont = batch["manifold_continuous"]
    stylecode = batch["stylecode"]
    relative_traj = batch["relative_trajectory"]
    return torch.cat([manifold_cont, stylecode, relative_traj], dim=2).permute(0, 2, 1)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    totals = {"loss": 0.0, "manifold_mse": 0.0, "style_mse": 0.0, "traj_mse": 0.0}
    total_batches = 0

    for batch in data_loader:
        text_embed = batch["text_embed"].to(device)
        lengths = batch["lengths"].to(device)
        mask = batch["mask"].to(device)
        target = build_target({k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})

        prediction = model(text_embed, lengths=lengths, target_length=target.shape[-1])
        prediction = prediction * mask.permute(0, 2, 1)
        losses = compute_masked_text2phase_manifold_losses(target, prediction, mask)

        for key in totals:
            totals[key] += losses[key].item()
        total_batches += 1

    if total_batches == 0:
        return totals
    return {key: value / total_batches for key, value in totals.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vq_args = build_vq_args(args.pretrained_save)
    diff_args = argparse.Namespace(
        load=args.load,
        pretrained_save=args.pretrained_save,
        std_cap=vq_args.std_cap,
        num_embed_vq=vq_args.num_embed_vq,
    )

    train_dataset = create_txt2phase_dataset_from_args(vq_args, diff_args, dataset_mode="train")[0]
    test_dataset = create_txt2phase_dataset_from_args(vq_args, diff_args, dataset_mode="test")[0]

    train_loader = build_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        target_length=args.target_length,
    )
    test_loader = build_loader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        target_length=args.target_length,
    )

    model = Text2PhaseTransformerBaseline(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=args.target_length,
        manifold_dim=vq_args.n_latent_channel,
        style_dim=vq_args.stylecode_dim,
        traj_dim=12,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_config = vars(args).copy()
    run_config["model_config"] = model.get_config()
    run_config["vq_latent_dim"] = vq_args.n_latent_channel
    with open(osp.join(args.save_dir, "baseline_args.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    best_test_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "manifold_mse": 0.0, "style_mse": 0.0, "traj_mse": 0.0}
        total_batches = 0

        for step, batch in enumerate(train_loader, start=1):
            text_embed = batch["text_embed"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)
            target = build_target({k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()})

            prediction = model(text_embed, lengths=lengths, target_length=target.shape[-1])
            prediction = prediction * mask.permute(0, 2, 1)
            losses = compute_masked_text2phase_manifold_losses(target, prediction, mask)

            optimizer.zero_grad()
            losses["loss"].backward()
            optimizer.step()

            for key in running:
                running[key] += losses[key].item()
            total_batches += 1

            if step % args.log_interval == 0:
                print(
                    f"epoch[{epoch}] step[{step}/{len(train_loader)}] "
                    f"loss[{losses['loss'].item():.6f}] "
                    f"manifold[{losses['manifold_mse'].item():.6f}] "
                    f"style[{losses['style_mse'].item():.6f}] "
                    f"traj[{losses['traj_mse'].item():.6f}]"
                )

        train_metrics = {key: value / max(total_batches, 1) for key, value in running.items()}
        test_metrics = evaluate(model, test_loader, device)

        print(
            f"epoch[{epoch}] "
            f"train_loss[{train_metrics['loss']:.6f}] "
            f"test_loss[{test_metrics['loss']:.6f}] "
            f"test_manifold[{test_metrics['manifold_mse']:.6f}] "
            f"test_style[{test_metrics['style_mse']:.6f}] "
            f"test_traj[{test_metrics['traj_mse']:.6f}]"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.get_config(),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }
        torch.save(checkpoint, osp.join(args.save_dir, "latest.pt"))

        if test_metrics["loss"] < best_test_loss:
            best_test_loss = test_metrics["loss"]
            torch.save(checkpoint, osp.join(args.save_dir, "best.pt"))


if __name__ == "__main__":
    main()
