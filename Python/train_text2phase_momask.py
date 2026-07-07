import argparse
import json
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader

from dataset import create_txt2phase_dataset_from_args
from models.text2phase_momask import Text2PhaseMoMask
from option import TrainVQOptionParser


def collate_momask_text2phase(batch, target_length=208, pad_id=-1):
    angle_xy = []
    stylecode = []
    manifold_index = []
    text_embed = []
    lengths = []
    relative_trajectory = []
    text = []

    for item in batch:
        length = item["m_length"]
        if length > target_length:
            continue
        lengths.append(length)
        angle_xy.append(item["angle_xy"])
        stylecode.append(item["stylecode"])
        manifold_index.append(item["manifold_index"].squeeze(-1) if item["manifold_index"].ndim == 2 else item["manifold_index"])
        text_embed.append(item["text_embed"])
        relative_trajectory.append(item["relative_trajectory"])
        text.append(item["text"])

    if len(lengths) == 0:
        raise ValueError("Empty batch after filtering by target_length.")

    def pad_tensor(x, pad_value=0.0):
        return torch.nn.functional.pad(x, (0, 0, 0, target_length - x.size(0)), value=pad_value)

    padded_angle = torch.stack([pad_tensor(x) for x in angle_xy])
    padded_style = torch.stack([pad_tensor(x) for x in stylecode])
    padded_traj = torch.stack([pad_tensor(x) for x in relative_trajectory])
    padded_idx = torch.stack(
        [torch.nn.functional.pad(x, (0, target_length - x.size(0)), value=pad_id) for x in manifold_index]
    )
    mask = (padded_idx != pad_id).unsqueeze(-1).float()

    return {
        "angle_xy": padded_angle,
        "stylecode": padded_style,
        "manifold_index": padded_idx,
        "text_embed": torch.stack(text_embed),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "relative_trajectory": padded_traj,
        "mask": mask,
        "text": text,
    }


def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    parser.add_argument("--token_loss_weight", type=float, default=2.0)
    parser.add_argument("--angle_loss_weight", type=float, default=2.0)
    parser.add_argument("--style_loss_weight", type=float, default=1.0)
    parser.add_argument("--traj_loss_weight", type=float, default=10.0)
    parser.add_argument("--freeze_token_emb", action="store_true")
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
    codebook = state_dict[embedding_keys[0]].detach().float()
    return codebook


def build_loader(dataset, batch_size, num_workers, shuffle, target_length):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=lambda batch: collate_momask_text2phase(batch, target_length=target_length),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = {
        "loss": 0.0,
        "token_loss": 0.0,
        "angle_mse": 0.0,
        "style_mse": 0.0,
        "traj_mse": 0.0,
        "token_acc": 0.0,
    }
    total_batches = 0

    for batch in loader:
        outputs = model(
            batch["manifold_index"].to(device),
            batch["text_embed"].to(device),
            batch["lengths"].to(device),
            batch["angle_xy"].to(device).permute(0, 2, 1),
            batch["stylecode"].to(device).permute(0, 2, 1),
            batch["relative_trajectory"].to(device).permute(0, 2, 1),
        )
        for key in totals:
            totals[key] += outputs[key].item()
        total_batches += 1

    return {key: value / max(total_batches, 1) for key, value in totals.items()}


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

    train_loader = build_loader(train_dataset, args.batch_size, args.num_workers, True, args.target_length)
    test_loader = build_loader(test_dataset, args.batch_size, args.num_workers, False, args.target_length)

    codebook = load_codebook(args.pretrained_save)
    model = Text2PhaseMoMask(
        num_tokens=vq_args.num_embed_vq,
        code_dim=codebook.shape[1],
        latent_dim=args.latent_dim,
        ff_size=args.latent_dim * 4,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        text_dim=512,
        cond_drop_prob=args.cond_drop_prob,
        aux_loss_weights={
            "token": args.token_loss_weight,
            "angle": args.angle_loss_weight,
            "style": args.style_loss_weight,
            "traj": args.traj_loss_weight,
        },
    ).to(device)
    if args.freeze_token_emb:
        model.load_and_freeze_token_emb(codebook.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_config = vars(args).copy()
    run_config["model_config"] = model.get_config()
    with open(osp.join(args.save_dir, "momask_args.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    best_test_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {
            "loss": 0.0,
            "token_loss": 0.0,
            "angle_mse": 0.0,
            "style_mse": 0.0,
            "traj_mse": 0.0,
            "token_acc": 0.0,
        }
        total_batches = 0

        for step, batch in enumerate(train_loader, start=1):
            outputs = model(
                batch["manifold_index"].to(device),
                batch["text_embed"].to(device),
                batch["lengths"].to(device),
                batch["angle_xy"].to(device).permute(0, 2, 1),
                batch["stylecode"].to(device).permute(0, 2, 1),
                batch["relative_trajectory"].to(device).permute(0, 2, 1),
            )
            optimizer.zero_grad()
            outputs["loss"].backward()
            optimizer.step()

            for key in running:
                running[key] += outputs[key].item()
            total_batches += 1

            if step % args.log_interval == 0:
                print(
                    f"epoch[{epoch}] step[{step}/{len(train_loader)}] "
                    f"loss[{outputs['loss'].item():.6f}] "
                    f"token[{outputs['token_loss'].item():.6f}] "
                    f"acc[{outputs['token_acc'].item():.6f}] "
                    f"angle[{outputs['angle_mse'].item():.6f}] "
                    f"style[{outputs['style_mse'].item():.6f}] "
                    f"traj[{outputs['traj_mse'].item():.6f}]"
                )

        train_metrics = {key: value / max(total_batches, 1) for key, value in running.items()}
        test_metrics = evaluate(model, test_loader, device)
        print(
            f"epoch[{epoch}] "
            f"train_loss[{train_metrics['loss']:.6f}] "
            f"test_loss[{test_metrics['loss']:.6f}] "
            f"test_token[{test_metrics['token_loss']:.6f}] "
            f"test_acc[{test_metrics['token_acc']:.6f}] "
            f"test_angle[{test_metrics['angle_mse']:.6f}] "
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
