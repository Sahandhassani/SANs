import argparse
from pathlib import Path

import torch.nn as nn
from torch.utils.data import DataLoader

from utils.dataset import KermanyBinaryDataset
from utils.trainer import fit
from utils.other import set_seed, get_device, save_json
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="kermany")
    parser.add_argument("--data_root", type=str, default="Data/raw/Kermany2018")

    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--result_dir", type=str, default="outputs/results")

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)
    device = get_device()

    print(f"[DEVICE] {device}")
    print(f"[MODEL] {args.model}")

    if args.dataset.lower() == "kermany":
        train_dataset = KermanyBinaryDataset(
            root_dir=args.data_root,
            split="train",
            image_size=args.image_size,
        )

        val_dataset = KermanyBinaryDataset(
            root_dir=args.data_root,
            split="test",
            image_size=args.image_size,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(
        model_name=args.model,
        num_classes=2,
        pretrained=args.pretrained,
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = __import__("torch").optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model,
    )

    result_path = Path(args.result_dir) / f"{args.model}_seed{args.seed}_history.json"
    save_json(history, result_path)

    print(f"[DONE] Results saved to {result_path}")


if __name__ == "__main__":
    main()
