from pathlib import Path
from tqdm import tqdm

import torch

from utils.metrics import compute_metrics


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    all_true = []
    all_pred = []
    all_prob = []

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        total_loss += loss.item() * images.size(0)

        all_true.extend(labels.detach().cpu().numpy())
        all_pred.extend(preds.detach().cpu().numpy())
        all_prob.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_true, all_pred, all_prob)
    metrics["loss"] = avg_loss

    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, phase="Validation"):
    model.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []
    all_prob = []

    for images, labels in tqdm(loader, desc=phase, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        total_loss += loss.item() * images.size(0)

        all_true.extend(labels.detach().cpu().numpy())
        all_pred.extend(preds.detach().cpu().numpy())
        all_prob.extend(probs.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_true, all_pred, all_prob)
    metrics["loss"] = avg_loss

    return metrics


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    checkpoint_dir,
    model_name,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1
    history = []

    for epoch in range(1, epochs + 1):
        print(f"\n[EPOCH {epoch}/{epochs}]")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            phase="Validation",
        )

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }

        history.append(row)

        print(f"Train: {train_metrics}")
        print(f"Val:   {val_metrics}")

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]

            save_path = checkpoint_dir / f"{model_name}_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_f1": best_f1,
                    "val_metrics": val_metrics,
                },
                save_path,
            )

            print(f"[SAVED] Best model saved to {save_path}")

    return history
