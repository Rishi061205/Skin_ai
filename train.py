

import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import build_model, get_optimizer, get_scheduler


# ── Helper: run one epoch ────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scheduler, phase):
    """
    phase = "train" → update weights
    phase = "val"   → only measure loss/accuracy, don't update weights
    """
    is_train = (phase == "train")
    model.train() if is_train else model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, desc=f"  {phase}", leave=False):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            logits = model(images)              # forward pass
            loss   = criterion(logits, labels)  # how wrong are we?

            if is_train:
                optimizer.zero_grad()           # clear old gradients
                loss.backward()                 # compute new gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()                # update weights
                scheduler.step()                # update learning rate

            preds          = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss    += loss.item() * images.size(0)
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy


# ── Early stopping ───────────────────────────────────────────
class EarlyStopping:
    """
    Watches validation loss. If it doesn't improve for `patience`
    consecutive epochs, it signals to stop training.
    Also saves the best model checkpoint automatically.
    """
    def __init__(self, patience, path):
        self.patience   = patience
        self.path       = path
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss, model, class_to_idx=None):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "class_to_idx": class_to_idx,
            }
            torch.save(checkpoint, self.path)
            print(f"    ✓ New best val loss: {val_loss:.4f} — checkpoint saved")
        else:
            self.counter += 1
            print(f"    No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True


# ── Main training function ───────────────────────────────────
def train():
    print(f"\nDevice: {config.DEVICE}")
    print("Loading data...\n")

    train_loader, val_loader, _, label2idx = get_dataloaders()

    num_classes = len(label2idx)
    print(f"Building model for {num_classes} classes...")
    model = build_model(num_classes=num_classes).to(config.DEVICE)

    # Loss function: CrossEntropyLoss is standard for multi-class classification.
    # label_smoothing=0.1 prevents the model from becoming overconfident —
    # it says "be 90% sure, not 100%" which generalizes better.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, steps_per_epoch=len(train_loader))

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    early_stop = EarlyStopping(
        patience = config.PATIENCE,
        path     = config.CHECKPOINT_PATH,
    )

    print(f"\nStarting training for up to {config.NUM_EPOCHS} epochs...\n")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()
        print(f"Epoch [{epoch}/{config.NUM_EPOCHS}]")

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scheduler, "train"
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, None, "val"
        )

        elapsed = time.time() - t0
        print(f"  Train — Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val   — Loss: {val_loss:.4f}   | Acc: {val_acc:.2f}%  [{elapsed:.0f}s]")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        early_stop.step(val_loss, model, label2idx)
        if early_stop.should_stop:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    print(f"\nTraining complete. Best val loss: {early_stop.best_loss:.4f}")
    print(f"Best model saved at: {config.CHECKPOINT_PATH}")

    _save_training_plot(history)
    return history


def _save_training_plot(history):
    """Saves a loss/accuracy curve plot to checkpoints/training_curves.png"""
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history["train_loss"], label="Train Loss")
        ax1.plot(history["val_loss"],   label="Val Loss")
        ax1.set_title("Loss over epochs")
        ax1.set_xlabel("Epoch")
        ax1.legend()

        ax2.plot(history["train_acc"], label="Train Acc")
        ax2.plot(history["val_acc"],   label="Val Acc")
        ax2.set_title("Accuracy over epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("%")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"{config.CHECKPOINT_DIR}/training_curves.png", dpi=150)
        print(f"Training curves saved to {config.CHECKPOINT_DIR}/training_curves.png")
    except Exception as e:
        print(f"Could not save plot: {e}")


if __name__ == "__main__":
    train()