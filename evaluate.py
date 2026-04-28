

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import config
from dataset import get_dataloaders
from model import build_model


def evaluate():
    print(f"\nDevice: {config.DEVICE}")
    print("Loading data and best model checkpoint...\n")

    _, _, test_loader, _ = get_dataloaders()

    # Load the best model checkpoint (includes class mapping)
    checkpoint   = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    class_to_idx = checkpoint["class_to_idx"]
    num_classes  = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_display_names = [idx_to_class[i] for i in range(num_classes)]

    model = build_model(num_classes=num_classes, pretrained=False).to(config.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(config.DEVICE)
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Overall accuracy 
    accuracy = (all_preds == all_labels).mean() * 100
    print(f"Test Accuracy: {accuracy:.2f}%\n")

    print("Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names = class_display_names,
        digits       = 3,
    ))

    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, class_display_names)


def _plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot       = True,
        fmt         = "d",
        cmap        = "Blues",
        xticklabels = class_names,
        yticklabels = class_names,
        ax          = ax,
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label",      fontsize=12)
    ax.set_title("Confusion Matrix — Test Set", fontsize=14)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = f"{config.CHECKPOINT_DIR}/confusion_matrix.png"
    plt.savefig(out, dpi=150)
    print(f"\nConfusion matrix saved to {out}")
    plt.show()


if __name__ == "__main__":
    evaluate()