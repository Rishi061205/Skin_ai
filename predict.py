
import argparse
import numpy as np
from PIL import Image
import torch

import config
from dataset import get_val_transforms
from model import build_model


def predict_single(img_path: str) -> dict:
    """
    Takes a path to a skin image.
    Returns a dict with the top prediction and all class probabilities.
    """
    # 1. Load checkpoint (includes class names + model weights)
    checkpoint   = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes  = len(class_to_idx)

    # 2. Load model
    model = build_model(num_classes=num_classes, pretrained=False).to(config.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 3. Load and preprocess image
    from PIL import Image as PILImage
    pil_img = PILImage.open(img_path).convert("RGB")
    tf      = get_val_transforms()
    tensor  = tf(pil_img).unsqueeze(0).to(config.DEVICE)

    # 4. Run inference
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze()

    # 5. Build output
    scores = {
        idx_to_class[i]: round(probs[i].item() * 100, 2)
        for i in range(num_classes)
    }
    top_class = max(scores, key=scores.get)
    top_conf  = scores[top_class]

    return {
        "prediction": top_class,
        "confidence": top_conf,
        "all_scores": dict(sorted(scores.items(), key=lambda x: -x[1])),
    }


def main():
    parser = argparse.ArgumentParser(description="Skin disease predictor")
    parser.add_argument("--image", required=True, help="Path to skin image (JPG/PNG)")
    args = parser.parse_args()

    print(f"\nAnalyzing: {args.image}\n")
    result = predict_single(args.image)

    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']}%\n")
    print("All class scores:")
    for disease, score in result["all_scores"].items():
        bar = "█" * int(score / 5)
        print(f"  {disease:<28} {score:>6.2f}%  {bar}")

    print("\n⚠  This tool is for educational purposes only.")
    print("   Consult a certified dermatologist for any medical concerns.")


if __name__ == "__main__":
    main()