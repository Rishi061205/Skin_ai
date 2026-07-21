# app.py  — Flask API for skin disease detection
# Run with:  python app.py
# Test with: open http://localhost:5000 in your browser
#
# Endpoints:
#   GET  /          → serves the frontend HTML page
#   POST /predict   → accepts an image, returns JSON prediction
#   GET  /health    → confirms the API is running


import os
import io
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from torchvision import transforms

import config
from model import build_model

app = Flask(__name__)

# ── Load model once at startup ───────────────────────────────
# We load the model when the server starts — not on every request.
# Loading takes ~2 seconds. Inference takes ~50ms.
print("Loading model...")
checkpoint   = torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE, weights_only=False)
CLASS_TO_IDX = checkpoint["class_to_idx"]
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
NUM_CLASSES  = len(CLASS_TO_IDX)

model = build_model(num_classes=NUM_CLASSES, pretrained=False).to(config.DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print(f"Model ready — {NUM_CLASSES} classes on {config.DEVICE}")

# ── Image transform (same as val transform in dataset.py) ────
transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD),
])

# ── Inference helper ─────────────────────────────────────────
def predict_image(pil_img):
    tensor = transform(pil_img).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze()

    scores = {
        IDX_TO_CLASS[i]: round(probs[i].item() * 100, 2)
        for i in range(NUM_CLASSES)
    }
    top_class = max(scores, key=scores.get)
    top_conf  = scores[top_class]

    return {
        "prediction": top_class,
        "confidence": top_conf,
        "all_scores": dict(sorted(scores.items(), key=lambda x: -x[1])),
    }


# ── Routes ───────────────────────────────────────────────────
@app.route("/health")
def health():
    """Quick check that the API is running."""
    return jsonify({"status": "ok", "classes": NUM_CLASSES, "device": str(config.DEVICE)})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a POST request with an image file.
    Returns JSON with prediction + confidence scores.

    Example using curl:
      curl -X POST http://localhost:5000/predict \
           -F "image=@path/to/skin.jpg"
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Send a file with key 'image'"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result    = predict_image(pil_img)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    """Serves the frontend UI."""
    return render_template("index.html")

# ── Start server ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\nStarting server at http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    app.run(debug=False, host="0.0.0.0", port=5000)