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
from flask import Flask, request, jsonify, render_template_string
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
    """Serves a simple test UI so you can try the model in a browser."""
    return render_template_string(HTML_UI)


# ── Simple built-in test UI ──────────────────────────────────
# This lets you test the API directly in a browser without
# needing a separate frontend yet.


HTML_UI = """
<!DOCTYPE html>
<html>
<head>
  <title>Skin disease detector</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #f5f5f5; color: #111; padding: 2rem; }
    .card { background: white; border-radius: 12px; border: 1px solid #e5e5e5; padding: 1.5rem; max-width: 520px; margin: 0 auto; }
    h1 { font-size: 18px; font-weight: 500; margin-bottom: 4px; }
    p.sub { font-size: 13px; color: #666; margin-bottom: 1.5rem; }
    .upload { border: 1.5px dashed #ccc; border-radius: 8px; padding: 2rem; text-align: center; cursor: pointer; margin-bottom: 1rem; }
    .upload:hover { background: #fafafa; }
    input[type=file] { display: none; }
    img#preview { max-width: 100%; border-radius: 8px; margin: 1rem 0; display: none; }
    button { background: #111; color: white; border: none; padding: 10px 24px; border-radius: 8px; font-size: 14px; cursor: pointer; width: 100%; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .result { margin-top: 1.5rem; display: none; }
    .top { font-size: 22px; font-weight: 500; margin-bottom: 4px; }
    .conf { font-size: 13px; color: #666; margin-bottom: 1rem; }
    .bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
    .bar-name { font-size: 12px; min-width: 180px; }
    .bar-bg { flex: 1; height: 7px; background: #eee; border-radius: 4px; overflow: hidden; }
    .bar-fill { height: 100%; background: #111; border-radius: 4px; transition: width 0.6s ease; }
    .bar-pct { font-size: 12px; color: #666; min-width: 40px; text-align: right; }
    .error { color: #c0392b; font-size: 13px; margin-top: 1rem; }
    .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid #fff; border-top-color: transparent; border-radius: 50%; animation: spin 0.6s linear infinite; vertical-align: middle; margin-right: 6px; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
<div class="card">
  <h1>Skin disease detector</h1>
  <p class="sub">Upload a skin image to detect the disease</p>

  <label class="upload" for="file-input">
    <input type="file" id="file-input" accept="image/*" onchange="previewImage(event)">
    <p style="font-size:14px;font-weight:500;margin-bottom:4px">Click to upload image</p>
    <p style="font-size:12px;color:#999">JPG, PNG supported</p>
  </label>

  <img id="preview" src="" alt="Preview">

  <button id="btn" onclick="runPrediction()" disabled>Analyse image</button>
  <p id="error" class="error"></p>

  <div class="result" id="result">
    <p class="top" id="top-label"></p>
    <p class="conf" id="top-conf"></p>
    <div id="bars"></div>
  </div>
</div>

<script>
let selectedFile = null;

function previewImage(e) {
  selectedFile = e.target.files[0];
  if (!selectedFile) return;
  const reader = new FileReader();
  reader.onload = ev => {
    const img = document.getElementById('preview');
    img.src = ev.target.result;
    img.style.display = 'block';
  };
  reader.readAsDataURL(selectedFile);
  document.getElementById('btn').disabled = false;
  document.getElementById('result').style.display = 'none';
  document.getElementById('error').textContent = '';
}

async function runPrediction() {
  if (!selectedFile) return;
  const btn = document.getElementById('btn');
  btn.innerHTML = '<span class="spinner"></span>Analysing...';
  btn.disabled = true;

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const res  = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) {
      document.getElementById('error').textContent = 'Error: ' + data.error;
      return;
    }

    document.getElementById('top-label').textContent = data.prediction;
    document.getElementById('top-conf').textContent  = `Confidence: ${data.confidence.toFixed(1)}%`;

    const bars = document.getElementById('bars');
    bars.innerHTML = '';
    Object.entries(data.all_scores).forEach(([name, pct]) => {
      bars.innerHTML += `
        <div class="bar-row">
          <span class="bar-name">${name}</span>
          <div class="bar-bg"><div class="bar-fill" style="width:0%" data-w="${pct}"></div></div>
          <span class="bar-pct">${pct.toFixed(1)}%</span>
        </div>`;
    });

    requestAnimationFrame(() => requestAnimationFrame(() => {
      document.querySelectorAll('.bar-fill').forEach(b => b.style.width = b.dataset.w + '%');
    }));

    document.getElementById('result').style.display = 'block';

  } catch(e) {
    document.getElementById('error').textContent = 'Could not connect to server.';
  } finally {
    btn.textContent = 'Analyse image';
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""

# ── Start server ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\nStarting server at http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    app.run(debug=False, host="0.0.0.0", port=5000)