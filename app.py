import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import wandb
import os
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Ultrasound Classifier",
    page_icon="🩺",
    layout="centered"
)

# ── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    h1 { font-family: 'IBM Plex Mono', monospace; color: #00e5ff; letter-spacing: -1px; }
    h3 { color: #90caf9; }

    .result-box {
        border-radius: 8px;
        padding: 20px 24px;
        margin-top: 16px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
    }
    .benign    { background: #0d3b2e; border: 1px solid #00e676; color: #00e676; }
    .normal    { background: #0d2a3b; border: 1px solid #00b0ff; color: #00b0ff; }
    .malignant { background: #3b0d0d; border: 1px solid #ff5252; color: #ff5252; }

    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        margin: 4px 0 12px 0;
    }
    .subtitle {
        color: #888;
        font-size: 0.9rem;
        margin-bottom: 24px;
    }
    .disclaimer {
        background: #1a1a2e;
        border-left: 3px solid #f39c12;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.82rem;
        color: #aaa;
        margin-top: 32px;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
LABEL_NAMES = ['benign', 'malignant', 'normal']
IMG_SIZE    = 128
MODEL_PATH  = "mobilenetv2_ultrasound.pt"

# W&B settings — update these to match your project
WANDB_ENTITY  = "YOUR_WANDB_USERNAME"   # ← replace with your wandb username
WANDB_PROJECT = "breast-ultrasound-classification"
WANDB_ARTIFACT = "ultrasound-model:latest"

# ── Load model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Download model from W&B artifacts if not already cached locally."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from W&B..."):
            api = wandb.Api()
            artifact = api.artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{WANDB_ARTIFACT}")
            artifact.download(root=".")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild MobileNetV2 architecture
    model = models.mobilenet_v2(weights=None)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[1]  = nn.Linear(model.last_channel, len(LABEL_NAMES))

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

# ── Preprocessing ──────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict(image: Image.Image, model, device):
    tensor = transform(image.convert("L")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    pred_idx = probs.argmax()
    return LABEL_NAMES[pred_idx], float(probs[pred_idx]), probs

# ── UI ─────────────────────────────────────────────────────────────────────
st.markdown("# 🩺 Breast Ultrasound Classifier")
st.markdown('<p class="subtitle">Upload a breast ultrasound image to classify it as <b>Normal</b>, <b>Benign</b>, or <b>Malignant</b>.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload ultrasound image (PNG or JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        model, device = load_model()
        label, confidence, all_probs = predict(image, model, device)

        color_class = label  # benign / malignant / normal
        st.markdown(f"""
            <div class="result-box {color_class}">
                Prediction: {label.upper()}<br/>
                Confidence: {confidence:.1%}
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### Class probabilities")
        bar_colors = {"benign": "#00e676", "malignant": "#ff5252", "normal": "#00b0ff"}
        for name, prob in zip(LABEL_NAMES, all_probs):
            st.markdown(f"**{name.capitalize()}** — {prob:.1%}")
            st.markdown(
                f'<div class="confidence-bar" style="width:{prob*100:.1f}%; background:{bar_colors[name]}"></div>',
                unsafe_allow_html=True
            )

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Research use only.</b> This tool is a demonstration of an MLOps pipeline
        and is not intended for clinical diagnosis. Always consult a qualified medical professional.
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Upload an ultrasound image above to get a prediction.")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    **Model:** MobileNetV2 (fine-tuned)  
    **Dataset:** Breast Ultrasound Images  
    **Classes:** Normal · Benign · Malignant  
    **Tracking:** Weights & Biases  
    """)
    st.markdown("---")
    st.markdown("Built as part of an MLOps pipeline tutorial.")