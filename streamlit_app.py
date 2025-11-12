import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from huggingface_hub import hf_hub_download
from pathlib import Path
from PIL import Image
import json, cv2, numpy as np

st.set_page_config(page_title="Wafer Defect Detection (Grad-CAM)", layout="wide")
st.title("🧪 Wafer Defect Detection — Vision SMAI (Grad-CAM XAI)")
st.caption("Upload a wafer map image to get class prediction and an explainability heatmap.")

# -----------------------------
# Config
# -----------------------------
HF_REPO = "EdmondChong/WaferDefectDetection"
MODEL_FILENAME = "wafer_efficientnet_b0_finetuned.pth"
LABELS_FILENAME = "labels.json"

# -----------------------------
# Download artifacts
# -----------------------------
@st.cache_resource
def fetch_artifacts():
    model_path = hf_hub_download(
        repo_id=HF_REPO, filename=MODEL_FILENAME, token=st.secrets["HF_TOKEN"]
    )
    labels = None
    try:
        labels_path = hf_hub_download(
            repo_id=HF_REPO, filename=LABELS_FILENAME, use_auth_token=st.secrets["HF_TOKEN"]
        )
        with open(labels_path, "r") as f:
            labels = json.load(f)
    except Exception:
        pass
    return model_path, labels

model_path, class_names = fetch_artifacts()

# -----------------------------
# Build model & load weights
# -----------------------------
class WaferDefectNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

@st.cache_resource
def load_model(model_path, labels):
    state_dict = torch.load(model_path, map_location="cpu")
    if "base_model.classifier.1.weight" in state_dict:
        num_classes = state_dict["base_model.classifier.1.weight"].shape[0]
    else:
        key = [k for k in state_dict.keys() if k.endswith("classifier.1.weight")][0]
        num_classes = state_dict[key].shape[0]

    model = WaferDefectNet(num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Force correct wafer class order
    classes = [
        "Center",
        "Edge-Loc",
        "Edge-Ring",
        "Loc",
        "None",
        "Scratch"
    ]
    return model, classes

model, CLASSES = load_model(model_path, class_names)

# -----------------------------
# Preprocessing
# -----------------------------
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
base_tf = weights.transforms()
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    base_tf
])

# -----------------------------
# Grad-CAM class
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        def fwd_hook(module, inp, out):
            self.activations = out

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer.register_forward_hook(fwd_hook)
        target_layer.register_backward_hook(bwd_hook)

    def __call__(self, x, class_idx=None):
        x = x.requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.detach().cpu().numpy(), class_idx

target_layer = model.base_model.features[-1]
grad_cam = GradCAM(model, target_layer)

# -----------------------------
# UI
# -----------------------------
uploaded = st.file_uploader("📤 Upload wafer map image (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Wafer Image", width=250)

    if st.button("Predict & Explain"):
        with st.spinner("Running inference…"):
            ipt = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                logits = model(ipt)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = int(np.argmax(probs))
                pred_name = CLASSES[pred_idx]
                conf = float(probs[pred_idx])

            st.subheader("Prediction")
            st.write(f"**Defect Class:** {pred_name}")
            st.write(f"**Confidence:** {conf:.2f}")

            # -----------------------------
            # Grad-CAM visualization (reversed colormap)
            # -----------------------------
            cam, _ = grad_cam(ipt, class_idx=pred_idx)
            cam = cam[0]
            cam = np.maximum(cam, 0)
            cam = cam / (cam.max() + 1e-8)

            heatmap = np.uint8(255 * cam)
            heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))

            # 🔥 Reverse the color map LUT so red/yellow = high attention
            lut = cv2.applyColorMap(np.arange(0, 256, dtype=np.uint8), cv2.COLORMAP_JET)
            lut = np.flip(lut, axis=0).copy()
            heatmap = cv2.LUT(heatmap, lut)

            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(np.array(img.convert("RGB")), 0.6, heatmap, 0.4, 0)

            st.write("### Visualization")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(img, caption="Original Wafer", width=300)
            with col2:
                st.image(overlay, caption="Grad-CAM Heatmap", width=300)

            st.caption("Note: Heatmap highlights spatial regions most influential for the predicted class (red/yellow = high importance).")
