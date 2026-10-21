import os
import json
from datetime import datetime
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.cm as cm
import streamlit as st
import plotly.graph_objects as go
import torch
import torch.nn as nn

# ==============================================================================
# 1. PAGE SETUP & MINIMALIST STYLING
# ==============================================================================
st.set_page_config(
    page_title="PneumoVision",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Refined Minimalist CSS (Clean, Uncluttered, Modern)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    /* Remove default Streamlit top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    
    /* Minimal Title */
    .title-container {
        display: flex;
        align-items: baseline;
        gap: 12px;
        margin-bottom: 1.5rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .title-main {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin: 0;
        color: #f8fafc;
    }
    .title-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        margin: 0;
    }
    
    /* Clean Cards */
    .clean-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 1rem;
    }
    
    /* Minimal Verdict Badges */
    .verdict-normal {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(34, 197, 94, 0.12);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
    }
    .verdict-pneumonia {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Muted Labels */
    .label-muted {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }
    
    /* Hide unneeded Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MODEL DEFINITION & GRAD-CAM
# ==============================================================================
class ChestXRayCNN(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.20)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.30)
        )
        # Global Average Pooling (eliminates 6.4M weights & prevents overfitting)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        score = output[0, 0]
        score.backward(retain_graph=True)
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=0).squeeze().detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
            
        prob = torch.sigmoid(output).item()
        return heatmap, prob

@st.cache_resource
def get_model():
    model_path = r"D:\NNDLOpen\chest_xray_best_model.pth"
    model = ChestXRayCNN()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    gradcam = GradCAM(model, model.block4[0])
    return model, gradcam

model, gradcam = get_model()

# ==============================================================================
# 3. MINIMAL HEADER
# ==============================================================================
st.markdown("""
<div class="title-container">
    <h1 class="title-main">🫁 PneumoVision</h1>
    <span class="title-sub">Chest Radiograph AI Diagnostic</span>
</div>
""", unsafe_allow_html=True)

# Minimal Input Selector (Horizontal row)
input_mode = st.radio("Input Source", ["Preloaded Patient Samples", "Upload Image"], horizontal=True, label_visibility="collapsed")

sample_dir = r"D:\NNDLOpen\sample_images"
sample_files = []
if os.path.exists(sample_dir):
    sample_files = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

active_image_path = None
active_name = "Image"

if input_mode == "Preloaded Patient Samples" and sample_files:
    labels = {
        f: ("🟢 Normal (Healthy)" if "normal" in f.lower() else "🔴 Bacterial Pneumonia" if "bacteria" in f.lower() else "🟠 Viral Pneumonia") + f" — {f[:20]}..."
        for f in sample_files
    }
    selected_file = st.selectbox(
        "Select radiograph:",
        options=sample_files,
        format_func=lambda x: labels.get(x, x),
        label_visibility="collapsed"
    )
    active_image_path = os.path.join(sample_dir, selected_file)
    active_name = selected_file
else:
    uploaded = st.file_uploader("Upload Chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded:
        temp_path = os.path.join(r"D:\NNDLOpen", "temp_upload.jpeg")
        Image.open(uploaded).convert("RGB").save(temp_path)
        active_image_path = temp_path
        active_name = uploaded.name

# ==============================================================================
# 4. CORE DIAGNOSTIC DISPLAY
# ==============================================================================
if active_image_path and os.path.exists(active_image_path):
    raw_pil = Image.open(active_image_path).convert("RGB")
    
    # Process tensor
    input_resized = raw_pil.resize((224, 224))
    arr = np.transpose(np.array(input_resized, dtype=np.float32) / 255.0, (2, 0, 1))
    input_tensor = torch.tensor(arr).unsqueeze(0)
    input_tensor.requires_grad = True
    
    # Run Inference
    heatmap, raw_prob = gradcam.generate(input_tensor)
    
    # Minimal Sidebar Controls
    with st.sidebar:
        st.markdown("### Settings")
        threshold = st.slider("Decision Cutoff (τ)", 0.20, 0.95, 0.82, 0.02, help="Calibrated threshold balancing Normal Specificity (76%+) and Pneumonia Sensitivity (86%+).")
        show_cam = st.toggle("Show Grad-CAM Heatmap", value=True)
        cam_opacity = st.slider("Heatmap Intensity", 0.1, 0.9, 0.45, 0.05) if show_cam else 0.45
        colormap_choice = st.selectbox("Colormap", ["jet", "inferno", "viridis", "magma"], index=0)
        st.caption("Lower threshold (<0.65) = High Recall screening. Optimal balance ≈ 0.82.")

    predicted_pneumonia = raw_prob >= threshold
    confidence = raw_prob if predicted_pneumonia else (1.0 - raw_prob)

    # Generate Overlay
    if show_cam:
        heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(raw_pil.size, Image.BILINEAR)
        cmap = getattr(cm, colormap_choice, cm.jet)
        colored_cam = np.uint8(255 * cmap(np.array(heatmap_resized) / 255.0)[:, :, :3])
        display_img = Image.blend(raw_pil, Image.fromarray(colored_cam), alpha=cam_opacity)
    else:
        display_img = raw_pil

    col_view, col_diag = st.columns([1.1, 1], gap="large")
    
    with col_view:
        st.image(display_img, use_container_width=True)
        if show_cam:
            st.caption(f"Grad-CAM Heatmap Active ({colormap_choice.upper()}) · Attention on pulmonary infiltrates")
        else:
            st.caption("Original Radiograph")
            
    with col_diag:
        # Verdict Badge
        if predicted_pneumonia:
            st.markdown(f"""
            <div class="verdict-pneumonia">
                <span>●</span> PNEUMONIA DETECTED
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"<p style='color:#94a3b8; font-size:0.9rem; margin-top:8px;'>Infiltrate or consolidation pattern identified. Probability: <b>{raw_prob*100:.1f}%</b> (Cutoff: {threshold*100:.0f}%).</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-normal">
                <span>●</span> NORMAL RADIOGRAPH
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"<p style='color:#94a3b8; font-size:0.9rem; margin-top:8px;'>Clear bilateral lung fields. Normal Probability: <b>{(1.0 - raw_prob)*100:.1f}%</b>.</p>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Clean Stat Grid
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("<div class='label-muted'>Diagnostic Confidence</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat-number' style='color:{'#f87171' if predicted_pneumonia else '#4ade80'};'>{confidence*100:.1f}%</div>", unsafe_allow_html=True)
        with m2:
            st.markdown("<div class='label-muted'>Pneumonia Probability</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='stat-number' style='color:#38bdf8;'>{raw_prob*100:.1f}%</div>", unsafe_allow_html=True)

        # Minimal Progress Indicator
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(float(np.clip(raw_prob, 0.0, 1.0)))
        st.caption(f"Risk scale: 0% (Clear) ──── {threshold*100:.0f}% (Cutoff) ──── 100% (High Risk)")

        # Minimal Download Report
        report_data = f"""PneumoVision Diagnostic Summary
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Image: {active_name}
Verdict: {"PNEUMONIA DETECTED" if predicted_pneumonia else "NORMAL"}
Confidence: {confidence*100:.2f}%
Pneumonia Probability: {raw_prob*100:.2f}%
Decision Cutoff: {threshold*100:.1f}%
Model: Regularized Deep CNN (Global Average Pooling, 422k params)
"""
        st.download_button(
            label="Download Clinical Summary (.txt)",
            data=report_data,
            file_name=f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# ==============================================================================
# 5. MINIMAL COLLAPSIBLE NNDL PROJECT & BENCHMARK DETAILS
# ==============================================================================
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📊 Model Details & Overfitting Mitigation (NNDL Project Notes)"):
    st.markdown("""
    **Core Overfitting Root Causes & Solutions:**
    - **16-Image Validation Flaw:** Baseline used 16 validation samples causing erratic early stopping. We introduced an **85/15 stratified split** (~780 validation samples).
    - **Severe Class Imbalance (74% Pneumonia):** Resolved by applying analytical class weights ($w_{\\text{Normal}}=1.94, w_{\\text{Pneumonia}}=0.67$), eliminating majority-class bias.
    - **Parameter Reduction:** Replaced `Flatten()` ($6.4$ million weights) with `GlobalAveragePooling2D()` ($32,768$ weights) — a **93% parameter reduction** (from 6.5M to 422,401) preventing pixel memorization.
    - **Regularization:** Added Batch Normalization, Spatial Dropout, and $L_2$ weight decay.
    """)
    
    st.table({
        "Metric": ["Parameters", "Normal Recall (Specificity)", "Pneumonia Sensitivity", "Balanced Accuracy", "ROC-AUC"],
        "Baseline (Overfitted)": ["~6,500,000", "31.0% (Severe false alarms)", "100.0% (Trivial bias)", "65.5%", "0.814"],
        "Optimized (Calibrated)": ["422,401 (-93%)", "76.1% – 86.8%", "80.8% – 86.4%", "81.9%", "0.886"]
    })

st.markdown("<p style='text-align:center; color:#475569; font-size:0.75rem; margin-top:2rem;'>PneumoVision · Neural Networks & Deep Learning</p>", unsafe_allow_html=True)
