"""
Interactive Metal Surface Defect Inspection Dashboard (CSE411 Team 6).
Streamlit Application for Real-Time Manufacturing Quality Control.
"""

from pathlib import Path
import json
import time
import cv2
import numpy as np
import streamlit as st
import torch

from src.deployment.dashboard import load_inspection_model, run_defect_inspection
from src.dataset.parser import CLASS_NAMES


st.set_page_config(
    page_title="Metal Surface Defect Inspection Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1e293b; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #64748b; margin-bottom: 1.5rem; }
    .metric-card { background-color: #f8fafc; border-radius: 8px; padding: 12px; border: 1px solid #e2e8f0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Header ---
st.markdown('<div class="main-title">🔬 Metal Surface Defect Detection & Quality Control</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">CSE411 Computer Vision — Team 6 (Joel Joseph Mathews, Karthik Das P, Vivek Binod, Vibhaas Nirantar Srivastava)</div>',
    unsafe_allow_html=True,
)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Inspection Configuration")

model_choice = st.sidebar.selectbox(
    "Defect Detection Model",
    options=["M4 (Proposed Integrated Model)", "M1 (Plain YOLOv5s Baseline)", "M2 (CLAHE + Bilateral)", "M3 (ECA Attention)"],
    index=0,
)
variant_code = model_choice.split()[0]

conf_thres = st.sidebar.slider(
    "Detection Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
    help="Minimum class probability required to report defect bounding box.",
)

iou_thres = st.sidebar.slider(
    "NMS IoU Threshold",
    min_value=0.10,
    max_value=0.80,
    value=0.45,
    step=0.05,
    help="IoU threshold for non-maximum suppression deduplication.",
)

st.sidebar.markdown("---")
st.sidebar.header("🔍 Visual Explainability & Preprocessing")
show_preprocessing = st.sidebar.checkbox("Show Preprocessed View (CLAHE + Bilateral)", value=True)
show_gradcam = st.sidebar.checkbox(
    "Enable Grad-CAM Visual Attribution (Diagnostic Mode)",
    value=False,
    help="Runs neural gradient backpropagation for visual heatmaps. Keep disabled for maximum real-time manufacturing FPS.",
)

colormap_name = st.sidebar.selectbox(
    "Grad-CAM Colormap",
    options=["JET", "VIRIDIS", "HOT", "INFERNO"],
    index=0,
)
colormap_dict = {
    "JET": cv2.COLORMAP_JET,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "HOT": cv2.COLORMAP_HOT,
    "INFERNO": cv2.COLORMAP_INFERNO,
}
cam_colormap = colormap_dict[colormap_name]
cam_alpha = st.sidebar.slider("Heatmap Blending Opacity", 0.1, 0.9, 0.55, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("📁 Image Ingestion")

# Sample images discovery
sample_dir = Path("data/NEU-DET/test/images")
sample_files = sorted(list(sample_dir.glob("*.jpg"))) if sample_dir.exists() else []

input_mode = st.sidebar.radio("Input Source", options=["Sample Dataset Image", "Upload Custom Image"])

selected_image_path = None
uploaded_file = None

if input_mode == "Sample Dataset Image" and sample_files:
    sample_options = {f.name: f for f in sample_files[:60]}  # Representative sample
    chosen_sample = st.sidebar.selectbox("Select Sample Defect", options=list(sample_options.keys()))
    selected_image_path = sample_options[chosen_sample]
elif input_mode == "Upload Custom Image":
    uploaded_file = st.sidebar.file_uploader("Upload Steel Surface Image", type=["jpg", "jpeg", "png", "bmp"])

# --- Load Image ---
image_rgb = None
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is not None:
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
elif selected_image_path is not None:
    bgr = cv2.imread(str(selected_image_path))
    if bgr is not None:
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

if image_rgb is None:
    # Synthetic fallback sample
    image_rgb = (np.random.normal(120, 20, (200, 200, 3)).clip(0, 255)).astype(np.uint8)

# --- Load Model and Execute Inspection ---
@st.cache_resource(show_spinner="Loading Inspection Model...")
def get_cached_model(var: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_inspection_model(variant=var, device=device)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_cached_model(variant_code)

results = run_defect_inspection(
    image_rgb=image_rgb,
    model=model,
    conf_thres=conf_thres,
    iou_thres=iou_thres,
    generate_gradcam=show_gradcam,
    cam_alpha=cam_alpha,
    cam_colormap=cam_colormap,
    device=device,
)

telemetry = results["telemetry"]
detections = results["detections"]

# --- Live Telemetry KPIs ---
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
col_kpi1.metric("Defects Detected", f"{results['num_defects']}", delta=f"{'FAIL' if results['num_defects'] > 0 else 'PASS'}")
col_kpi2.metric(
    "Inference Latency",
    f"{telemetry['infer_latency_ms']:.1f} ms",
    delta=f"{telemetry['total_latency_ms']:.1f} ms total pipeline",
)
col_kpi3.metric(
    "Throughput",
    f"{telemetry.get('infer_fps', telemetry['fps']):.0f} FPS",
    delta="Real-Time (>30 FPS)" if telemetry.get("infer_fps", telemetry["fps"]) >= 30.0 else "CPU Bound",
)
col_kpi4.metric("Active Hardware", f"{telemetry['device'].upper()}", delta=f"Model: {variant_code}")

st.markdown("---")

# --- Visual Inspection Layout ---
img_cols = []
if show_preprocessing and show_gradcam:
    c1, c2, c3, c4 = st.columns(4)
    img_cols = [(c1, "Raw Steel Surface", results["raw_image"]),
                (c2, "CLAHE + Bilateral Enhanced", results["preprocessed_image"]),
                (c3, "Defect Localizations (NMS)", results["annotated_image"]),
                (c4, f"Grad-CAM Saliency ({colormap_name})", results["gradcam_overlay"])]
elif show_preprocessing:
    c1, c2, c3 = st.columns(3)
    img_cols = [(c1, "Raw Steel Surface", results["raw_image"]),
                (c2, "CLAHE + Bilateral Enhanced", results["preprocessed_image"]),
                (c3, "Defect Localizations (NMS)", results["annotated_image"])]
elif show_gradcam:
    c1, c2, c3 = st.columns(3)
    img_cols = [(c1, "Raw Steel Surface", results["raw_image"]),
                (c2, "Defect Localizations (NMS)", results["annotated_image"]),
                (c3, f"Grad-CAM Saliency ({colormap_name})", results["gradcam_overlay"])]
else:
    c1, c2 = st.columns(2)
    img_cols = [(c1, "Raw Steel Surface", results["raw_image"]),
                (c2, "Defect Localizations (NMS)", results["annotated_image"])]

for col, title, img in img_cols:
    with col:
        st.subheader(title)
        st.image(img, width="stretch")

# --- Defect Log Table ---
st.markdown("---")
st.subheader("📋 Defect Log & Telemetry Audit")

if detections:
    log_data = []
    for d in detections:
        box = d["box_xyxy"]
        log_data.append({
            "Defect Category": d["class_name"].replace("_", " ").title(),
            "Confidence": f"{d['confidence'] * 100:.2f}%",
            "Bounding Box (X1, Y1, X2, Y2)": f"[{box[0]}, {box[1]}, {box[2]}, {box[3]}]",
            "Severity": "Critical" if d["confidence"] > 0.6 else "Moderate",
        })
    st.dataframe(log_data, width="stretch")

    report_json = json.dumps({
        "timestamp": time.time(),
        "model": variant_code,
        "telemetry": telemetry,
        "detections": detections,
    }, indent=2)
    st.download_button(
        label="📥 Download Defect Inspection Audit (JSON)",
        data=report_json,
        file_name="defect_inspection_report.json",
        mime="application/json",
    )
else:
    st.success("✅ No defects identified above the confidence threshold. Component passes quality control.")
