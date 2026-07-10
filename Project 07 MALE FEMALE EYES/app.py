import json
import os
import textwrap
 
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
 
# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
IMG_SIZE = (128, 128)  # must match training's IMG_SIZE
 
# Resolve paths relative to THIS script's folder, not the process's
# working directory (Streamlit Cloud's cwd is the repo root, which may
# not be the same folder app.py lives in if it's inside a subfolder).
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_CANDIDATES = [
    os.path.join(APP_DIR, "eye_gender_model.keras"),
    os.path.join(APP_DIR, "eye_gender_model.h5"),
]
CLASS_INDICES_PATH = os.path.join(APP_DIR, "class_indices.json")
 
st.set_page_config(page_title="Eye Classifier", page_icon="◉", layout="centered")
 
# ------------------------------------------------------------
# Styling — "optical scan" theme: deep slate background, warm
# iris-amber accent, cool sclera-cyan secondary, serif display
# type for the headline, monospace for the readout numbers.
# ------------------------------------------------------------
ACCENT = "#D98E3E"     # iris amber
ACCENT2 = "#6FA8B0"    # sclera cyan
TRACK = "#24343E"      # gauge track / borders
BG = "#101B24"
SURFACE = "#17242E"
TEXT = "#EDF4F3"
TEXT_MUTED = "#8FA3A8"
 
st.markdown(
    textwrap.dedent(
        f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
    <style>
      .stApp {{ background: radial-gradient(circle at 50% -10%, #16262F 0%, {BG} 55%); color: {TEXT}; }}
      [data-testid="stHeader"] {{ background: transparent; }}
      #MainMenu, footer {{ visibility: hidden; }}
      .irid-hero {{ text-align: center; padding: 1.2rem 0 0.4rem 0; }}
      .irid-ring {{ width: 64px; height: 64px; margin: 0 auto 1rem auto; border-radius: 50%; background: radial-gradient({ACCENT} 0%, {ACCENT} 28%, transparent 30%), conic-gradient({ACCENT2} 0deg 340deg, transparent 340deg 360deg); box-shadow: 0 0 24px rgba(217,142,62,0.35); }}
      .irid-title {{ font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.6rem; letter-spacing: 0.01em; margin: 0; color: {TEXT}; }}
      .irid-subtitle {{ font-family: 'Inter', sans-serif; color: {TEXT_MUTED}; font-size: 1rem; margin-top: 0.35rem; }}
      .irid-card {{ background: {SURFACE}; border: 1px solid {TRACK}; border-radius: 18px; padding: 1.6rem; margin-top: 1.4rem; }}
      [data-testid="stFileUploaderDropzone"] {{ background: {BG} !important; border: 1px dashed {TRACK} !important; border-radius: 14px !important; }}
      [data-testid="stFileUploaderDropzone"]:hover {{ border-color: {ACCENT2} !important; }}
      [data-testid="stImage"] img {{ border-radius: 14px; border: 1px solid {TRACK}; }}
      .irid-gauge-wrap {{ position: relative; width: 168px; height: 168px; margin: 0.4rem auto 0.8rem auto; }}
      .irid-gauge-hole {{ position: absolute; top: 14px; left: 14px; width: 140px; height: 140px; border-radius: 50%; background: {SURFACE}; display: flex; flex-direction: column; align-items: center; justify-content: center; }}
      .irid-gauge-pct {{ font-family: 'JetBrains Mono', monospace; font-size: 1.9rem; font-weight: 500; color: {TEXT}; }}
      .irid-gauge-label {{ font-family: 'Inter', sans-serif; font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase; color: {TEXT_MUTED}; margin-top: 0.15rem; }}
      .irid-result-label {{ font-family: 'Inter', sans-serif; font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: {TEXT_MUTED}; text-align: center; margin-bottom: 0.3rem; }}
      .irid-result-value {{ font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.6rem; text-align: center; color: {ACCENT}; margin-bottom: 0.4rem; }}
      [data-testid="stFileUploaderDropzoneInstructions"] small {{ display: none; }}
    </style>
    <div class="irid-hero">
      <div class="irid-ring"></div>
      <p class="irid-title">Eye Classifier</p>
      <p class="irid-subtitle">This app predicts whether it's male eyes or female eyes</p>
    </div>
    """
    ),
    unsafe_allow_html=True,
)
 
 
# ------------------------------------------------------------
# Cached loaders (so the model isn't reloaded on every interaction)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    for path in MODEL_PATH_CANDIDATES:
        if os.path.exists(path):
            return tf.keras.models.load_model(path)
 
    # Nothing found — show exactly what IS in the app folder to make
    # debugging a misplaced/missing file fast.
    try:
        found = os.listdir(APP_DIR)
    except Exception:
        found = ["<could not list APP_DIR>"]
 
    raise FileNotFoundError(
        f"No model file found. Expected one of {MODEL_PATH_CANDIDATES}.\n"
        f"APP_DIR = {APP_DIR}\n"
        f"Files actually present there: {found}"
    )
 
 
@st.cache_resource
def load_class_names():
    # class_indices.json looks like {"femaleeyes": 0, "maleeyes": 1}
    # We need it inverted -> {0: "femaleeyes", 1: "maleeyes"}
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH) as f:
            class_indices = json.load(f)
        return {v: k for k, v in class_indices.items()}
    # Fallback if the file wasn't uploaded alongside the model
    return {0: "femaleeyes", 1: "maleeyes"}
 
 
def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)  # shape: (1, H, W, 3)
 
 
def clean_label(raw_label: str) -> str:
    # turns "maleeyes" / "femaleeyes" into "Male" / "Female"
    raw_label = raw_label.lower()
    if "female" in raw_label:
        return "Female"
    if "male" in raw_label:
        return "Male"
    return raw_label
 
 
# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
try:
    model = load_model()
    idx_to_class = load_class_names()
except FileNotFoundError as e:
    st.error("Model loading failed:")
    st.code(str(e))
    st.stop()
 
st.markdown('<div class="irid-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose an eye image", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed"
)
 
if uploaded_file is not None:
    image = Image.open(uploaded_file)
 
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image(image, use_container_width=True)
 
    with st.spinner("Analyzing..."):
        x = preprocess_image(image)
        prob = float(model.predict(x, verbose=0)[0][0])  # sigmoid output, class "1"
 
        # class 1 probability = prob, class 0 probability = 1 - prob
        predicted_idx = 1 if prob >= 0.5 else 0
        confidence = prob if predicted_idx == 1 else 1 - prob
        predicted_label = clean_label(idx_to_class.get(predicted_idx, str(predicted_idx)))
 
    with col2:
        pct = confidence * 100
        deg = confidence * 360
        st.markdown(
            textwrap.dedent(
                f"""
            <p class="irid-result-label">Prediction</p>
            <p class="irid-result-value">{predicted_label}</p>
            <div class="irid-gauge-wrap" style="background: conic-gradient({ACCENT} {deg}deg, {TRACK} {deg}deg 360deg); border-radius: 50%;">
              <div class="irid-gauge-hole">
                <span class="irid-gauge-pct">{pct:.0f}%</span>
                <span class="irid-gauge-label">confidence</span>
              </div>
            </div>
            """
            ),
            unsafe_allow_html=True,
        )
st.markdown("</div>", unsafe_allow_html=True)
 
 
