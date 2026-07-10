import json
import os
 
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
 
# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
IMG_SIZE = (128, 128)  # must match training's IMG_SIZE
MODEL_PATH_CANDIDATES = ["eye_gender_model.keras", "eye_gender_model.h5"]
CLASS_INDICES_PATH = "class_indices.json"
 
st.set_page_config(page_title="Eye Gender Classifier", page_icon="👁️", layout="centered")
 
 
# ------------------------------------------------------------
# Cached loaders (so the model isn't reloaded on every interaction)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    for path in MODEL_PATH_CANDIDATES:
        if os.path.exists(path):
            return tf.keras.models.load_model(path)
    raise FileNotFoundError(
        f"No model file found. Expected one of {MODEL_PATH_CANDIDATES} "
        "in the app's working directory."
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
st.title("👁️ Eye Gender Classifier")
st.write(
    "Upload a close-up photo of a human eye and the model will predict "
    "whether it belongs to a male or female, based on a CNN trained on "
    "the [Eyes RTTE dataset](https://www.kaggle.com/datasets/pavelbiz/eyes-rtte)."
)
 
with st.sidebar:
    st.header("About")
    st.markdown(
        "- Model: custom CNN (binary classifier)\n"
        f"- Input size: {IMG_SIZE[0]}x{IMG_SIZE[1]}\n"
        "- Trained in Google Colab, deployed with Streamlit"
    )
 
try:
    model = load_model()
    idx_to_class = load_class_names()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
 
uploaded_file = st.file_uploader(
    "Choose an eye image", type=["jpg", "jpeg", "png", "bmp"]
)
 
if uploaded_file is not None:
    image = Image.open(uploaded_file)
 
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)
 
    with st.spinner("Analyzing..."):
        x = preprocess_image(image)
        prob = float(model.predict(x, verbose=0)[0][0])  # sigmoid output, class "1"
 
        # class 1 probability = prob, class 0 probability = 1 - prob
        predicted_idx = 1 if prob >= 0.5 else 0
        confidence = prob if predicted_idx == 1 else 1 - prob
        predicted_label = clean_label(idx_to_class.get(predicted_idx, str(predicted_idx)))
 
    with col2:
        st.subheader("Prediction")
        st.metric(label="Predicted class", value=predicted_label)
        st.metric(label="Confidence", value=f"{confidence * 100:.1f}%")
        st.progress(confidence)
 
    st.caption(
        "Note: this model reflects patterns in its training data only "
        "and should not be treated as a reliable or fair predictor of "
        "gender in any real-world or sensitive application."
    )
else:
    st.info("Upload an image to get a prediction.")
 
