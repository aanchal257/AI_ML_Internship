import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------------------------------------------------
# Config — must match what the model was trained on
# ---------------------------------------------------------
IMG_WIDTH, IMG_HEIGHT = 150, 150

# Dynamically get the folder where app.py lives, then attach the model filename
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "gender_image_classifier.h5")

st.set_page_config(
    page_title="Male vs Female Classifier",
    page_icon="🧑",
    layout="centered",
)


# ---------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ---------------------------------------------------------
@st.cache_resource
def get_model():
    if not os.path.isfile(MODEL_PATH):
        return None, f"Model file not found at '{MODEL_PATH}'."
    try:
        model = load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------
# Preprocessing (matches training: resize -> RGB -> /255)
# ---------------------------------------------------------
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    pil_img = pil_img.convert("RGB")  # handles grayscale/PNG safely
    pil_img = pil_img.resize((IMG_WIDTH, IMG_HEIGHT))
    x = np.array(pil_img, dtype=np.float32)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)  # add batch dimension
    return x


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🧑 Male vs Female Classifier 👩")
st.write("Upload a face image to get a prediction.")

model, load_error = get_model()

if load_error:
    st.error(
        "Could not load the model.\n\n"
        f"Details: {load_error}\n\n"
        "Make sure `gender_image_classifier.h5` is in the same folder as `app.py`."
    )
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Classifying..."):
            x = preprocess_image(img)
            predictions = model.predict(x)
            pred = float(predictions[0][0])

        # class_mode='binary' with flow_from_directory assigns classes
        # alphabetically -> female=0, male=1
        if pred >= 0.5:
            label = "Male 🧑"
            confidence = pred
        else:
            label = "Female 👩"
            confidence = 1 - pred

        st.success(f"Prediction: **{label}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
        st.progress(min(max(confidence, 0.0), 1.0))

        with st.expander("Raw model output"):
            st.write("predictions:", predictions.tolist())
            st.write("predictions.shape:", predictions.shape)
            st.caption("Score close to 0 → Female, score close to 1 → Male.")

    except Exception as e:
        st.error(f"Something went wrong while processing the image: {e}")
else:
    st.info("👆 Upload a JPG or PNG image to get a prediction.")

st.markdown("---")
st.caption(
    "Model: CNN (3× Conv2D + MaxPooling, Dense(128) + Dropout, sigmoid output), "
    "150×150 RGB input, trained for binary male/female classification."
)
st.caption(
    "⚠️ Note: gender-classification models can be biased by their training data "
    "and may not perform equally well across all ages, ethnicities, or presentations. "
    "Predictions should be treated as approximate, not authoritative."
)
