
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Page settings
st.set_page_config(
    page_title="Crop Disease Classifier",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Crop Disease Classifier")
st.write("Upload a tomato leaf image to predict the disease.")

# Model path
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "crop_disease_model.keras")

# Load model
model = tf.keras.models.load_model(model_path)

# Class names
class_names = [
    "Bacterial Spot",
    "Early Blight",
    "Healthy",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites Two-spotted Spider Mite",
    "Target Spot",
    "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus"
]

# Upload image
uploaded_file = st.file_uploader(
    "Choose a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    # Prepare image
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[-3:][::-1]

    predicted_index = top_indices[0]
    predicted_disease = class_names[predicted_index]

    confidence = predictions[predicted_index] * 100
    second_confidence = predictions[top_indices[1]] * 100

    # Checks
    confidence_ok = confidence >= 70
    margin_ok = (confidence - second_confidence) >= 15

    if not confidence_ok or not margin_ok:

        st.subheader("Prediction")

        st.error("❌ Invalid or unclear image")

        st.warning(
            "Please upload a clear image of a tomato leaf."
        )

    else:

        st.subheader("Prediction")

        st.success(f"🌿 Disease: {predicted_disease}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("Top 3 Predictions")

        for index in top_indices:
            st.write(
                f"**{class_names[index]}** — "
                f"{predictions[index] * 100:.2f}%"
            )

st.warning(
    "This is an AI model prediction and should not be treated as a definitive agricultural diagnosis."
)