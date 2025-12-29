import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Label mapping (A–Z without J)
labels = [chr(i + 65) if i < 9 else chr(i + 66) for i in range(25)]

st.title("Sign Language Classification")
st.write("Upload a hand sign image (28x28 grayscale)")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]

    st.image(img, caption="Uploaded Image", width=200)
    st.success(f"Predicted Sign: {predicted_class}")
