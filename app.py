import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/mnist_model.h5")

st.title("✍️ Handwritten Digit Recognition")

st.write("Upload an image of a handwritten digit (0–9)")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # grayscale
    st.image(image, caption='Uploaded Image', width=150)

    # Preprocess
    image = image.resize((28,28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1,28,28,1)

    # Prediction
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)

    st.success(f"Predicted Digit: {digit}")