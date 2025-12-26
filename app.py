import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Image Classifier")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("image_model.h5")

model = load_model()

class_names = ['cat', 'dog']

st.title("🐱🐶 Image Classification App")
st.write("Upload an image and get prediction")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"✅ Predicted Class: **{predicted_class}**")
