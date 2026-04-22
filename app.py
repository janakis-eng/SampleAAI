import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---- Title ----
st.title("Flower Classification App")

# ---- Upload Image ----
uploaded_file = st.file_uploader("Upload Flower Image", type=["jpg", "png", "jpeg"])

# ---- Slider ----
epochs = st.slider("Select Epochs", 1, 20, 5)

# ---- Buttons ----
cnn_btn = st.button("Run CNN")
resnet_btn = st.button("Run ResNet")

# ---- Load Image ----
def preprocess(image):
    img = image.resize((180,180))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

# 
def load_cnn():
    model = tf.keras.models.load_model("cnn_model.h5")
    return model

# 
def load_resnet():
    model = tf.keras.models.load_model("resnet.h5")
    return model

# ---- Display Image ----
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    img_array = preprocess(image)

# ---- CNN Prediction ----
if cnn_btn and uploaded_file:
    model = load_cnn()
    preds = model.predict(img_array)
    st.write("CNN Prediction:", np.argmax(preds))

# ---- ResNet Prediction ----
if resnet_btn and uploaded_file:
    model = load_resnet()
    preds = model.predict(img_array)
    st.write("ResNet Prediction:", np.argmax(preds))