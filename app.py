import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

file_id = "16K8jeWhcC6RWmliW4pbsnZAWeimJVW0e"
url = 'https://drive.google.com/file/d/16K8jeWhcC6RWmliW4pbsnZAWeimJVW0e'
model_path = "trained_potato_plant_disease_model.keras"

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = Image.open(test_image).convert("RGB") 
    image = image.resize((128, 128))
    input_arr = np.array(image) / 255.0 
    input_arr = np.expand_dims(input_arr, axis=0)  
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

img = Image.open("steptodown.com534056.jpg")
st.image(img)

if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image:
        image = Image.open(test_image)
        st.image(image, use_container_width=True)

    if st.button("Predict"):
        if test_image:
            st.snow()
            st.write("Our Prediction")
            try:
                result_index = model_prediction(test_image)
                class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
                st.success(f"Model is Predicting it's a {class_name[result_index]}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Please upload an image first.")
