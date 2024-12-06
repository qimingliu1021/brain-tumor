
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import plotly.graph_objects as go
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from PIL import Image
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

output_dir = "saliency_maps"
os.makedirs(output_dir, exist_ok=True)

def generate_saliency_map(model, img_array, class_index, img_size): 
    with tf.GradientTape() as tape: 
        img_tensor = tf.convert_to_tensor(img_array)
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        target_class = predictions[:, class_index]

    gradients = tape.gradient(target_class, img_tensor)
    gradients = tf.math.abs(gradients)
    gradients = tf.reduce_max(gradients, axis=-1)
    gradients = gradients.numpy().squeeze()

    gradients = cv2.resize(gradients, img_size)

    center = (gradients.shape[0] // 2, gradients.shape[1] // 2)
    radius = min(center[0], center[1]) - 10
    y, x = np.ogrid[:gradients.shape[0], :gradients.shape[1]]
    mask = (x - center[0])**2 + (y-center[1])**2 <= radius**2

    gradients = gradients * mask

    brain_gradients = gradients[mask]

    brain_gradients = gradients[mask]
    if brain_gradients.max() > brain_gradients.min(): 
        brain_gradients = (brain_gradients - brain_gradients.min()) / (brain_gradients.max() - brain_gradients.min())
    gradients[mask] = brain_gradients

    threshold = np.percentile(gradients[mask], 80)
    gradients[gradients < threshold] = 0

    gradients = cv2.GaussianBlur(gradients, (11, 11), 0)

    heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, img_size)

    original_img = image.img_to_array(img)
    superimposed_img = heatmap * 0.7 + original_img * 0.3
    superimposed_img = superimposed_img.astype(np.uint8)

    img_path = os.path.join(output_dir, upload_file.name)
    with open(img_path, "wb") as f: 
        f.write(upload_file.getbuffer())

    saliency_map_path = f"saliency_maps/{upload_file.name}"

    cv2.imwrite(saliency_map_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    return superimposed_img


def load_xception_model(model_path): 
    img_shape=(299, 299, 3)
    base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet',
                                               input_shape=img_shape, pooling='max')
    model = Sequential([
        base_model, 
        Flatten(), 
        Dropout(rate=0.3), 
        Dense(128, activation='relu'), 
        Dropout(rate=0.25), 
        Dense(4, activation='softmax')
    ])

    model.build((None,) + img_shape)

    model.compile(Adamax(learning_rate=0.001), 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy', 
                             Precision(), 
                             Recall()])
    model.load_weights(model_path)

    return model

# Streamlit application setup
st.title("Brain Tumor Classification")

st.write("Upload an image of a brain MRI scan to classify.")

upload_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if upload_file is not None: 
    selected_model = st.radio(
        "Select Model", 
        ("Transfer Learning - Xception", "Custom CNN")
    )

    if selected_model == "Transfer Learning - Xception": 
        model = load_xception_model('xception_model.weights.h5')
        img_size = (299, 299)
    else: 
        model = load_model('cnn_model.h5')
        img_size = (224, 224)

    labels = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']
    img = image.load_img(upload_file, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)

    class_index = np.argmax(prediction[0])
    result = labels[class_index]

    st.write(f"Predicted Class: {result}")
    st.write("Predictions:")
    for label, prob in zip(labels, prediction[0]): 
        st.write(f"{label}: {prob:.4f}")

    saliency_map = generate_saliency_map(model, img_array, class_index, img_size)
    
    col1, col2 = st.columns(2)
    with col1: 
        st.image(upload_file, caption="Uploaded Image", use_column_width=True)
    with col2: 
        st.image(saliency_map, caption="Saliency Map", use_column_width=True)
