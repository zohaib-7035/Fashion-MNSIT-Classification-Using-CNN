# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="🧥 Fashion Classifier", layout="centered")

# Load model
model = load_model('trained_fashion_mnist_model.h5')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ----- 🔥 Premium Styling -----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');

html, body, .stApp {
    background: linear-gradient(145deg, #2e2e2e, #1e1e1e);
    color: white;
    font-family: 'Orbitron', sans-serif;
    overflow-x: hidden;
}

/* Shiny Title */
h1 {
    text-align: center;
    font-size: 3rem;
    background: linear-gradient(to right, #f8f8f8, #888, #f8f8f8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 3s infinite;
}

@keyframes shine {
    0% {background-position: 0%;}
    100% {background-position: 200%;}
}

.instruction {
    text-align: center;
    font-size: 1.2rem;
    color: #ccc;
    margin-bottom: 20px;
    animation: fadeIn 2s ease-in-out;
}

.uploaded-img {
    display: flex;
    justify-content: center;
    margin-top: 20px;
    animation: fadeInZoom 1s ease-out;
}

img:hover {
    box-shadow: 0 0 20px #9efff3;
    border-radius: 10px;
    transform: rotate(1deg) scale(1.1);
    transition: 0.5s ease-in-out;
}

/* Glassmorphic prediction box */
.prediction-box {
    margin-top: 30px;
    padding: 25px;
    border-radius: 15px;
    background: rgba(255, 255, 255, 0.05);
    box-shadow: 0 8px 32px 0 rgba( 31, 38, 135, 0.37 );
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    text-align: center;
    font-size: 1.5rem;
    color: #00ffe7;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% {box-shadow: 0 0 10px #00ffe7;}
    100% {box-shadow: 0 0 25px #00ffe7;}
}

@keyframes fadeInZoom {
    0% {transform: scale(0.8); opacity: 0;}
    100% {transform: scale(1); opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

# ----- 🔘 Title -----
st.markdown("<h1>🧥 Fashion MNIST Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='instruction'>Upload a 28x28 grayscale image of a fashion item</p>", unsafe_allow_html=True)

# ----- 📤 Upload Image -----
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    st.markdown("<div class='uploaded-img'>", unsafe_allow_html=True)
    st.image(image, caption="🖼️ Image (Resized to 28x28)", width=200)
    st.markdown("</div>", unsafe_allow_html=True)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(tf.nn.softmax(predictions[0])) * 100

    # ----- 🔮 Prediction Box -----
    prediction_text = f"Prediction: {class_names[predicted_class]}<br>Confidence: {confidence:.2f}%"
    st.markdown(f"<div class='prediction-box'>{prediction_text}</div>", unsafe_allow_html=True)

    # ----- 📊 Class Probabilities -----
    st.markdown("### 🔢 Class Probabilities")
    fig, ax = plt.subplots()
    probs = tf.nn.softmax(predictions[0]).numpy()
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(class_names)))
    ax.barh(class_names, probs, color=colors)
    ax.set_xlim([0, 1])
    ax.set_xlabel('Confidence')
    ax.invert_yaxis()
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#2e2e2e')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    st.pyplot(fig)
