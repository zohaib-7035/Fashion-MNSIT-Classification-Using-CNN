
# 🧥 Fashion MNIST Classifier (Streamlit App)

A modern, interactive, and visually engaging web app that classifies fashion items using a deep learning model trained on the Fashion MNIST dataset. Built with TensorFlow, Streamlit, and Python.

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://your-username.streamlit.app)

---

## 📸 Demo

<img src="screenshot.png" width="100%">

---

## 🚀 Features

- 🌟 Predicts fashion item from grayscale image (28x28)
- 🧠 Deep learning model (ConvNet using TensorFlow)
- 💫 Beautiful UI with animated predictions
- 📊 Confidence bar chart for all classes
- 🎨 Glassmorphism and dark-themed UI for a modern experience

---

## 📁 Files

| File                  | Description                                     |
|-----------------------|-------------------------------------------------|
| `app.py`              | Main Streamlit app script                       |
| `trained_fashion_mnist_model.h5` | Pretrained CNN model for prediction     |
| `requirements.txt`    | List of Python packages needed for deployment   |
| `README.md`           | This file                                       |

---

## 🧪 How to Run Locally

```bash
# 1. Clone this repository
git clone https://github.com/your-username/fashion-mnist-app.git
cd fashion-mnist-app

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
````

---

## 🌐 How to Deploy on Streamlit Cloud

1. Upload the project to a GitHub repository.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and log in with GitHub.
3. Click “New app”, select your repo.
4. Set `app.py` as the entry point.
5. Click **Deploy** – you’ll get a live link to share!

---

## 🧠 Model Architecture

* 3 Convolutional Layers (`Conv2D`)
* MaxPooling Layers
* Dropout & Batch Normalization for regularization
* Dense Layers with ReLU
* Final Dense layer with 10 logits (for 10 classes)

---

## 🧵 Classes

The app can recognize the following classes from Fashion MNIST:

```
['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

---

## 📦 Requirements

```
streamlit
tensorflow
numpy
Pillow
matplotlib
```

---

## ✨ Credits

* Trained using [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
* Inspired by TensorFlow tutorials
* UI enhanced with custom CSS and design

---



Made with ❤️ by Zohaib Shahid

---

## 📜 License

This project is licensed under the MIT License.

```

---



