import streamlit as st
import requests
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Загружаем digits dataset
digits = load_digits()

st.title("Распознавание loaddigits 🔢")

# Выбор изображения
index = st.slider("Выбери индекс в массиве (0–1796)", 0, len(digits.data) - 1, 0)
image = digits.images[index]
vector = digits.data[index].tolist()

# Выбор модели
model_name = st.selectbox("Выбери модель:", [
    "Logistic Regression",
    "SVM (RBF kernel)",
    "Random Forest",
    "Gradient Boosting",
    "MLP Neural Network"
])

# Отображение изображения
st.image(image / 16.0, width=200, caption=f"Правильный ответ (метка): {digits.target[index]}")

# Кнопка для отправки запроса
if st.button("Сделать предсказание"):
    payload = {
        "model_name": model_name,
        "data": [vector]
    }
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        prediction = response.json()["predictions"][0]
        st.success(f"🤖 Модель предсказала: {prediction}")
    except Exception as e:
        st.error(f"Ошибка: {e}")
