import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_digits
from train_models import train_all_models

# ────────────────────────────
# 🔢 Загрузка данных
digits = load_digits()
X, y = digits.data, digits.target
images = digits.images

# ────────────────────────────
st.title("💡 Распознавание цифр с помощью моделей ML")

# ────────────────────────────
st.sidebar.header("1️⃣ Обучение моделей")
if st.sidebar.button("Обучить все модели"):
    results = train_all_models()
    st.success("✅ Модели обучены и сохранены.")
    df = pd.DataFrame(results, columns=["Model", "Accuracy", "Training Time (s)"])
    st.write("📊 Результаты обучения:", df)

# ────────────────────────────
st.sidebar.header("2️⃣ Предсказание")
index = st.sidebar.slider("Выбери индекс изображения (0–1796)", 0, len(X)-1, 0)
image = images[index]
vector = X[index].reshape(1, -1)

model_name = st.sidebar.selectbox("Выбери модель:", [
    "Logistic Regression",
    "SVM (RBF kernel)",
    "Random Forest",
    "Gradient Boosting",
    "MLP Neural Network"
])

safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
model_path = os.path.join("models", f"{safe_name}.joblib")

st.image(image / 16.0, caption=f"Цифра: {y[index]}", width=200)

if st.sidebar.button("Сделать предсказание"):
    if not os.path.exists(model_path):
        st.error("⚠️ Модель не найдена. Сначала обучите её.")
    else:
        model = joblib.load(model_path)
        pred = model.predict(vector)[0]
        st.success(f"🤖 Предсказание модели: **{pred}**")

# ────────────────────────────
st.sidebar.header("3️⃣ Результаты (если уже обучали)")
if os.path.exists("model_results.csv"):
    df = pd.read_csv("model_results.csv")
    st.sidebar.dataframe(df)
