
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from trainer import train_all_models, model_defs, MODEL_DIR, TEST_DATA_PATH
from utils import load_test_data, load_model, evaluate_model

st.set_page_config(layout="wide")
st.title("🧠 Обучение и оценка моделей на датасете цифр")

model_names = list(model_defs.keys())

# Проверка наличия хотя бы одной обученной модели
section = st.sidebar.radio("Выберите действие:", [
    "🏗️ Обучить модели",
    "📊 Оценить модели",
    "🔍 Проверить метку",
    "🔎 Предсказание по индексу"
])

trained_models = [name for name in model_names if os.path.exists(os.path.join(MODEL_DIR, f"{name}.joblib"))]
if section != "🏗️ Обучить модели" and not trained_models:
    st.error("❌ Ни одна модель не обучена. Пожалуйста, сначала выполните обучение.")
    st.stop()

if section == "🏗️ Обучить модели":
    st.subheader("📚 Обучение всех моделей")
    if st.button("Начать обучение"):
        with st.spinner("Обучение моделей..."):
            df = train_all_models()
        st.success("✅ Обучение завершено")
        st.dataframe(df)

elif section == "📊 Оценить модели":
    st.subheader("📊 Сравнение и отчёты")
    X_test, y_test = load_test_data()
    if X_test is None:
        st.error("❌ test_data.npz не найден.")
        st.stop()

    mode = st.radio("Режим:", ["Одна модель", "Сравнение всех"])
    if mode == "Одна модель":
        selected_model = st.selectbox("Выберите модель:", model_names)
        model = load_model(MODEL_DIR, selected_model)
        if model is None:
            st.error("Модель не найдена.")
            st.stop()
        acc, report, cm = evaluate_model(model, X_test, y_test)
        st.metric("Accuracy", f"{acc:.4f}")
        st.dataframe(pd.DataFrame(report).T.style.format(precision=3))
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix - {selected_model}")
        st.pyplot(fig)
    else:
        results = []
        for name in model_names:
            model = load_model(MODEL_DIR, name)
            if model:
                acc, _, _ = evaluate_model(model, X_test, y_test)
                results.append({"Model": name.replace("_", " ").title(), "Accuracy": round(acc, 4)})
        df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
        st.dataframe(df.set_index("Model").style.format(precision=4))

elif section == "🔍 Проверить метку":
    st.subheader("🔍 Проверка по конкретной цифре")
    X_test, y_test = load_test_data()
    if X_test is None:
        st.error("❌ test_data.npz не найден.")
        st.stop()

    selected_digit = st.selectbox("Выберите цифру:", sorted(np.unique(y_test)))
    indices = np.where(y_test == selected_digit)[0]

    if len(indices) == 0:
        st.warning("Нет изображений с этой меткой.")
        st.stop()

    cols = st.columns(len(model_names))
    for i, name in enumerate(model_names):
        model = load_model(MODEL_DIR, name)
        if model:
            preds = model.predict(X_test[indices])
            acc = (preds == y_test[indices]).mean()
            with cols[i]:
                st.markdown(f"**{name.replace('_', ' ').title()}**")
                st.metric("Точность", f"{acc:.4f}")

    st.subheader("🖼 Примеры изображений и предсказаний")
    show = min(12, len(indices))
    img_cols = st.columns(6)
    for idx, img_index in enumerate(indices[:show]):
        image = X_test[img_index].reshape(8, 8)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap="gray")
        ax.axis("off")
        preds = []
        for name in model_names:
            model = load_model(MODEL_DIR, name)
            if model:
                pred = model.predict([X_test[img_index]])[0]
                preds.append(f"{name.split('_')[0]}: {pred}")
        with img_cols[idx % 6]:
            st.pyplot(fig)
            st.caption(f"True: {y_test[img_index]}\n" + "\n".join(preds))



elif section == "🔎 Предсказание по индексу":
    st.subheader("🔎 Предсказание по индексу в тестовой выборке")
    X_test, y_test = load_test_data()
    if X_test is None:
        st.error("❌ test_data.npz не найден.")
        st.stop()

    if "idx" not in st.session_state:
        st.session_state.idx = 0

    col1, col2 = st.columns([4, 1])
    with col1:
        st.session_state.idx = st.slider(
            "Выберите индекс примера",
            min_value=0, max_value=len(X_test)-1,
            value=st.session_state.idx, step=1
        )
    with col2:
        if st.button("➡️ Следующий"):
            st.session_state.idx = (st.session_state.idx + 1) % len(X_test)
            st.rerun()

    image = X_test[st.session_state.idx].reshape(8, 8)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    st.pyplot(fig)
    st.caption(f"Истинная метка: {y_test[st.session_state.idx]}")

    st.write("### Предсказания моделей:")
    for name in model_names:
        model = load_model(MODEL_DIR, name)
        if model:
            pred = model.predict([X_test[st.session_state.idx]])[0]
            st.write(f"**{name.replace('_', ' ').title()}** → предсказание: **{pred}**")
