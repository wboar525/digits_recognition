
import os
import time
import joblib
import numpy as np
import pandas as pd

from datetime import datetime
LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "train_log.csv")
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

MODEL_DIR = "models"
TEST_DATA_PATH = "test_data.npz"
RESULTS_PATH = "model_results.csv"

model_defs = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "svm_rbf_kernel": SVC(kernel="rbf", probability=True),
    "random_forest": RandomForestClassifier(),
    "gradient_boosting": GradientBoostingClassifier(),
    "mlp_neural_network": MLPClassifier(max_iter=1000)
}

def train_all_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.3, random_state=42, stratify=digits.target
    )
    np.savez(TEST_DATA_PATH, X_test=X_test, y_test=y_test)

    results = []
    for name, model in model_defs.items():
        start = time.time()
        model.fit(X_train, y_train)
        duration = round(time.time() - start, 3)
        y_pred = model.predict(X_test)
        acc = round(accuracy_score(y_test, y_pred), 4)
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
        results.append((name.replace("_", " ").title(), acc, duration))

    df = pd.DataFrame(results, columns=["Model", "Accuracy", "Training Time (s)"])
    df.to_csv(RESULTS_PATH, index=False)

    # Логирование времени запуска
    os.makedirs(LOG_DIR, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = pd.DataFrame([[now] + df["Accuracy"].tolist()], columns=["Timestamp"] + df["Model"].tolist())
    if os.path.exists(LOG_PATH):
        existing = pd.read_csv(LOG_PATH)
        log_entry = pd.concat([existing, log_entry], ignore_index=True)
    log_entry.to_csv(LOG_PATH, index=False)
    return df
