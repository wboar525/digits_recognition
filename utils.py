
import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_test_data(path="test_data.npz"):
    if not os.path.exists(path):
        return None, None
    data = np.load(path)
    return data["X_test"], data["y_test"]

def load_model(model_dir, model_name):
    path = os.path.join(model_dir, f"{model_name}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    return acc, report, cm
