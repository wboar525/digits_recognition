import time
import csv
import os
import joblib as jl
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def train_all_models():
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = [
        ("Logistic Regression", Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ])),
        ("SVM (RBF kernel)", Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", random_state=42))
        ])),
        ("Random Forest", Pipeline([
            ("classifier", RandomForestClassifier(random_state=42))
        ])),
        ("Gradient Boosting", Pipeline([
            ("classifier", GradientBoostingClassifier(random_state=42))
        ])),
        ("MLP Neural Network", Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(max_iter=300, random_state=42))
        ])),
    ]

    results = []
    for name, pipeline in models:
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        end_time = time.time()
        train_time = end_time - start_time
        accuracy = pipeline.score(X_test, y_test)
        results.append((name, accuracy, train_time))

    with open("model_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy", "TrainingTime"])
        for row in results:
            writer.writerow([row[0], f"{row[1]:.4f}", f"{row[2]:.4f}"])

    # Сохраняем каждую модель
    os.makedirs('models', exist_ok=True)

    for name, pipeline in models:
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
        jl.dump(pipeline, os.path.join('models',f"{safe_name}.joblib"))

    return results


