import os
import csv
import numpy as np
import joblib as jl
from fastapi import FastAPI, HTTPException
from schemes import ModelResult, PredictRequest, PredictResponse
from train_models import train_all_models

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "ML Model Training API is running!"}

@app.get("/train", response_model=list[ModelResult])
def train_models():
    results = train_all_models()

    output = []
    for r in results:
        model_name = r[0]
        accuracy = r[1]
        training_time = r[2]

        result = ModelResult(
            model=model_name,
            accuracy=accuracy,
            training_time=training_time
        )

        output.append(result)

    return output

@app.get("/results", response_model=list[ModelResult])
def get_results():
    results = []
    try:
        with open("model_results.csv", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(ModelResult(
                    model=row["Model"],
                    accuracy=float(row["Accuracy"]),
                    training_time=float(row["TrainingTime"])
                ))
    except FileNotFoundError:
        return []
    return results

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    safe_name = request.model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
    model_path = os.path.join('models',f"{safe_name}.joblib")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail=f"Model '{request.model_name}' not found. Train it first.")

    model = jl.load(model_path)
    try:
        input_array = np.array(request.data)  # shape (n_samples, 64)
        if input_array.ndim != 2 or input_array.shape[1] != 64:
            raise ValueError("Each sample must have exactly 64 features.")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid input format: {str(e)}")

    predictions = model.predict(input_array).tolist()
    return PredictResponse(predictions=predictions)

