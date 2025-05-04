from typing import List
from pydantic import BaseModel

class ModelResult(BaseModel):
    model: str
    accuracy: float
    training_time: float

class PredictRequest(BaseModel):
    model_name: str            # имя модели, например: "Logistic Regression"
    data: List[List[float]]    # массив изображений, каждый — список из 64 фичей

class PredictResponse(BaseModel):
    predictions: List[int]     # список предсказаний
