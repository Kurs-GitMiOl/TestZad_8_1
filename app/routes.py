# Imports

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import APIRouter
import joblib
import numpy as np
from pathlib import Path


# Load the pre-trained model safely regardless of current working directory
# Loads the trained model safely, no matter the current folder
# __file__ = actual plik (main.py)

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "iris_model.joblib"
print("Loading model from:", MODEL_PATH)
model = joblib.load(MODEL_PATH)


app = FastAPI(title="Iris Zad 8.1")



# Create router
router = APIRouter()    # po przeniesieniu do osobnego pliku

# Schemat danych wejściowych
# Input data schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Endpoint predykcyjny @app przed przeniesieniem do routes
@router.post("/predict")
def predict(data: IrisInput):
    """
    Prediction endpoint for iris flower classification.
    It takes flower features in JSON format, predicts the flower class,
    and returns both the class number and the class name.
    """

    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = model.predict(features)[0]

    target_names = ["setosa", "versicolor", "virginica"]

    return {
        "prediction_class": int(prediction),
        "prediction_name": target_names[prediction]
    }

# zwraca podstawowe statystyki wprowadzonych cech
@router.post("/describe_input")
def describe_input(data: IrisInput):
    """
    Endpoint that describes iris flower features.
    It takes flower features in JSON, calculates the minimum, maximum, and mean values, and returns them.
    """

    features = [data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]
    return {
        "min": min(features),
        "max": max(features),
        "mean": sum(features)/len(features)
    }



# Zwraca prawdopodobieństwo przynależności do każdej klasy kwiatu Iris,
# zamiast tylko jednej przewidywanej klasy.
@router.post("/predict_proba")
def predict_proba(input: IrisInput):
    """
    Return prediction probabilities for each Iris flower class.
    The endpoint takes flower features and returns a probability for each class.
    """

    features = [[
        input.sepal_length,
        input.sepal_width,
        input.petal_length,
        input.petal_width
    ]]


    probabilities = model.predict_proba(features)[0]
    return {
        "setosa": float(probabilities[0]),
        "versicolor": float(probabilities[1]),
        "virginica": float(probabilities[2])
    }

    # kod do usunuiecia bo nie ma sprawdzania jezeli sam sobie piszę resztę kodu
    # # Sprawdzenie, czy model wspiera predict_proba
    # if hasattr(model, "predict_proba"):
    #     probabilities = model.predict_proba(features)[0]
    #     return {
    #         "setosa": float(probabilities[0]),
    #         "versicolor": float(probabilities[1]),
    #         "virginica": float(probabilities[2])
    #     }
    # else:
    #     return {"error": "Model nie wspiera predict_proba"}


#bez wysyłania body JSON, można go wywołać bezpośrednio w przeglądarce
# przykład http://127.0.0.1:8000/describe_input_get?sepal_length=6.0&sepal_width=2.8&petal_length=4.5&petal_width=1.5
@router.get("/describe_input_get")
def describe_input_get(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    """
    Describe Iris input values using query parameters.
    The endpoint returns min, max, and mean values of the input features.
    """

    features = [sepal_length, sepal_width, petal_length, petal_width]
    return {
        "min": min(features),
        "max": max(features),
        "mean": sum(features)/len(features)
    }