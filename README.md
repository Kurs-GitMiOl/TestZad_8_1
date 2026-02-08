# Zad 8.1 Iris ML API

## 1. Project description

The project shows a simple AI system shared as a FastAPI web service.
The system uses a previously trained classification model
to predict the Iris flower species based on given features.

## 2. Used model
- Type: SVM Classifier (Support Vector Machine
- Pipeline: 
StandardScaler() – scales the input features 
SVC(kernel='linear', probability=True) – linear SVM classifier with probability output
- Library: scikit-learn

## 2. Input Data from dataset
Features:
- sepal_length – float
- sepal_width – float
- petal_length – float
- petal_width – float

## 3. Used technologies
- Python
- scikit-learn
- FastAPI
- uv
- Git

## 4. Installation and run

- open the terminal and go to the project folder
- create a virtual environment: python -m venv .venv
- activate it: .venv\Scripts\activate
- install dependencies: uv sync
- start the FastAPI server: uvicorn app.main:app --reload
- the API will be available at: http://127.0.0.1:8000
- open the API documentation at: http://127.0.0.1:8000/docs

## 5. Requirements
- Python 3.10+
- uv

# 6. Endpoints description and examples

- Endpoint `/predict`

    Prediction endpoint for iris flower classification.
    It takes flower features in JSON format, predicts the flower class,
    and returns both the class number and the class name.

### Example JSON requests to /predict
1. Setosa

Request:
{
  "sepal_length": 5.0,
  "sepal_width": 3.4,
  "petal_length": 1.5,
  "petal_width": 0.2
}

Response:
{
  "prediction_class": 0,
  "prediction_name": "setosa"
}


2. Versicolor

Request:
{
  "sepal_length": 6.0,
  "sepal_width": 2.7,
  "petal_length": 4.5,
  "petal_width": 1.5
}

Response:
{
  "prediction_class": 1,
  "prediction_name": "versicolor"
}


3. Virginica

Request:
{
  "sepal_length": 6.5,
  "sepal_width": 3.0,
  "petal_length": 5.2,
  "petal_width": 2.0
}

Response:
{
  "prediction_class": 2,
  "prediction_name": "virginica"
}


- Endpoint `/describe_input` 

    Endpoint that describes iris flower features.
    It takes flower features in JSON, calculates the minimum, maximum, and mean values, and returns them.


### Example JSON requests to `/describe_input`
1. Example

Request:
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}

Response:
{
  "min": 0.2,
  "max": 5.1,
  "mean": 2.55
}

2. Example

Request:
{
  "sepal_length": 6.0,
  "sepal_width": 2.8,
  "petal_length": 4.5,
  "petal_width": 1.5
}

Response:
{
  "min": 1.5,
  "max": 6.0,
  "mean": 3.7
}

- Endpoint `/predict_proba`

    Return prediction probabilities for each Iris flower class.
    The endpoint takes flower features and returns a probability for each class.

### Example JSON requests to `/predict_proba`
1. Example

Request:
{
  "sepal_length": 5.9,
  "sepal_width": 3.0,
  "petal_length": 4.2,
  "petal_width": 1.5
}

Response:
{
  "setosa": 0.0,
  "versicolor": 0.85,
  "virginica": 0.15
}

2. Example

Request:
{
  "sepal_length": 6.5,
  "sepal_width": 3.0,
  "petal_length": 5.2,
  "petal_width": 2.0
}

Response:
{
  "setosa": 0.0,
  "versicolor": 0.05,
  "virginica": 0.95
}


- Endpoint `/describe_input_get`

    Describe Iris input values using query parameters.
    The endpoint returns min, max, and mean values of the input features.

### Example requests to  `/describe_input_get`
1. Example

http://127.0.0.1:8000/describe_input_get?sepal_length=6.0&sepal_width=2.8&petal_length=4.5&petal_width=1.5

Response:
{
  "min": 1.5,
  "max": 6,
  "mean": 3.7
}
2. Example

http://127.0.0.1:8000/describe_input_get?sepal_length=5.4&sepal_width=3.2&petal_length=1.6&petal_width=0.4

Response:
{
  "min": 0.4,
  "max": 5.4,
  "mean": 2.6500000000000004
}

## 7.  Tests
This project includes basic tests for all main endpoints of application.
The tests are written using pytest and FastAPI’s TestClient.

- POST /predict – checks prediction_class and prediction_name
- POST /predict_proba – checks probabilities for each class and that they sum to ~1
- POST /describe_input – checks min, max, mean values.
- POST /describe_input_invalid – sends wrong data and checks the API returns 422 error
- GET /describe_input_get – checks min, max, mean values from query parameters

How to run tests:
- Install dependencies including pytest: pip install pytest
- Run all tests: pytest