# Zad 8.1 Iris ML API
# Do przetłumaczenia na angielski ????????????????
## Opis projektu
Projekt przedstawia prosty system AI udostępniony jako usługa webowa FastApi.
System używa wcześniej wytrenowany model klasyfikacyjny,
do przewidywania gatuneku kwiatu Iris na podstawie podanych cech.

## Wykorzystane technologie
- Python
- scikit-learn
- FastAPI
- uv
- Git

## Instalacja i uruchomienie
- uruchom swoje środowisko Python
    uv venv
    uv sync
  
- opcionalnie  
    python -m venv .venv
    source .venv/bin/activate   # Linux/macOS
    .venv\Scripts\activate      # Windows
    pip install .
- wytrenuj model  - python train_model.py
- uruchom API - app.main:app --reload
  Api bedzie dostępne po linkem http://127.0.0.1:8000
- przejdz do API documentation http://127.0.0.1:8000/docs

### Wymagania
- Python 3.10+
- uv

### Przykładowe zapytania w formacie JSON do /predict

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

### Przykładowe zapytania w formacie JSON do /describe_input
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

### Przykładowe zapytania w formacie JSON do `/predict_proba`

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

### Przykładowe zapytania do `/describe_input_get`

http://127.0.0.1:8000/describe_input_get?sepal_length=6.0&sepal_width=2.8&petal_length=4.5&petal_width=1.5
Response:
{
  "min": 1.5,
  "max": 6,
  "mean": 3.7
}

http://127.0.0.1:8000/describe_input_get?sepal_length=5.4&sepal_width=3.2&petal_length=1.6&petal_width=0.4
Response:
{
  "min": 0.4,
  "max": 5.4,
  "mean": 2.6500000000000004
}