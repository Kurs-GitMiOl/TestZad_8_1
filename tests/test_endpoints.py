from fastapi.testclient import TestClient
from app.main import app

# Create a test client for the FastAPI app
client = TestClient(app)

# Basic test for the /predict endpoint:
# Sends a sample Iris flower JSON and checks if the response
# returns status 200 and contains the expected prediction fields
def test_predict():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.2,
            "petal_length": 2.4,
            "petal_width": 1.2
        }
    )

    # Check HTTP status code
    assert response.status_code == 200

    data = response.json()

    # Check response structure
    assert "prediction_class" in data
    assert "prediction_name" in data

# ---------------------------------------------------------------

 # Test /predict with minimum and maximum feature values from Iris dataset
def test_predict_edge_cases():
    min_features = {
        "sepal_length": 4.3,
        "sepal_width": 2.0,
        "petal_length": 1.0,
        "petal_width": 0.1
    }

    response_min = client.post("/predict", json=min_features)
    assert response_min.status_code == 200
    data_min = response_min.json()
    assert "prediction_class" in data_min
    assert "prediction_name" in data_min

    # Test /predict with maximum feature values from Iris dataset
    max_features = {
        "sepal_length": 7.9,
        "sepal_width": 4.4,
        "petal_length": 6.9,
        "petal_width": 2.5
    }

    response_max = client.post("/predict", json=max_features)
    assert response_max.status_code == 200
    data_max = response_max.json()
    assert "prediction_class" in data_max
    assert "prediction_name" in data_max

#----------------------------------------------------------------

# Test /predict_proba:
# Send flower data and check probabilities sum to ~1
def test_predict_proba():
    response = client.post(
        "/predict_proba",
        json={
            "sepal_length": 6.0,
            "sepal_width": 3.9,
            "petal_length": 4.5,
            "petal_width": 2.6
        }
    )

    assert response.status_code == 200

    data = response.json()

    # Check probabilities exist
    assert "setosa" in data
    assert "versicolor" in data
    assert "virginica" in data

    # Check probabilities sum to ~1.0
    total = data["setosa"] + data["versicolor"] + data["virginica"]
    assert abs(total - 1.0) < 0.002

# -------------------------------------------------------------------

# Test /describe_input
# Send flower data and check min, max, mean values
def test_describe_input():
    response = client.post(
        "/describe_input",
        json={
            "sepal_length": 5.0,
            "sepal_width": 2.8,
            "petal_length": 4.5,
            "petal_width": 1.2
        }
    )

    # Check HTTP status code
    assert response.status_code == 200

    data = response.json()

    # Check response structure
    assert "min" in data
    assert "max" in data
    assert "mean" in data

    # Check values are correct
    assert data["min"] == 1.2
    assert data["max"] == 5.0
    assert data["mean"] == (5.0 + 2.8 + 4.5 + 1.2) / 4



#------------------------------------------------------------------------

# Test /describe_input_get
# send flower data as query params and check min, max, mean
def test_describe_input_get():
    response = client.get(
        "/describe_input_get",
        params={
            "sepal_length": 5.5,
            "sepal_width": 2.8,
            "petal_length": 4.0,
            "petal_width": 1.5
        }
    )

    # Check HTTP status code
    assert response.status_code == 200

    data = response.json()

    # Check response structure
    assert "min" in data
    assert "max" in data
    assert "mean" in data

    # Check values are correct
    expected_min = min([5.5, 2.8, 4.0, 1.5])
    expected_max = max([5.5, 2.8, 4.0, 1.5])
    expected_mean = sum([5.5, 2.8, 4.0, 1.5]) / 4

    assert data["min"] == expected_min
    assert data["max"] == expected_max
    assert data["mean"] == expected_mean


#-------------------------------------------------------------------------

# Test /describe_input with invalid data
# Check it returns 422 error
def test_describe_input_invalid():
    response = client.post("/describe_input", json={
        "sepal_length": 2.0,
        "sepal_width": "error",
        "petal_length": 3.0,
        "petal_width": 1.0
    })

    assert response.status_code == 422


# -----------------------------------------------------------------------


# Check if endpoint returns basic model configuration
def test_model_info():
    response = client.get("/model_info")

    # Check HTTP status code
    assert response.status_code == 200

    data = response.json()

    # Check response structure
    assert "model_type" in data
    assert "kernel" in data
    assert "probability" in data

    # Check returned values
    assert data["model_type"] == "SVC"
    assert data["kernel"] == "linear"
    assert data["probability"] is True