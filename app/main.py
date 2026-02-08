




# Imports
from fastapi import FastAPI
from app.routes import router  # import router z routes.py
from pathlib import Path

# Create the path to the pre-trained Iris model
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "iris_model.joblib"


# Create a FastAPI application
app = FastAPI(title="Iris Zad 8.1")

# Include all endpoints from routes.py
app.include_router(router)
