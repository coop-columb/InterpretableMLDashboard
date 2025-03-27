# backend/fast.py
from fastapi import FastAPI
import pandas as pd # Keep pandas import from previous attempt, useful later
from pathlib import Path # Keep pathlib import, useful later

# Define the base directory of the project (adjust if needed, depends on run context)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Create an instance of the FastAPI class
app = FastAPI(title="Interpretable ML Dashboard API", version="0.1.0")

# Define a root endpoint (a simple test)
@app.get("/")
async def read_root():
    """
    Root endpoint providing a welcome message.
    """
    return {"message": "Welcome to the Interpretable ML Dashboard API!"}

# Define the dataset summary endpoint
@app.get("/dataset-summary/")
async def get_dataset_summary():
    """
    Provides a basic summary of the dataset.
    (Currently returns placeholder data)
    """
    # Placeholder data - we will replace this with real logic later
    summary_data = {
        "dataset_name": "RarePlanes (Placeholder)",
        "total_images_train": 0,
        "total_images_test": 0,
        "total_annotations_train": 0,
        "annotation_types": ["aircraft (Placeholder)"],
        "image_format": "PS-RGB (Placeholder)"
    }
    return summary_data

