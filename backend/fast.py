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

# Placeholder for the dataset summary endpoint we will add next
# @app.get("/dataset-summary/")
# async def get_dataset_summary():
#     # Logic will go here
#     pass

