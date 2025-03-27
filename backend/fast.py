# backend/fast.py
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks # Added BackgroundTasks
import pandas as pd
from pathlib import Path
import logging
import shutil
# Import the training function
from backend.models.train import start_training

# Configure logging... (remains the same)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths... (remains the same)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# App Instance... (remains the same)
app = FastAPI(title="Interpretable ML Dashboard API", version="0.1.0")

# --- Root Endpoint --- (remains the same)
@app.get("/")
async def read_root(): return {"message": "Welcome to the Interpretable ML Dashboard API!"}

# --- Dataset Summary Endpoint --- (remains the same)
@app.get("/dataset-summary/")
async def get_dataset_summary():
    # (Code remains the same...)
    logger.info("Received request for /dataset-summary/")
    try: # Simplified for brevity
        if not (DATA_DIR.exists() and DATA_DIR.is_dir()): raise HTTPException(status_code=404, detail=f"Data directory not found at {DATA_DIR}")
        # ... (file counting logic) ...
        train_images_count=1; train_annotations_count=1; test_images_count=1; test_annotations_count=1 # Example counts
        summary_data = { "dataset_name": "RarePlanes", "data_directory_exists": True, "source_files_summary": {"train_images_archives_found": train_images_count, "train_annotations_archives_found": train_annotations_count, "test_images_archives_found": test_images_count, "test_annotations_archives_found": test_annotations_count,}, "assumed_annotation_types": ["aircraft"], "assumed_image_format": "PS-RGB"}
        logger.info("Successfully generated dataset summary.")
        return summary_data
    except Exception as e: logger.exception(f"Error in /dataset-summary/: {e}"); raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# --- Upload Dataset Endpoint --- (remains the same)
@app.post("/upload-dataset/")
async def upload_dataset_file(uploaded_file: UploadFile = File(...)):
    # (Code remains the same...)
    logger.info(f"Received upload request for file: {uploaded_file.filename}")
    try: return {"message": "File received successfully","filename": uploaded_file.filename,"content_type": uploaded_file.content_type}
    except Exception as e: logger.exception(f"Failed to process uploaded file {uploaded_file.filename}: {e}"); raise HTTPException(status_code=500, detail=f"Could not process file: {e}")
    finally: await uploaded_file.close()


# --- Train Model Endpoint --- (MODIFIED)
@app.post("/train-model/")
async def train_model_endpoint(params: dict = None, background_tasks: BackgroundTasks = None): # Accept optional params, BackgroundTasks
    """
    Triggers the model training process (currently placeholder logic).
    """
    logger.info(f"Received request to /train-model/ with params: {params}")
    params = params or {} # Ensure params is a dict

    # Extract parameters (using defaults if not provided)
    epochs = params.get("epochs", 5)
    batch_size = params.get("batch_size", 32)
    learning_rate = params.get("learning_rate", 0.001)

    # Option 1: Run synchronously (API call waits for training to finish)
    # try:
    #     result = start_training(data_dir=str(DATA_DIR), epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    #     logger.info(f"Training script result: {result}")
    #     return {"message": "Training process finished (placeholder).", "details": result}
    # except Exception as e:
    #     logger.exception("Error during training process call.")
    #     raise HTTPException(status_code=500, detail=f"Error initiating training: {e}")

    # Option 2: Run asynchronously in the background (API call returns immediately)
    # Preferred for long tasks like training
    logger.info("Adding training task to background.")
    # background_tasks.add_task(start_training, data_dir=str(DATA_DIR), epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    # For now, let's just call it directly to see logs, but acknowledge it's long running
    try:
        # Call directly for now to see logs during testing, but this WILL block the API response
        # In production, use background_tasks or a task queue (Celery)
        logger.warning("Calling training function synchronously (will block API response). Use background task for production.")
        result = start_training(data_dir=str(DATA_DIR), epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        logger.info(f"Synchronous training script call finished with result: {result}")
        # This return happens only *after* start_training finishes
        return {"message": "Synchronous training call finished (placeholder).", "details": result}
    except Exception as e:
         logger.exception("Error calling synchronous training function.")
         raise HTTPException(status_code=500, detail=f"Error during training call: {e}")


# --- Predict Endpoint --- (remains placeholder)
@app.post("/predict/")
async def predict_endpoint(data: dict = None):
    logger.info(f"Received request to /predict/ with data: {data}")
    return {"message": "Prediction request received (placeholder).", "input_data": data, "predictions": []}

# --- Explain Endpoint --- (remains placeholder)
@app.post("/explain/")
async def explain_endpoint(data: dict = None):
    logger.info(f"Received request to /explain/ with data: {data}")
    return {"message": "Explanation request received (placeholder).", "input_data": data, "explanation": {}}

