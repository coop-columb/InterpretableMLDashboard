# backend/fast.py
from fastapi import FastAPI, HTTPException, File, UploadFile # Added File, UploadFile
import pandas as pd
from pathlib import Path
import logging
import shutil # Import shutil for potential file operations later

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the base directory of the project (relative to this file)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads" # Define a directory for uploads

# Create upload directory if it doesn't exist (optional, good practice)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Create an instance of the FastAPI class
app = FastAPI(title="Interpretable ML Dashboard API", version="0.1.0")

# --- Root Endpoint ---
@app.get("/")
async def read_root():
    """Root endpoint providing a welcome message."""
    return {"message": "Welcome to the Interpretable ML Dashboard API!"}

# --- Dataset Summary Endpoint ---
@app.get("/dataset-summary/")
async def get_dataset_summary():
    """Provides a basic summary of the dataset by counting source archive files."""
    logger.info("Received request for /dataset-summary/")
    train_images_count = 0
    train_annotations_count = 0
    # ... (rest of the summary logic - keeping it concise here) ...
    test_images_count = 0
    test_annotations_count = 0

    try:
        if DATA_DIR.exists() and DATA_DIR.is_dir():
            logger.info(f"Scanning data directory: {DATA_DIR}")
            for f in DATA_DIR.iterdir():
                if f.is_file() and f.name.endswith('.tar.gz'):
                    if "train" in f.name and "PS-RGB" in f.name: train_images_count += 1
                    elif "train" in f.name and "geojson" in f.name: train_annotations_count += 1
                    elif "test" in f.name and "PS-RGB" in f.name: test_images_count += 1
                    elif "test" in f.name and "geojson" in f.name: test_annotations_count += 1
            logger.info(f"File counts: train_img={train_images_count}, train_ann={train_annotations_count}, test_img={test_images_count}, test_ann={test_annotations_count}")
        else:
            logger.error(f"Data directory not found at {DATA_DIR}")
            raise HTTPException(status_code=404, detail=f"Data directory not found at {DATA_DIR}")

        summary_data = {
            "dataset_name": "RarePlanes", "data_directory_exists": True,
            "source_files_summary": {
                "train_images_archives_found": train_images_count,
                "train_annotations_archives_found": train_annotations_count,
                "test_images_archives_found": test_images_count,
                "test_annotations_archives_found": test_annotations_count,
            },
            "assumed_annotation_types": ["aircraft"], "assumed_image_format": "PS-RGB"
        }
        logger.info("Successfully generated dataset summary.")
        return summary_data
    except Exception as e:
        logger.exception(f"An unexpected error occurred in /dataset-summary/: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- Upload Dataset Endpoint ---
@app.post("/upload-dataset/")
async def upload_dataset_file(uploaded_file: UploadFile = File(...)):
    """
    Accepts a file upload.
    Basic implementation just confirms receipt and filename.
    NOTE: Large file uploads via HTTP have limitations.
    """
    logger.info(f"Received upload request for file: {uploaded_file.filename}")
    # Define path to save the file (optional, could process in memory)
    # destination_path = UPLOAD_DIR / uploaded_file.filename

    try:
        # Example: Save the uploaded file to the UPLOAD_DIR
        # This is a basic way, consider chunking for large files
        # with open(destination_path, "wb") as buffer:
        #     shutil.copyfileobj(uploaded_file.file, buffer)
        # logger.info(f"File '{uploaded_file.filename}' saved to '{destination_path}'")

        # For now, just return confirmation without saving
        return {
            "message": "File received successfully",
            "filename": uploaded_file.filename,
            "content_type": uploaded_file.content_type
        }

    except Exception as e:
        logger.exception(f"Failed to process uploaded file {uploaded_file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not process file: {e}")
    finally:
        # Ensure the UploadFile resource is closed
         await uploaded_file.close() # Important for cleanup


# --- Placeholder for future endpoints ---
# @app.post("/train-model/")
# async def train_model_endpoint(): pass

# @app.post("/predict/")
# async def predict_endpoint(): pass

# @app.post("/explain/")
# async def explain_endpoint(): pass

