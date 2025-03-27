# backend/fast.py
from fastapi import FastAPI, HTTPException # Import HTTPException
import pandas as pd
from pathlib import Path
import logging # Import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the base directory of the project (relative to this file)
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
    Provides a basic summary of the dataset by counting source archive files.
    Handles potential errors during file access.
    """
    logger.info("Received request for /dataset-summary/") # Add logging
    train_images_count = 0
    train_annotations_count = 0
    test_images_count = 0
    test_annotations_count = 0

    try: # Add a try block for file operations
        if DATA_DIR.exists() and DATA_DIR.is_dir():
            logger.info(f"Scanning data directory: {DATA_DIR}")
            for f in DATA_DIR.iterdir():
                if f.is_file() and f.name.endswith('.tar.gz'):
                    # --- File counting logic (same as before) ---
                    if "train" in f.name and "PS-RGB" in f.name:
                        train_images_count += 1
                    elif "train" in f.name and "geojson" in f.name:
                        train_annotations_count += 1
                    elif "test" in f.name and "PS-RGB" in f.name:
                        test_images_count += 1
                    elif "test" in f.name and "geojson" in f.name:
                        test_annotations_count += 1
            logger.info(f"File counts: train_img={train_images_count}, train_ann={train_annotations_count}, test_img={test_images_count}, test_ann={test_annotations_count}")

        else:
            logger.error(f"Data directory not found at {DATA_DIR}")
            # Raise an HTTP exception if the data dir doesn't exist
            raise HTTPException(status_code=404, detail=f"Data directory not found at {DATA_DIR}")

        # Structure the summary data (same as before)
        summary_data = {
            "dataset_name": "RarePlanes",
            "data_directory_exists": True, # True if we passed the check above
            "source_files_summary": {
                "train_images_archives_found": train_images_count,
                "train_annotations_archives_found": train_annotations_count,
                "test_images_archives_found": test_images_count,
                "test_annotations_archives_found": test_annotations_count,
            },
            "assumed_annotation_types": ["aircraft"],
            "assumed_image_format": "PS-RGB"
        }
        logger.info("Successfully generated dataset summary.")
        return summary_data

    except Exception as e:
        # Catch any other unexpected errors during the process
        logger.exception(f"An unexpected error occurred in /dataset-summary/: {e}") # Log the full exception
        # Return a generic server error response
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

