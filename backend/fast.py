# backend/fast.py
from fastapi import FastAPI
import pandas as pd # Keep pandas import, useful later
from pathlib import Path # Import Path for file system operations

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
    """
    train_images_count = 0
    train_annotations_count = 0
    test_images_count = 0
    test_annotations_count = 0 # Assuming test annotations exist based on roadmap

    # Ensure DATA_DIR exists before iterating
    if DATA_DIR.exists() and DATA_DIR.is_dir():
        for f in DATA_DIR.iterdir():
            # Check if it's a .tar.gz file we expect
            if f.is_file() and f.name.endswith('.tar.gz'):
                if "train" in f.name and "PS-RGB" in f.name:
                    train_images_count += 1
                elif "train" in f.name and "geojson" in f.name:
                    train_annotations_count += 1
                elif "test" in f.name and "PS-RGB" in f.name:
                    test_images_count += 1
                elif "test" in f.name and "geojson" in f.name:
                    test_annotations_count += 1
    else:
        # Handle case where data directory might be missing (optional, good practice)
        print(f"Warning: Data directory not found at {DATA_DIR}")
        # Or raise an HTTPException(status_code=404, detail="Data directory not found")

    # Structure the summary data
    summary_data = {
        "dataset_name": "RarePlanes",
        "data_directory_exists": DATA_DIR.exists(),
        "source_files_summary": {
            "train_images_archives_found": train_images_count,
            "train_annotations_archives_found": train_annotations_count,
            "test_images_archives_found": test_images_count,
            "test_annotations_archives_found": test_annotations_count,
        },
        # Note: These are counts of ARCHIVES, not files inside them.
        "assumed_annotation_types": ["aircraft"],
        "assumed_image_format": "PS-RGB"
    }
    return summary_data

