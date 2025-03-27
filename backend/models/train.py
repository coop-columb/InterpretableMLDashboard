# backend/models/train.py
import tensorflow as tf
import logging
from pathlib import Path
import tarfile
import json # Import the json library

# Import model building function from model.py
from .model import build_simple_cnn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define paths (remains the same)
MODEL_SAVE_DIR = Path(__file__).resolve().parent.parent / "saved_models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

def load_rareplanes_data(data_dir: Path):
    """
    Loads RarePlanes data. Reads and parses the first GeoJSON annotation file found.
    """
    logger.info(f"Attempting to load data from {data_dir}...")

    annotation_archive_name = "RarePlanes_train_geojson_aircraft_tiled.tar.gz"
    annotation_archive_path = data_dir / annotation_archive_name

    if not annotation_archive_path.exists():
        logger.error(f"Annotation archive not found at: {annotation_archive_path}")
        return None, None

    annotations_loaded = False # Flag to check if we processed at least one file
    try:
        logger.info(f"Opening tar archive: {annotation_archive_path}...")
        with tarfile.open(annotation_archive_path, "r:gz") as tar:
            logger.info(f"Iterating through members...")
            for member in tar.getmembers():
                # Check if it's a file and ends with .geojson
                if member.isfile() and member.name.lower().endswith('.geojson'):
                    logger.info(f"Processing GeoJSON member: {member.name}")
                    try:
                        # Use extractfile to get a file-like object (bytes)
                        with tar.extractfile(member) as f:
                            if f is not None:
                                content_bytes = f.read()
                                # Decode bytes to string (assuming utf-8)
                                content_str = content_bytes.decode('utf-8')
                                logger.info(f"  - Read {len(content_bytes)} bytes. Decoding successful.")
                                # Parse the JSON string into a Python dictionary
                                data = json.loads(content_str)
                                logger.info(f"  - Parsed JSON data type: {type(data)}")
                                # Log some info about the parsed data (e.g., number of features)
                                features = data.get('features', [])
                                logger.info(f"  - Found {len(features)} features in this file.")
                                annotations_loaded = True # Mark that we processed one
                                # Process only the first GeoJSON file for now for testing
                                logger.info("  - Successfully processed first GeoJSON file. Breaking loop.")
                                break # Remove this break later to process all files
                            else:
                                 logger.warning(f"  - Could not extract file object for {member.name}")
                    except json.JSONDecodeError as json_err:
                        logger.error(f"  - Error decoding JSON from {member.name}: {json_err}")
                        # Decide whether to continue or stop if one file fails
                    except Exception as inner_e:
                         logger.exception(f"  - Error processing member {member.name}: {inner_e}")
                         # Decide whether to continue or stop

            if not annotations_loaded:
                 logger.warning("No GeoJSON files were successfully processed in the archive.")

    except tarfile.TarError as e:
        logger.error(f"Error opening or reading tar archive {annotation_archive_path}: {e}")
        return None, None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during tar extraction: {e}")
        return None, None

    logger.info("Data loading/parsing step finished.")
    # Still return None as we haven't created tf.data.Datasets yet
    return None, None

# --- start_training function remains the same ---
def start_training(data_dir: str, epochs: int = 5, batch_size: int = 32, learning_rate: float = 0.001):
    logger.info("--- Starting Model Training (Placeholder Steps) ---")
    logger.info(f"Parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    train_ds, val_ds = load_rareplanes_data(Path(data_dir))
    if train_ds is None: logger.warning("Placeholder data loading returned None. Cannot proceed with actual training steps.")
    input_shape = (256, 256, 3); num_classes = 1
    model = build_simple_cnn(input_shape=input_shape, num_classes=num_classes)
    logger.info("Compiling model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    if train_ds is not None: logger.info(f"Placeholder: Would train for {epochs} epochs.")
    else: logger.warning("Skipping model fitting placeholder as data was not loaded.")
    model_save_path = MODEL_SAVE_DIR / "simple_cnn_placeholder.keras"
    logger.info(f"Placeholder: Saving model to {model_save_path}")
    logger.info("--- Model Training Script Finished (Placeholder Steps) ---")
    return {"status": "Training script finished (placeholder steps, GeoJSON parsing tested)", "model_path": str(model_save_path)}

