# backend/models/train.py
import tensorflow as tf
import logging
from pathlib import Path
import tarfile # Import the tarfile library
# Import model building function from model.py
from .model import build_simple_cnn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Define paths (remains the same)
MODEL_SAVE_DIR = Path(__file__).resolve().parent.parent / "saved_models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

def load_rareplanes_data(data_dir: Path):
    """
    Loads and preprocesses RarePlanes data. Starts by extracting archive contents.
    """
    logger.info(f"Attempting to load data from {data_dir}...")

    # --- Start: Tarfile Extraction Logic ---
    annotation_archive_name = "RarePlanes_train_geojson_aircraft_tiled.tar.gz"
    annotation_archive_path = data_dir / annotation_archive_name

    if not annotation_archive_path.exists():
        logger.error(f"Annotation archive not found at: {annotation_archive_path}")
        # Decide how to handle this - raise error or return None?
        return None, None

    extracted_filenames = []
    try:
        logger.info(f"Opening tar archive: {annotation_archive_path}...")
        with tarfile.open(annotation_archive_path, "r:gz") as tar:
            # Option 1: Get all member names
            # extracted_filenames = tar.getnames()

            # Option 2: Iterate and get member info (more flexible later)
            members = tar.getmembers()
            extracted_filenames = [member.name for member in members]
            logger.info(f"Found {len(extracted_filenames)} members in the archive.")
            if extracted_filenames:
                logger.info("First few members:")
                for name in extracted_filenames[:10]: # Log first 10 filenames
                    logger.info(f"  - {name}")
            # TODO: Add logic here later to actually *extract* or *read* specific files
            # e.g., for member in members:
            #       if member.name.endswith('.geojson'):
            #           f = tar.extractfile(member)
            #           if f:
            #               content = f.read() # Read file content
            #               # Process GeoJSON content...
    except tarfile.TarError as e:
        logger.error(f"Error opening or reading tar archive {annotation_archive_path}: {e}")
        return None, None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during tar extraction: {e}")
        return None, None
    # --- End: Tarfile Extraction Logic ---

    logger.info("Data loading/extraction step finished.")
    # Still return None as we haven't created tf.data.Datasets yet
    return None, None # (train_dataset, validation_dataset)

def start_training(data_dir: str, epochs: int = 5, batch_size: int = 32, learning_rate: float = 0.001):
    """
    Main function to orchestrate the model training process (calls load_rareplanes_data).
    """
    logger.info("--- Starting Model Training (Placeholder Steps) ---")
    logger.info(f"Parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

    # 1. Load Data (Now calls the updated function)
    train_ds, val_ds = load_rareplanes_data(Path(data_dir))
    # The warning about None being returned is now expected until we implement full loading
    if train_ds is None:
         logger.warning("Placeholder data loading returned None. Cannot proceed with actual training steps.")

    # 2. Build Model (remains the same placeholder)
    input_shape = (256, 256, 3); num_classes = 1
    model = build_simple_cnn(input_shape=input_shape, num_classes=num_classes)

    # 3. Compile Model (remains the same placeholder)
    logger.info("Compiling model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])

    # 4. Train Model (placeholder actions) - Skip if data is None
    if train_ds is not None:
         logger.info("Starting model fitting (placeholder)...")
         logger.info(f"Placeholder: Would train for {epochs} epochs.")
    else:
         logger.warning("Skipping model fitting placeholder as data was not loaded.")

    # 5. Save Model (placeholder action)
    model_save_path = MODEL_SAVE_DIR / "simple_cnn_placeholder.keras"
    logger.info(f"Placeholder: Saving model to {model_save_path}")

    logger.info("--- Model Training Script Finished (Placeholder Steps) ---")
    return {"status": "Training script finished (placeholder steps, data extraction tested)", "model_path": str(model_save_path)}

