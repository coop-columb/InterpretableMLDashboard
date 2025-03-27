# backend/models/train.py
import tensorflow as tf
import logging
from pathlib import Path
import tarfile
import json
import time # For timing data loading

# Import model building function
from .model import build_simple_cnn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Add timestamp to logs

# Define paths
MODEL_SAVE_DIR = Path(__file__).resolve().parent.parent / "saved_models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

def load_rareplanes_data(data_dir: Path):
    """
    Loads RarePlanes data. Reads and parses ALL GeoJSON annotation files
    from the archive and stores annotations.
    """
    logger.info(f"Attempting to load annotation data from {data_dir}...")
    start_time = time.time()

    annotation_archive_name = "RarePlanes_train_geojson_aircraft_tiled.tar.gz"
    annotation_archive_path = data_dir / annotation_archive_name

    if not annotation_archive_path.exists():
        logger.error(f"Annotation archive not found at: {annotation_archive_path}")
        return None, None # Indicate failure to load data

    # Dictionary to store annotations: {image_id: [list of bboxes]}
    all_annotations = {}
    total_features_processed = 0
    processed_file_count = 0

    try:
        logger.info(f"Opening tar archive: {annotation_archive_path}...")
        with tarfile.open(annotation_archive_path, "r:gz") as tar:
            members = tar.getmembers()
            total_members = len(members)
            logger.info(f"Found {total_members} members. Iterating...")

            for idx, member in enumerate(members):
                # Log progress occasionally
                if idx > 0 and idx % 500 == 0:
                     logger.info(f"  Processed {idx}/{total_members} members...")

                if member.isfile() and member.name.lower().endswith('.geojson'):
                    # Assume filename stem is the image ID
                    image_id = Path(member.name).stem
                    try:
                        with tar.extractfile(member) as f:
                            if f is not None:
                                content_bytes = f.read()
                                content_str = content_bytes.decode('utf-8')
                                data = json.loads(content_str)
                                features = data.get('features', [])
                                if features: # Only process if there are features
                                    bboxes = []
                                    for feature in features:
                                        try:
                                            # GeoJSON polygon coords: [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]]
                                            coords = feature['geometry']['coordinates'][0]
                                            # Extract min/max (assuming axis-aligned for simplicity, might need rotation handling)
                                            xmin = min(p[0] for p in coords)
                                            ymin = min(p[1] for p in coords)
                                            xmax = max(p[0] for p in coords)
                                            ymax = max(p[1] for p in coords)
                                            # Store as [xmin, ymin, xmax, ymax] - adjust format as needed for your model
                                            bboxes.append([xmin, ymin, xmax, ymax])
                                        except (KeyError, IndexError, TypeError) as feature_err:
                                            logger.warning(f"Could not extract bbox from feature in {member.name}: {feature_err}")
                                            continue # Skip this feature

                                    if bboxes: # Only add if bboxes were successfully extracted
                                         all_annotations[image_id] = bboxes
                                         total_features_processed += len(bboxes)
                                         processed_file_count += 1
                            else:
                                 logger.warning(f"Could not extract file object for {member.name}")
                    except json.JSONDecodeError as json_err:
                        logger.error(f"Error decoding JSON from {member.name}: {json_err}")
                    except Exception as inner_e:
                         logger.exception(f"Error processing member {member.name}: {inner_e}")

    except tarfile.TarError as e:
        logger.error(f"Error opening or reading tar archive {annotation_archive_path}: {e}")
        return None, None
    except Exception as e:
        logger.exception(f"An unexpected error occurred during tar extraction: {e}")
        return None, None

    end_time = time.time()
    logger.info(f"Finished processing annotation archive in {end_time - start_time:.2f} seconds.")
    logger.info(f"Found {total_features_processed} annotations across {len(all_annotations)} images (out of {processed_file_count} GeoJSON files with features).")

    # Store or pass 'all_annotations' for the next step (matching with images)
    # For now, just log summary and return None for datasets
    return None, None # (train_dataset, validation_dataset)

# --- start_training function remains the same ---
def start_training(data_dir: str, epochs: int = 5, batch_size: int = 32, learning_rate: float = 0.001):
    logger.info("--- Starting Model Training (Placeholder Steps) ---")
    logger.info(f"Parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    # Load data step now processes all annotations
    train_ds, val_ds = load_rareplanes_data(Path(data_dir))
    if train_ds is None: logger.warning("Data loading returned None (expected at this stage). Cannot proceed with actual training steps.")
    # Build Model
    input_shape = (256, 256, 3); num_classes = 1
    model = build_simple_cnn(input_shape=input_shape, num_classes=num_classes)
    # Compile Model
    logger.info("Compiling model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    # Train Model Placeholder
    if train_ds is not None: logger.info(f"Placeholder: Would train for {epochs} epochs.")
    else: logger.warning("Skipping model fitting placeholder as data was not loaded.")
    # Save Model Placeholder
    model_save_path = MODEL_SAVE_DIR / "simple_cnn_placeholder.keras"
    logger.info(f"Placeholder: Saving model to {model_save_path}")
    logger.info("--- Model Training Script Finished (Placeholder Steps) ---")
    return {"status": "Training script finished (placeholder steps, all annotations parsed)", "model_path": str(model_save_path)}

