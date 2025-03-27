# backend/models/train.py
import tensorflow as tf
import logging
from pathlib import Path
import tarfile
import json
import time

# Import model building function
from .model import build_simple_cnn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define paths
MODEL_SAVE_DIR = Path(__file__).resolve().parent.parent / "saved_models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

def load_rareplanes_data(data_dir: Path):
    """
    Loads RarePlanes data. Parses all GeoJSON annotations, then extracts
    raw bytes for the first corresponding image file found.
    """
    logger.info(f"Attempting to load annotation data from {data_dir}...")
    start_time = time.time()
    annotation_archive_name = "RarePlanes_train_geojson_aircraft_tiled.tar.gz"
    annotation_archive_path = data_dir / annotation_archive_name

    # --- Annotation Processing ---
    all_annotations = {}
    total_features_processed = 0
    processed_file_count = 0
    if not annotation_archive_path.exists(): logger.error(f"Annotation archive not found: {annotation_archive_path}"); return None, None
    try:
        logger.info(f"Opening annotation archive: {annotation_archive_path}...")
        with tarfile.open(annotation_archive_path, "r:gz") as tar:
            members = tar.getmembers()
            total_members = len(members)
            logger.info(f"Found {total_members} annotation members. Iterating...")
            for idx, member in enumerate(members):
                if idx > 0 and idx % 1000 == 0: logger.info(f"  Processed {idx}/{total_members} annotation members...")
                if member.isfile() and member.name.lower().endswith('.geojson'):
                    image_id = Path(member.name).stem
                    try:
                        with tar.extractfile(member) as f:
                            if f:
                                data = json.loads(f.read().decode('utf-8'))
                                features = data.get('features', [])
                                if features:
                                    bboxes = []
                                    for feature in features:
                                        try:
                                            coords = feature['geometry']['coordinates'][0]
                                            xmin, ymin = min(p[0] for p in coords), min(p[1] for p in coords)
                                            xmax, ymax = max(p[0] for p in coords), max(p[1] for p in coords)
                                            bboxes.append([xmin, ymin, xmax, ymax])
                                        except Exception: continue
                                    if bboxes:
                                         all_annotations.setdefault(image_id, []).extend(bboxes)
                                         total_features_processed += len(bboxes)
                                         processed_file_count += 1
                    except Exception as inner_e: logger.warning(f"Skipping member {member.name} due to error: {inner_e}")
    except Exception as e: logger.exception(f"Error processing annotation archive: {e}"); return None, None
    logger.info(f"Finished annotation processing in {time.time() - start_time:.2f} seconds.")
    if not all_annotations: logger.error("No annotations were successfully loaded."); return None, None
    logger.info(f"Found {total_features_processed} annotations across {len(all_annotations)} images.")
    # --- End Annotation Processing ---


    # --- Start: Image Extraction Logic (First Image Only - CORRECTED PATH) ---
    image_archive_name = "RarePlanes_train_PS-RGB_tiled.tar.gz"
    image_archive_path = data_dir / image_archive_name
    image_extracted = False

    if not image_archive_path.exists(): logger.error(f"Image archive not found: {image_archive_path}"); return None, None

    try:
        first_image_id = next(iter(all_annotations))
        # --- CORRECTED: Use .png extension ---
        image_member_name = f"./PS-RGB_tiled/{first_image_id}.png"
        # --- End Correction ---
        logger.info(f"Attempting to extract image bytes for ID '{first_image_id}' from member '{image_member_name}'...")

        with tarfile.open(image_archive_path, "r:gz") as tar:
            try:
                image_member_info = tar.getmember(image_member_name) # Find specific member
                logger.info(f"Found image member: {image_member_info.name}, Size: {image_member_info.size}")
                with tar.extractfile(image_member_info) as f_img: # Extract file object
                    if f_img:
                        image_bytes = f_img.read() # Read raw bytes
                        logger.info(f"Successfully extracted {len(image_bytes)} bytes for image '{first_image_id}'.")
                        image_extracted = True
                        # TODO: Process image_bytes with PIL/OpenCV
                    else: logger.error(f"Could not extract file object for image member: {image_member_name}")
            except KeyError:
                logger.error(f"Image member '{image_member_name}' not found in archive {image_archive_path}.")
                # Optional: List some members if debugging needed
                # logger.info("First few members in image archive:")
                # for m in tar.getmembers()[:10]: logger.info(f"  - {m.name}")

    except tarfile.TarError as e: logger.error(f"Error opening image archive {image_archive_path}: {e}"); return None, None
    except StopIteration: logger.error("Cannot extract image - annotation dictionary was empty."); return None, None
    except Exception as e: logger.exception(f"An unexpected error occurred during image extraction: {e}"); return None, None
    # --- End: Image Extraction Logic ---

    if not image_extracted: logger.warning("Failed to extract the first test image.")

    logger.info("Data loading step finished (annotations processed, first image bytes extracted).")
    return None, None

# --- start_training function remains the same ---
def start_training(data_dir: str, epochs: int = 5, batch_size: int = 32, learning_rate: float = 0.001):
    logger.info("--- Starting Model Training (Placeholder Steps) ---")
    logger.info(f"Parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    train_ds, val_ds = load_rareplanes_data(Path(data_dir))
    if train_ds is None: logger.warning("Data loading returned None (expected at this stage)...")
    input_shape=(256, 256, 3); num_classes=1 # Placeholders
    model = build_simple_cnn(input_shape=input_shape, num_classes=num_classes)
    logger.info("Compiling model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    if train_ds is not None: logger.info(f"Placeholder: Would train for {epochs} epochs.")
    else: logger.warning("Skipping model fitting placeholder...")
    model_save_path = MODEL_SAVE_DIR / "simple_cnn_placeholder.keras"
    logger.info(f"Placeholder: Saving model to {model_save_path}")
    logger.info("--- Model Training Script Finished (Placeholder Steps) ---")
    # Update status message
    return {"status": "Training script finished (placeholder steps, annotations processed, first image extracted - path fixed)", "model_path": str(model_save_path)}

