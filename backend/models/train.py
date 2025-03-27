# backend/models/train.py
import tensorflow as tf
import logging
from pathlib import Path
import tarfile
import json
import time
import io
import numpy as np
from PIL import Image # Ensure PIL is imported

# Import model building function
from .model import build_simple_cnn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

MODEL_SAVE_DIR = Path(__file__).resolve().parent.parent / "saved_models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- Configuration ---
TARGET_IMG_WIDTH = 256  # Match placeholder model input
TARGET_IMG_HEIGHT = 256
MAX_IMAGES_TO_LOAD = 10 # Limit for initial testing

def load_rareplanes_data(data_dir: Path):
    """
    Loads RarePlanes data. Parses annotations, then loops through a
    limited number of images, extracting, loading, resizing, and normalizing them.
    """
    logger.info(f"Attempting to load annotation data from {data_dir}...")
    start_time = time.time()
    annotation_archive_name = "RarePlanes_train_geojson_aircraft_tiled.tar.gz"
    annotation_archive_path = data_dir / annotation_archive_name

    # --- Annotation Processing ---
    all_annotations = {}
    # (Code remains the same as before to populate all_annotations)
    # ... [omitted for brevity, assume it populates all_annotations correctly] ...
    total_features_processed = 0 # Need to calculate this if code omitted
    if not annotation_archive_path.exists(): logger.error(f"Anno archive not found: {annotation_archive_path}"); return None, None
    try:
        with tarfile.open(annotation_archive_path, "r:gz") as tar:
            members = tar.getmembers(); total_members = len(members)
            for idx, member in enumerate(members):
                if member.isfile() and member.name.lower().endswith('.geojson'):
                    image_id = Path(member.name).stem
                    try:
                        with tar.extractfile(member) as f:
                            if f:
                                data = json.loads(f.read().decode('utf-8'))
                                features = data.get('features', [])
                                if features:
                                    bboxes = [[min(p[0] for p in feat['geometry']['coordinates'][0]), min(p[1] for p in feat['geometry']['coordinates'][0]), max(p[0] for p in feat['geometry']['coordinates'][0]), max(p[1] for p in feat['geometry']['coordinates'][0])] for feat in features if 'geometry' in feat and feat['geometry']]
                                    if bboxes:
                                         all_annotations.setdefault(image_id, []).extend(bboxes)
                                         total_features_processed += len(bboxes)
                    except Exception: continue # Simplified error handling
    except Exception as e: logger.exception(f"Error processing annotation archive: {e}"); return None, None
    logger.info(f"Finished annotation processing. Found {total_features_processed} annotations across {len(all_annotations)} images.")
    if not all_annotations: logger.error("No annotations loaded."); return None, None
    # --- End Annotation Processing ---


    # --- Start: Loop Image Extraction, Loading & Preprocessing ---
    image_archive_name = "RarePlanes_train_PS-RGB_tiled.tar.gz"
    image_archive_path = data_dir / image_archive_name
    processed_images = []
    processed_annotations = []
    processed_image_count = 0

    if not image_archive_path.exists(): logger.error(f"Image archive not found: {image_archive_path}"); return None, None

    try:
        logger.info(f"Opening image archive: {image_archive_path}...")
        with tarfile.open(image_archive_path, "r:gz") as tar:
            # Loop through images that have annotations
            for image_id, bboxes in all_annotations.items():
                if processed_image_count >= MAX_IMAGES_TO_LOAD:
                    logger.info(f"Reached processing limit ({MAX_IMAGES_TO_LOAD} images). Stopping image loading.")
                    break # Stop after processing the limit

                image_member_name = f"./PS-RGB_tiled/{image_id}.png"
                logger.info(f"Processing image ID '{image_id}' ({processed_image_count + 1}/{MAX_IMAGES_TO_LOAD})...")

                try:
                    image_member_info = tar.getmember(image_member_name)
                    with tar.extractfile(image_member_info) as f_img:
                        if f_img:
                            image_bytes = f_img.read()
                            # Load with Pillow
                            image = Image.open(io.BytesIO(image_bytes))

                            # --- Preprocessing ---
                            # 1. Resize
                            image_resized = image.resize((TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT))
                            image_np = np.array(image_resized)

                            # 2. Ensure 3 Channels (e.g., handle grayscale if necessary)
                            if image_np.ndim == 2: # Grayscale
                                 image_np = np.stack((image_np,)*3, axis=-1)
                                 logger.info(f"  - Converted grayscale to 3 channels.")
                            elif image_np.shape[2] == 4: # RGBA
                                 image_np = image_np[:, :, :3] # Drop alpha channel
                                 logger.info(f"  - Dropped alpha channel.")

                            if image_np.shape[2] != 3:
                                 logger.warning(f"  - Image has unexpected shape {image_np.shape}. Skipping.")
                                 continue

                            # 3. Normalize (scale pixels to 0-1)
                            image_norm = image_np.astype(np.float32) / 255.0
                            # --- End Preprocessing ---

                            logger.info(f"  - Extracted and processed image. Shape: {image_norm.shape}, Dtype: {image_norm.dtype}")

                            # Store the processed data
                            processed_images.append(image_norm)
                            processed_annotations.append(bboxes) # Store corresponding bboxes
                            processed_image_count += 1
                        else: logger.warning(f"Could not extract file object for {image_member_name}")
                except KeyError:
                    logger.warning(f"Image member '{image_member_name}' not found in archive. Skipping.")
                except Exception as img_proc_err:
                     logger.exception(f"Error processing image {image_id}: {img_proc_err}")

    except tarfile.TarError as e: logger.error(f"Error opening image archive {image_archive_path}: {e}"); return None, None
    except Exception as e: logger.exception(f"An unexpected error occurred during image processing loop: {e}"); return None, None
    # --- End: Image Processing Loop ---

    logger.info(f"Successfully processed {len(processed_images)} image-annotation pairs.")
    logger.info("Data loading step finished.")

    # TODO: Convert processed_images and processed_annotations into tf.data.Dataset
    # For now, just returning None
    return None, None

# --- start_training function remains the same ---
def start_training(data_dir: str, epochs: int = 5, batch_size: int = 32, learning_rate: float = 0.001):
    logger.info("--- Starting Model Training (Placeholder Steps) ---")
    logger.info(f"Parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    # Load data step now processes multiple images
    train_ds, val_ds = load_rareplanes_data(Path(data_dir))
    if train_ds is None: logger.warning("Data loading returned None (expected)...")
    input_shape=(TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT, 3); num_classes=1 # Use constants
    model = build_simple_cnn(input_shape=input_shape, num_classes=num_classes)
    logger.info("Compiling model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    if train_ds is not None: logger.info(f"Placeholder: Would train for {epochs} epochs.")
    else: logger.warning("Skipping model fitting placeholder...")
    model_save_path = MODEL_SAVE_DIR / "simple_cnn_placeholder.keras"
    logger.info(f"Placeholder: Saving model to {model_save_path}")
    logger.info("--- Model Training Script Finished (Placeholder Steps) ---")
    # Update status message
    return {"status": f"Training script finished (placeholder steps, processed {MAX_IMAGES_TO_LOAD} images)", "model_path": str(model_save_path)}

