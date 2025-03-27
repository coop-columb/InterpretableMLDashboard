# backend/models/train.py
import tensorflow as tf
import logging
from pathlib import Path
import tarfile
import json
import time
import io
import numpy as np
from PIL import Image

# Import model building function
from .model import build_simple_cnn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

MODEL_SAVE_DIR = Path(__file__).resolve().parent.parent / "saved_models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- Configuration ---
TARGET_IMG_WIDTH = 256
TARGET_IMG_HEIGHT = 256
MAX_IMAGES_TO_LOAD = 10 # Keep limit for testing

def load_rareplanes_data(data_dir: Path):
    """
    Loads RarePlanes data. Parses all GeoJSON annotations, processes a limited
    number of images (load, resize, normalize) and generates DUMMY processed annotations.
    """
    logger.info(f"Attempting to load annotation data from {data_dir}...")
    start_time = time.time()
    annotation_archive_name = "RarePlanes_train_geojson_aircraft_tiled.tar.gz"
    annotation_archive_path = data_dir / annotation_archive_name

    # --- Annotation Processing (MUST BE PRESENT!) ---
    all_annotations = {}
    total_features_processed = 0
    processed_anno_file_count = 0

    if not annotation_archive_path.exists(): logger.error(f"Anno archive not found: {annotation_archive_path}"); return None, None
    try:
        logger.info(f"Opening annotation archive: {annotation_archive_path}...")
        with tarfile.open(annotation_archive_path, "r:gz") as tar:
            members = tar.getmembers(); total_members = len(members)
            logger.info(f"Found {total_members} annotation members. Iterating...")
            for idx, member in enumerate(members):
                # Log progress occasionally
                if idx > 0 and idx % 1000 == 0: logger.info(f"  Processed {idx}/{total_members} annotation members...")

                # Ensure it's a geojson file we process
                if member.isfile() and member.name.lower().endswith('.geojson'):
                    image_id = Path(member.name).stem # Get ID from filename stem
                    try:
                        with tar.extractfile(member) as f:
                            if f:
                                data = json.loads(f.read().decode('utf-8')) # Read, decode, parse
                                features = data.get('features', [])
                                if features: # Only process if features exist
                                    bboxes_for_image = [] # Collect bboxes for *this* image
                                    for feature in features:
                                        try:
                                            # Extract original bbox coordinates
                                            coords = feature['geometry']['coordinates'][0]
                                            xmin, ymin = min(p[0] for p in coords), min(p[1] for p in coords)
                                            xmax, ymax = max(p[0] for p in coords), max(p[1] for p in coords)
                                            # Store original pixel/geo coordinates
                                            bboxes_for_image.append([xmin, ymin, xmax, ymax])
                                        except Exception as feature_err:
                                            logger.debug(f"Skipping feature due to error in {member.name}: {feature_err}")
                                            continue # Skip malformed feature

                                    # Add to main dictionary if bboxes were found for this image
                                    if bboxes_for_image:
                                         # Use setdefault to ensure list exists, then extend
                                         all_annotations.setdefault(image_id, []).extend(bboxes_for_image)
                                         total_features_processed += len(bboxes_for_image)
                                         processed_anno_file_count += 1
                    except Exception as inner_e:
                         logger.warning(f"Skipping annotation member {member.name} due to error: {inner_e}")
    except Exception as e:
        logger.exception(f"Error processing annotation archive: {e}"); return None, None

    logger.info(f"Finished annotation processing in {time.time() - start_time:.2f} seconds.")
    if not all_annotations:
        logger.error("No annotations were successfully loaded. Cannot proceed.")
        return None, None # Critical error, stop here
    logger.info(f"Found {total_features_processed} annotations across {len(all_annotations)} images (from {processed_anno_file_count} files).")
    # --- End Annotation Processing ---


    # --- Start: Loop Image & DUMMY Annotation Processing ---
    image_archive_name = "RarePlanes_train_PS-RGB_tiled.tar.gz"
    image_archive_path = data_dir / image_archive_name
    processed_images = []
    processed_annotations = [] # List of lists of DUMMY processed bboxes
    processed_image_count = 0

    if not image_archive_path.exists(): logger.error(f"Image archive not found: {image_archive_path}"); return None, None

    try:
        logger.info(f"Opening image archive to process up to {MAX_IMAGES_TO_LOAD} images: {image_archive_path}...")
        with tarfile.open(image_archive_path, "r:gz") as tar:
            # Now loop through the images *that have annotations*
            for image_id, bboxes_orig in all_annotations.items():
                if processed_image_count >= MAX_IMAGES_TO_LOAD:
                    logger.info(f"Reached processing limit ({MAX_IMAGES_TO_LOAD} images).")
                    break

                image_member_name = f"./PS-RGB_tiled/{image_id}.png" # Correct extension
                logger.info(f"Processing image ID '{image_id}' ({processed_image_count + 1}/{MAX_IMAGES_TO_LOAD})...")

                try:
                    image_member_info = tar.getmember(image_member_name)
                    with tar.extractfile(image_member_info) as f_img:
                        if f_img:
                            image_bytes = f_img.read()
                            image = Image.open(io.BytesIO(image_bytes))
                            # Image Preprocessing (remains the same)
                            image_resized = image.resize((TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT))
                            image_np = np.array(image_resized)
                            if image_np.ndim == 2: image_np = np.stack((image_np,)*3, axis=-1)
                            if image_np.shape[2] == 4: image_np = image_np[:, :, :3]
                            if image_np.shape[2] != 3: logger.warning(f"Unexpected shape {image_np.shape}. Skipping."); continue
                            image_norm = image_np.astype(np.float32) / 255.0
                            logger.info(f"  - Processed image. Shape: {image_norm.shape}, Dtype: {image_norm.dtype}")

                            # Annotation Preprocessing (Simplified - DUMMY DATA)
                            # TODO: Implement correct GeoJSON -> pixel coord transformation here later!
                            processed_bboxes = []
                            for _ in bboxes_orig: # Loop same number of times as original bboxes
                                dummy_bbox = [0.25, 0.25, 0.75, 0.75] # [ymin, xmin, ymax, xmax] format
                                processed_bboxes.append(dummy_bbox)

                            # Log example transformation for the first image processed
                            if processed_image_count == 0 and bboxes_orig and processed_bboxes:
                                 logger.info(f"  - Original bbox example: {bboxes_orig[0]} (NOTE: Geographic, not pixel!)")
                                 logger.info(f"  - Processed bbox example (DUMMY): {processed_bboxes[0]}")

                            # Store pair
                            processed_images.append(image_norm)
                            processed_annotations.append(processed_bboxes) # Append the list of DUMMY boxes
                            processed_image_count += 1

                        else: logger.warning(f"Could not extract file object for {image_member_name}")
                except KeyError: logger.warning(f"Image member '{image_member_name}' not found. Skipping.")
                except Exception as img_proc_err: logger.exception(f"Error processing image/annotations for {image_id}: {img_proc_err}")
    except Exception as e: logger.exception(f"Error during image processing loop: {e}"); return None, None
    # --- End Image & Annotation Processing Loop ---

    logger.info(f"Successfully processed {len(processed_images)} image-DUMMY_annotation pairs.")
    logger.info("Data loading step finished.")
    # TODO: Convert processed_images and processed_annotations into tf.data.Dataset
    return None, None

# --- start_training function remains the same ---
def start_training(data_dir: str, epochs: int = 5, batch_size: int = 32, learning_rate: float = 0.001):
    logger.info("--- Starting Model Training (Placeholder Steps) ---")
    logger.info(f"Parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    train_ds, val_ds = load_rareplanes_data(Path(data_dir))
    if train_ds is None: logger.warning("Data loading returned None (expected)...")
    input_shape=(TARGET_IMG_WIDTH, TARGET_IMG_HEIGHT, 3); num_classes=1
    model = build_simple_cnn(input_shape=input_shape, num_classes=num_classes)
    logger.info("Compiling model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    if train_ds is not None: logger.info(f"Placeholder: Would train for {epochs} epochs.")
    else: logger.warning("Skipping model fitting placeholder...")
    model_save_path = MODEL_SAVE_DIR / "simple_cnn_placeholder.keras"
    logger.info(f"Placeholder: Saving model to {model_save_path}")
    logger.info("--- Model Training Script Finished (Placeholder Steps) ---")
    return {"status": f"Training script finished (placeholder steps, processed {MAX_IMAGES_TO_LOAD} images & DUMMY annos)", "model_path": str(model_save_path)}

