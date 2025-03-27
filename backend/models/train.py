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

# --- Helper Function to Load Annotations ---
def _load_annotations(data_dir: Path) -> dict:
    # [ Ensure FULL, WORKING annotation loading logic is here ]
    logger.info(f"Loading annotation data from {data_dir}...")
    start_time = time.time()
    annotation_archive_name = "RarePlanes_train_geojson_aircraft_tiled.tar.gz"
    annotation_archive_path = data_dir / annotation_archive_name
    all_annotations = {}
    total_features_processed = 0
    processed_anno_file_count = 0

    if not annotation_archive_path.exists(): logger.error(f"Anno archive not found: {annotation_archive_path}"); return {}
    try:
        with tarfile.open(annotation_archive_path, "r:gz") as tar:
            members = tar.getmembers(); total_members = len(members)
            logger.info(f"Found {total_members} anno members. Iterating...")
            for idx, member in enumerate(members):
                if idx > 0 and idx % 1000 == 0: logger.info(f"  Processed {idx}/{total_members} anno members...")
                if member.isfile() and member.name.lower().endswith('.geojson'):
                    image_id = Path(member.name).stem
                    try:
                        with tar.extractfile(member) as f:
                            if f:
                                data = json.loads(f.read().decode('utf-8'))
                                features = data.get('features', [])
                                if features:
                                    bboxes_for_image = [[min(p[0] for p in feat['geometry']['coordinates'][0]), min(p[1] for p in feat['geometry']['coordinates'][0]), max(p[0] for p in feat['geometry']['coordinates'][0]), max(p[1] for p in feat['geometry']['coordinates'][0])] for feat in features if 'geometry' in feat and feat['geometry']]
                                    if bboxes_for_image:
                                         all_annotations.setdefault(image_id, []).extend(bboxes_for_image)
                                         total_features_processed += len(bboxes_for_image)
                                         processed_anno_file_count += 1
                    except Exception as inner_e: logger.warning(f"Skipping anno member {member.name}: {inner_e}")
    except Exception as e: logger.exception(f"Error processing annotation archive: {e}"); return {}
    logger.info(f"Finished anno processing in {time.time() - start_time:.2f}s. Found {total_features_processed} annos across {len(all_annotations)} images (from {processed_anno_file_count} files).")
    if not all_annotations: logger.error("Annotation dictionary is empty after processing!")
    return all_annotations

# --- Main Data Loading Function ---
def load_rareplanes_data(data_dir: Path, max_items_to_process: int | None = None) -> tf.data.Dataset | None:
    """Loads annotations and creates a tf.data.Dataset using an inner generator."""
    all_annotations = _load_annotations(data_dir)
    if not all_annotations:
         logger.error("Cannot create dataset because annotation loading failed.")
         return None

    image_archive_path = data_dir / "RarePlanes_train_PS-RGB_tiled.tar.gz"
    if not image_archive_path.exists(): logger.error(f"Image archive missing: {image_archive_path}"); return None

    # --- Inner Python Generator Function (Handle max_items=-1 for no limit) ---
    def _data_generator(target_height: int, target_width: int, max_items: int): # Now expects int, uses -1 for no limit
        logger.info(f"Data generator starting. Processing up to {'all' if max_items == -1 else max_items} items.")
        processed_count = 0
        try:
            logger.info(f"Generator: Attempting to open image archive: {image_archive_path}")
            with tarfile.open(image_archive_path, "r:gz") as tar:
                logger.info("Generator: Image archive opened.")
                for image_id, bboxes_orig in all_annotations.items():
                    logger.debug(f"Generator loop: Processing image_id '{image_id}'")
                    # Check for limit using -1 convention
                    if max_items != -1 and processed_count >= max_items:
                        logger.info(f"Generator reached limit ({max_items}). Breaking loop.")
                        break

                    image_member_name = f"./PS-RGB_tiled/{image_id}.png"
                    try:
                        image_member_info = tar.getmember(image_member_name)
                        with tar.extractfile(image_member_info) as f_img:
                            if f_img:
                                # --- [ Image processing logic as before ] ---
                                image_bytes = f_img.read()
                                image = Image.open(io.BytesIO(image_bytes))
                                image_resized = image.resize((target_width, target_height))
                                image_np = np.array(image_resized)
                                if image_np.ndim == 2: image_np = np.stack((image_np,)*3, axis=-1)
                                if image_np.shape[2] == 4: image_np = image_np[:, :, :3]
                                if image_np.shape[2] != 3: logger.warning(f"Generator: Skipping {image_id}: bad shape {image_np.shape}"); continue
                                image_norm = image_np.astype(np.float32) / 255.0
                                processed_bboxes = [[0.25, 0.25, 0.75, 0.75] for _ in bboxes_orig]

                                logger.info(f"Generator: Yielding item {processed_count + 1} for {image_id}")
                                yield image_norm, np.array(processed_bboxes, dtype=np.float32)
                                processed_count += 1
                            else: logger.warning(f"Generator: Could not extract file object for {image_member_name}")
                    except KeyError: logger.warning(f"Generator: Image member '{image_member_name}' not found. Skipping.")
                    except Exception as img_proc_err: logger.exception(f"Generator: Error processing pair for {image_id}: {img_proc_err}")

                logger.info(f"Generator: Finished iterating through annotation keys.")
        except tarfile.TarError as e: logger.error(f"Generator: Error opening image archive: {e}")
        except Exception as e: logger.exception(f"Generator: Error: {e}")
        finally: logger.info(f"Data generator finished after yielding {processed_count} items.")
    # --- End Inner Generator Function ---

    output_signature = (
        tf.TensorSpec(shape=(TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
    )

    # Convert None to -1 for TensorFlow argument compatibility
    max_items_arg = max_items_to_process if max_items_to_process is not None else -1

    try:
        logger.info("Creating tf.data.Dataset from generator...")
        tf_dataset = tf.data.Dataset.from_generator(
            _data_generator,
            args=[TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, max_items_arg], # Pass -1 if no limit specified
            output_signature=output_signature
        )
        logger.info("tf.data.Dataset created successfully.")
        return tf_dataset
    except Exception as e:
        logger.exception(f"Failed to create tf.data.Dataset: {e}")
        return None

# --- Training Orchestration ---
def start_training(data_dir: str, epochs: int = 5, batch_size: int = 32, learning_rate: float = 0.001):
    logger.info("--- Starting Model Training ---")
    logger.info(f"Parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

    # Load Full Dataset (max_items_to_process=None)
    tf_dataset = load_rareplanes_data(Path(data_dir), max_items_to_process=None)

    if tf_dataset is None:
        logger.error("Failed to create dataset. Aborting training.")
        return {"status": "Failed: Dataset creation error", "model_path": None}

    # Enhance Pipeline
    buffer_size = 1000 # Adjust if needed
    logger.info(f"Shuffling dataset with buffer size {buffer_size}")
    tf_dataset = tf_dataset.shuffle(buffer_size=buffer_size)
    logger.info(f"Batching dataset with batch size {batch_size}")
    tf_dataset = tf_dataset.batch(batch_size)
    logger.info("Applying prefetching to dataset")
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    # Placeholder steps...
    input_shape=(TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, 3); num_classes=1
    model = build_simple_cnn(input_shape=input_shape, num_classes=num_classes)
    model.summary(print_fn=logger.info)
    logger.info("Compiling model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    logger.info(f"Placeholder: Starting model training for {epochs} epochs.")
    # history = model.fit(tf_dataset, epochs=epochs) # Actual training call
    model_save_path = MODEL_SAVE_DIR / "simple_cnn_placeholder.keras"
    logger.info(f"Placeholder: Saving model to {model_save_path}")
    # model.save(model_save_path) # Actual save call
    logger.info("--- Model Training Script Finished (Placeholder Steps) ---")
    final_status = "Training script finished (Dataset pipeline configured for ALL data, placeholder fit)"
    return {"status": final_status, "model_path": str(model_save_path)}

