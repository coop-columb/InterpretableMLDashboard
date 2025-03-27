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
import xml.etree.ElementTree as ET

from .model import build_simple_cnn, MAX_BOXES # Import MAX_BOXES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

MODEL_SAVE_DIR = Path(__file__).resolve().parent.parent / "saved_models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_IMG_WIDTH = 256
TARGET_IMG_HEIGHT = 256

# --- Helper: Load Annotations ---
def _load_annotations(data_dir: Path) -> dict:
    # (No changes needed here)
    logger.info(f"Loading annotation data from {data_dir}...")
    start_time = time.time(); all_annotations = {}; total_features_processed = 0; processed_anno_file_count = 0
    annotation_archive_path = data_dir / "RarePlanes_train_geojson_aircraft_tiled.tar.gz"
    if not annotation_archive_path.exists(): logger.error(f"Anno archive not found: {annotation_archive_path}"); return {}
    try:
        with tarfile.open(annotation_archive_path, "r:gz") as tar:
            members = tar.getmembers(); logger.info(f"Found {len(members)} anno members. Iterating...")
            for idx, member in enumerate(members):
                if member.isfile() and member.name.lower().endswith('.geojson'):
                    image_id = Path(member.name).stem
                    try:
                        with tar.extractfile(member) as f:
                            if f:
                                data = json.loads(f.read().decode('utf-8')); features = data.get('features', [])
                                if features:
                                    bboxes_for_image = []
                                    for feature in features:
                                        try: coords = feature['geometry']['coordinates'][0]; bboxes_for_image.append([min(p[0] for p in coords), min(p[1] for p in coords), max(p[0] for p in coords), max(p[1] for p in coords)])
                                        except Exception: continue
                                    if bboxes_for_image: all_annotations.setdefault(image_id, []).extend(bboxes_for_image); total_features_processed += len(bboxes_for_image); processed_anno_file_count += 1
                    except Exception as inner_e: logger.warning(f"Skipping anno member {member.name}: {inner_e}")
    except Exception as e: logger.exception(f"Error processing annotation archive: {e}"); return {}
    logger.info(f"Finished anno processing in {time.time() - start_time:.2f}s. Found {total_features_processed} annos across {len(all_annotations)} images.")
    return all_annotations

# --- Helper: Parse GeoTransform ---
def _get_geotransform_from_xml(xml_path: Path) -> tuple[float, ...] | None:
    # (No changes needed here)
    if not xml_path.exists(): logger.warning(f"Metadata file not found: {xml_path}"); return None
    try:
        tree = ET.parse(xml_path); root = tree.getroot(); geotransform_str = None
        possible_paths = [ ".//Metadata/Item[@name='GEOTRANSFORM']", ".//GeoTransform" ]
        for path in possible_paths: elem = root.find(path); if elem is not None and elem.text: geotransform_str = elem.text; break
        if geotransform_str: parts = geotransform_str.replace(',', ' ').split();
        if len(parts) == 6: return tuple(float(p) for p in parts); logger.warning(f"Could not parse GeoTransform string '{geotransform_str}' in {xml_path}")
        else: logger.warning(f"Could not find GeoTransform metadata item in {xml_path}")
        return None
    except Exception as e: logger.error(f"Error reading GeoTransform from {xml_path}: {e}"); return None

# --- Helper: Geo to Pixel ---
def _geo_to_pixel(lon: float, lat: float, gt: tuple[float, ...]) -> tuple[float, float]|None:
    # (No changes needed here)
    if gt is None or len(gt)!=6: return None; ulx, x_res, _, uly, _, y_res = gt
    if x_res == 0 or y_res == 0: logger.warning("Invalid GeoTransform: x_res or y_res is zero."); return None
    x_pixel = (lon - ulx) / x_res; y_pixel = (lat - uly) / y_res; return (x_pixel, y_pixel)

# --- Main Data Loading Function (Pad BBoxes, Update Signature) ---
def load_rareplanes_data(data_dir: Path, max_items_to_process: int | None = None) -> tf.data.Dataset | None:
    all_annotations = _load_annotations(data_dir)
    if not all_annotations: return None
    image_archive_path = data_dir / "RarePlanes_train_PS-RGB_tiled.tar.gz"
    image_metadata_dir = data_dir / "RarePlanes_train_PS-RGB_tiled"
    if not image_archive_path.exists(): logger.error(f"Image archive missing: {image_archive_path}"); return None

    def _data_generator(target_height: int, target_width: int, max_items: int):
        logger.info(f"Data generator starting. Processing up to {'all' if max_items == -1 else max_items} items.")
        processed_count, yielded_count, skipped_xml, skipped_bbox, skipped_img = 0, 0, 0, 0, 0
        try:
            with tarfile.open(image_archive_path, "r:gz") as tar:
                for image_id, bboxes_geo in all_annotations.items():
                    if max_items != -1 and yielded_count >= max_items: break
                    processed_count += 1
                    if processed_count % 1000 == 0: logger.info(f"Generator processed {processed_count}/{len(all_annotations)} potential images...")

                    xml_filename=f"{image_id}.png.aux.xml"; xml_member_name=f"./PS-RGB_tiled/{xml_filename}"; xml_path_absolute=image_metadata_dir/xml_filename
                    geotransform = None
                    try: # Try tar first
                        member_info = tar.getmember(xml_member_name)
                        with tar.extractfile(member_info) as f_xml:
                            if f_xml:
                                xml_content=f_xml.read(); tree=ET.ElementTree(ET.fromstring(xml_content)); root=tree.getroot(); gt_str=None
                                possible_paths=[".//Metadata/Item[@name='GEOTRANSFORM']", ".//GeoTransform"];
                                for path in possible_paths: elem = root.find(path); if elem is not None and elem.text: gt_str = elem.text; break
                                if gt_str: parts=gt_str.replace(',',' ').split();
                                if len(parts)==6: geotransform=tuple(float(p) for p in parts)
                    except KeyError: geotransform = _get_geotransform_from_xml(xml_path_absolute) # Try external file
                    except Exception as xml_err: logger.error(f"Error getting GeoTransform for {image_id}: {xml_err}")

                    if geotransform is None: skipped_xml += 1; continue

                    image_member_name = f"./PS-RGB_tiled/{image_id}.png"
                    try:
                        image_member_info = tar.getmember(image_member_name)
                        with tar.extractfile(image_member_info) as f_img:
                            if f_img:
                                image_bytes = f_img.read(); image = Image.open(io.BytesIO(image_bytes)); orig_w, orig_h = image.size
                                image_resized = image.resize((target_width, target_height)); image_np = np.array(image_resized)
                                if image_np.ndim == 2: image_np = np.stack((image_np,)*3, axis=-1)
                                if image_np.shape[2] == 4: image_np = image_np[:, :, :3]
                                if image_np.shape[2] != 3: skipped_img+=1; continue
                                image_norm = image_np.astype(np.float32) / 255.0

                                processed_bboxes = []
                                for xmin_geo, ymin_geo, xmax_geo, ymax_geo in bboxes_geo:
                                    tl_pix = _geo_to_pixel(xmin_geo, ymax_geo, geotransform); br_pix = _geo_to_pixel(xmax_geo, ymin_geo, geotransform)
                                    if tl_pix is None or br_pix is None: skipped_bbox+=1; continue
                                    xmin_pix, ymin_pix = tl_pix; xmax_pix, ymax_pix = br_pix
                                    xmin_pix=max(0.,xmin_pix); ymin_pix=max(0.,ymin_pix); xmax_pix=min(float(orig_w),xmax_pix); ymax_pix=min(float(orig_h),ymax_pix)
                                    if xmax_pix<=xmin_pix or ymax_pix<=ymin_pix: skipped_bbox+=1; continue
                                    x_s=float(target_width)/orig_w; y_s=float(target_height)/orig_h
                                    xmin_p_s=xmin_pix*x_s; ymin_p_s=ymin_pix*y_s; xmax_p_s=xmax_pix*x_s; ymax_p_s=ymax_pix*y_s
                                    xmin_n=xmin_p_s/target_width; ymin_n=ymin_p_s/target_height; xmax_n=xmax_p_s/target_width; ymax_n=ymax_p_s/target_height
                                    xmin_n=max(0.,min(1.,xmin_n)); ymin_n=max(0.,min(1.,ymin_n)); xmax_n=max(0.,min(1.,xmax_n)); ymax_n=max(0.,min(1.,ymax_n))
                                    processed_bboxes.append([ymin_n, xmin_n, ymax_n, xmax_n])

                                if not processed_bboxes: skipped_bbox+=len(bboxes_geo); continue

                                # --- Padding Logic ---
                                bboxes_np = np.array(processed_bboxes, dtype=np.float32)
                                num_boxes = bboxes_np.shape[0]

                                if num_boxes > MAX_BOXES:
                                    logger.warning(f"Image {image_id} has {num_boxes} boxes, truncating to {MAX_BOXES}.")
                                    bboxes_np = bboxes_np[:MAX_BOXES, :]
                                    num_boxes = MAX_BOXES # Update count after truncating

                                # Pad if necessary
                                num_padding = MAX_BOXES - num_boxes
                                if num_padding > 0:
                                    padding = np.zeros((num_padding, 4), dtype=np.float32)
                                    padded_bboxes = np.concatenate([bboxes_np, padding], axis=0)
                                else:
                                    padded_bboxes = bboxes_np # Already correct size or truncated
                                # --- End Padding Logic ---

                                # Yield padded bboxes
                                yield image_norm, padded_bboxes
                                yielded_count += 1
                            else: skipped_img+=1
                    except KeyError: skipped_img+=1
                    except Exception as img_proc_err: logger.exception(f"Error processing pair for {image_id}: {img_proc_err}"); skipped_img+=1
        except Exception as e: logger.exception(f"Generator Error: {e}")
        finally: logger.info(f"Data generator finished. Processed: {processed_count}. Yielded: {yielded_count}. Skipped (XML): {skipped_xml}. Skipped (BBox): {skipped_bbox}. Skipped (Image): {skipped_img}.")

    # --- Update Output Signature ---
    output_signature = (
        tf.TensorSpec(shape=(TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(MAX_BOXES, 4), dtype=tf.float32) # Fixed shape
    )
    # --- End Update ---
    max_items_arg = max_items_to_process if max_items_to_process is not None else -1
    try:
        logger.info("Creating tf.data.Dataset from generator...")
        tf_dataset = tf.data.Dataset.from_generator(_data_generator, args=[TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, max_items_arg], output_signature=output_signature)
        logger.info("tf.data.Dataset created successfully.")
        return tf_dataset
    except Exception as e: logger.exception(f"Failed to create tf.data.Dataset: {e}"); return None

# --- Training Orchestration ---
def start_training(data_dir: str, epochs: int = 5, batch_size: int = 32, learning_rate: float = 0.001):
    logger.info("--- Starting Model Training ---")
    logger.info(f"Parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

    logger.info(f"--- Loading full dataset ---")
    tf_dataset = load_rareplanes_data(Path(data_dir), max_items_to_process=None)

    if tf_dataset is None:
        logger.error("Failed to create dataset. Aborting training.")
        return {"status": "Failed: Dataset creation error", "model_path": None}

    buffer_size = 1000
    logger.info(f"Shuffling dataset with buffer size {buffer_size}")
    tf_dataset = tf_dataset.shuffle(buffer_size=buffer_size)
    logger.info(f"Batching dataset with batch size {batch_size}")
    tf_dataset = tf_dataset.batch(batch_size) # Should now work with padded bboxes
    logger.info("Applying prefetching to dataset")
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    # Build Model (now outputs MAX_BOXES, 4)
    input_shape=(TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, 3)
    model = build_simple_cnn(input_shape=input_shape, num_classes=0) # num_classes not used by model now
    model.summary(print_fn=logger.info)

    # Compile Model - Use MSE loss
    logger.info("Compiling model with MSE loss...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')

    # Train Model
    logger.info(f"Starting model training for {epochs} epochs...")
    history = None
    try:
        history = model.fit(tf_dataset, epochs=epochs)
        logger.info("Model training finished.")
        if history: logger.info(f"Training history: {history.history}")
    except Exception as train_err:
        logger.exception(f"Error during model training: {train_err}")

    # Save Model
    model_save_path = MODEL_SAVE_DIR / "simple_cnn_trained_mse_padded.keras" # New name
    logger.info(f"Saving model to {model_save_path}")
    try:
        model.save(model_save_path)
        logger.info("Model saved successfully.")
    except Exception as save_err:
         logger.exception(f"Error saving model: {save_err}")

    logger.info("--- Model Training Script Finished ---")
    final_status = f"Training finished after {epochs} epochs (MSE loss, padded, check logs)."
    return {"status": final_status, "model_path": str(model_save_path)}

# End of file backend/models/train.py
