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

from .model import build_simple_cnn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

MODEL_SAVE_DIR = Path(__file__).resolve().parent.parent / "saved_models"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_IMG_WIDTH = 256
TARGET_IMG_HEIGHT = 256

def _load_annotations(data_dir: Path) -> dict:
    """Loads and parses all GeoJSON annotations from the archive."""
    logger.info(f"Loading annotation data from {data_dir}...")
    start_time = time.time()
    annotation_archive_name = "RarePlanes_train_geojson_aircraft_tiled.tar.gz"
    annotation_archive_path = data_dir / annotation_archive_name
    all_annotations = {}
    total_features_processed = 0
    processed_anno_file_count = 0

    if not annotation_archive_path.exists():
        logger.error(f"Anno archive not found: {annotation_archive_path}")
        return {}

    try:
        with tarfile.open(annotation_archive_path, "r:gz") as tar:
            members = tar.getmembers(); total_members = len(members)
            logger.info(f"Found {total_members} anno members. Iterating...")
            for idx, member in enumerate(members):
                # Minimal logging for full run
                # if idx > 0 and idx % 1000 == 0: logger.info(f"  Processed {idx}/{total_members} anno members...")

                if member.isfile() and member.name.lower().endswith('.geojson'):
                    image_id = Path(member.name).stem
                    try:
                        with tar.extractfile(member) as f:
                            if f:
                                data = json.loads(f.read().decode('utf-8'))
                                features = data.get('features', [])
                                if features:
                                    bboxes_for_image = []
                                    for feature in features:
                                        try:
                                            coords = feature['geometry']['coordinates'][0]
                                            xmin_geo = min(p[0] for p in coords)
                                            ymin_geo = min(p[1] for p in coords)
                                            xmax_geo = max(p[0] for p in coords)
                                            ymax_geo = max(p[1] for p in coords)
                                            bboxes_for_image.append([xmin_geo, ymin_geo, xmax_geo, ymax_geo])
                                        except Exception as feature_err:
                                            logger.debug(f"Skipping feature due to error in {member.name}: {feature_err}")
                                            continue

                                    if bboxes_for_image:
                                        all_annotations.setdefault(image_id, []).extend(bboxes_for_image)
                                        total_features_processed += len(bboxes_for_image)
                                        processed_anno_file_count += 1
                    except Exception as inner_e: logger.warning(f"Skipping anno member {member.name}: {inner_e}")
    except Exception as e: logger.exception(f"Error processing annotation archive: {e}"); return {}

    logger.info(f"Finished anno processing in {time.time() - start_time:.2f}s. Found {total_features_processed} annos across {len(all_annotations)} images (from {processed_anno_file_count} files).")
    if not all_annotations: logger.error("Annotation dictionary is empty after processing!")
    return all_annotations

def _get_geotransform_from_xml(xml_path: Path) -> tuple[float, ...] | None:
    """Parses a GDAL .aux.xml file to extract the GeoTransform."""
    if not xml_path.exists():
        logger.warning(f"Metadata file not found: {xml_path}")
        return None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        geotransform_str = None
        possible_paths = [".//Metadata/Item[@name='GEOTRANSFORM']", ".//GeoTransform"]
        for path in possible_paths:
             elem = root.find(path)
             if elem is not None and elem.text: geotransform_str = elem.text; break
        if geotransform_str:
            parts = geotransform_str.replace(',', ' ').split()
            if len(parts) == 6: return tuple(float(p) for p in parts)
            else: logger.warning(f"Could not parse GeoTransform string '{geotransform_str}' in {xml_path}")
        else: logger.warning(f"Could not find GeoTransform metadata item in {xml_path}")
        return None
    except ET.ParseError: logger.error(f"Error parsing XML file: {xml_path}"); return None
    except Exception as e: logger.error(f"Unexpected error reading GeoTransform from {xml_path}: {e}"); return None

def _geo_to_pixel(lon: float, lat: float, geotransform: tuple[float, ...]) -> tuple[float, float] | None:
    """Converts geographic coordinates (lon, lat) to pixel coordinates (x, y) using GDAL GeoTransform."""
    if geotransform is None or len(geotransform) != 6: return None
    ulx, x_res, _, uly, _, y_res = geotransform
    if x_res == 0 or y_res == 0: logger.warning("Invalid GeoTransform: x_res or y_res is zero."); return None
    x_pixel = (lon - ulx) / x_res
    y_pixel = (lat - uly) / y_res
    return x_pixel, y_pixel

def load_rareplanes_data(data_dir: Path, max_items_to_process: int | None = None) -> tf.data.Dataset | None:
    """Loads annotations and creates a tf.data.Dataset using an inner generator."""
    all_annotations = _load_annotations(data_dir)
    if not all_annotations: return None

    image_archive_path = data_dir / "RarePlanes_train_PS-RGB_tiled.tar.gz"
    image_metadata_dir = data_dir / "RarePlanes_train_PS-RGB_tiled" # Adjust if needed

    if not image_archive_path.exists(): logger.error(f"Image archive missing: {image_archive_path}"); return None

    def _data_generator(target_height: int, target_width: int, max_items: int):
        logger.info(f"Data generator starting. Processing up to {'all' if max_items == -1 else max_items} items.")
        processed_count = 0
        yielded_count = 0 # Track actual yields
        skipped_xml_count = 0
        skipped_bbox_count = 0
        skipped_img_count = 0
        try:
            logger.debug(f"Generator: Attempting to open image archive: {image_archive_path}")
            with tarfile.open(image_archive_path, "r:gz") as tar:
                logger.debug("Generator: Image archive opened.")
                for image_id, bboxes_geo in all_annotations.items():
                    if max_items != -1 and yielded_count >= max_items: # Limit based on yield count
                        logger.info(f"Generator reached yield limit ({max_items}). Breaking loop.")
                        break

                    processed_count += 1
                    if processed_count % 500 == 0: # Log progress less often
                         logger.info(f"Generator processed {processed_count}/{len(all_annotations)} potential images...")

                    xml_filename = f"{image_id}.png.aux.xml"
                    xml_member_name = f"./PS-RGB_tiled/{xml_filename}"
                    xml_path_absolute = image_metadata_dir / xml_filename
                    geotransform = None
                    try:
                        member_info = tar.getmember(xml_member_name)
                        with tar.extractfile(member_info) as f_xml:
                            if f_xml:
                                xml_content = f_xml.read(); tree = ET.ElementTree(ET.fromstring(xml_content)); root = tree.getroot()
                                gt_str = None; possible_paths = [".//Metadata/Item[@name='GEOTRANSFORM']", ".//GeoTransform"]
                                for path in possible_paths:
                                    elem = root.find(path);
                                    if elem is not None and elem.text: gt_str = elem.text; break
                                if gt_str: parts = gt_str.replace(',', ' ').split();
                                if len(parts) == 6: geotransform = tuple(float(p) for p in parts)
                        if geotransform is None: logger.debug(f"Could not parse GeoTransform from tar member {xml_member_name}")
                    except KeyError:
                        logger.debug(f"XML {xml_member_name} not in tar, trying {xml_path_absolute}")
                        geotransform = _get_geotransform_from_xml(xml_path_absolute)
                    except Exception as xml_err: logger.error(f"Error getting GeoTransform for {image_id}: {xml_err}"); geotransform = None

                    if geotransform is None: skipped_xml_count += 1; logger.debug(f"Skipping image {image_id} due to missing geotransform."); continue

                    image_member_name = f"./PS-RGB_tiled/{image_id}.png"
                    try:
                        image_member_info = tar.getmember(image_member_name)
                        with tar.extractfile(image_member_info) as f_img:
                            if f_img:
                                image_bytes = f_img.read(); image = Image.open(io.BytesIO(image_bytes)); orig_w, orig_h = image.size
                                image_resized = image.resize((target_width, target_height)); image_np = np.array(image_resized)
                                if image_np.ndim == 2: image_np = np.stack((image_np,)*3, axis=-1)
                                if image_np.shape[2] == 4: image_np = image_np[:, :, :3]
                                if image_np.shape[2] != 3: logger.warning(f"Generator: Skipping {image_id}: bad shape {image_np.shape}"); skipped_img_count+=1; continue
                                image_norm = image_np.astype(np.float32) / 255.0

                                processed_bboxes = []
                                for bbox_geo in bboxes_geo:
                                    xmin_geo, ymin_geo, xmax_geo, ymax_geo = bbox_geo
                                    top_left_pix = _geo_to_pixel(xmin_geo, ymax_geo, geotransform)
                                    bottom_right_pix = _geo_to_pixel(xmax_geo, ymin_geo, geotransform)
                                    if top_left_pix is None or bottom_right_pix is None: skipped_bbox_count+=1; continue
                                    xmin_pix, ymin_pix = top_left_pix; xmax_pix, ymax_pix = bottom_right_pix
                                    xmin_pix = max(0.0, xmin_pix); ymin_pix = max(0.0, ymin_pix)
                                    xmax_pix = min(float(orig_w), xmax_pix); ymax_pix = min(float(orig_h), ymax_pix)
                                    if xmax_pix <= xmin_pix or ymax_pix <= ymin_pix: skipped_bbox_count+=1; continue
                                    x_scale = float(target_width)/orig_w; y_scale = float(target_height)/orig_h
                                    xmin_pix_scaled=xmin_pix*x_scale; ymin_pix_scaled=ymin_pix*y_scale; xmax_pix_scaled=xmax_pix*x_scale; ymax_pix_scaled=ymax_pix*y_scale
                                    xmin_norm=xmin_pix_scaled/target_width; ymin_norm=ymin_pix_scaled/target_height; xmax_norm=xmax_pix_scaled/target_width; ymax_norm=ymax_pix_scaled/target_height
                                    xmin_norm=max(0.0, min(1.0, xmin_norm)); ymin_norm=max(0.0, min(1.0, ymin_norm)); xmax_norm=max(0.0, min(1.0, xmax_norm)); ymax_norm=max(0.0, min(1.0, ymax_norm))
                                    normalized_bbox = [ymin_norm, xmin_norm, ymax_norm, xmax_norm]
                                    processed_bboxes.append(normalized_bbox)

                                if not processed_bboxes: skipped_bbox_count+=len(bboxes_geo); logger.debug(f"Skipping {image_id} as no valid bboxes remained after processing."); continue

                                # Minimal logging during full run
                                # logger.info(f"Generator: Yielding item {yielded_count + 1} for {image_id} with {len(processed_bboxes)} boxes")
                                yield image_norm, np.array(processed_bboxes, dtype=np.float32)
                                yielded_count += 1

                            else: logger.warning(f"Generator: Could not extract file object for {image_member_name}"); skipped_img_count+=1;
                    except KeyError: logger.warning(f"Generator: Image member '{image_member_name}' not found. Skipping."); skipped_img_count+=1;
                    except Exception as img_proc_err: logger.exception(f"Generator: Error processing pair for {image_id}: {img_proc_err}"); skipped_img_count+=1;

                logger.info(f"Generator: Finished iterating through annotation keys.")
        except tarfile.TarError as e: logger.error(f"Generator: Error opening image archive: {e}")
        except Exception as e: logger.exception(f"Generator: Error: {e}")
        finally:
             logger.info(f"Data generator finished. Processed: {processed_count}. Yielded: {yielded_count}. Skipped (XML): {skipped_xml_count}. Skipped (BBox): {skipped_bbox_count}. Skipped (Image): {skipped_img_count}.")

    output_signature = (
        tf.TensorSpec(shape=(TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32)
    )
    max_items_arg = max_items_to_process if max_items_to_process is not None else -1

    try:
        logger.info("Creating tf.data.Dataset from generator...")
        tf_dataset = tf.data.Dataset.from_generator(
            _data_generator,
            args=[TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, max_items_arg],
            output_signature=output_signature
        )
        logger.info("tf.data.Dataset created successfully.")
        return tf_dataset
    except Exception as e:
        logger.exception(f"Failed to create tf.data.Dataset: {e}")
        return None

# --- Training Orchestration (Final Pipeline Config) ---
def start_training(data_dir: str, epochs: int = 5, batch_size: int = 32, learning_rate: float = 0.001):
    logger.info("--- Starting Model Training ---")
    logger.info(f"Parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

    # --- Load FULL dataset ---
    logger.info(f"--- Loading full dataset ---")
    tf_dataset = load_rareplanes_data(Path(data_dir), max_items_to_process=None)

    if tf_dataset is None:
        logger.error("Failed to create dataset. Aborting training.")
        return {"status": "Failed: Dataset creation error", "model_path": None}

    # --- Remove .take(1) testing block ---

    # --- Re-enable pipeline enhancements ---
    buffer_size = 1000 # Adjust based on memory
    logger.info(f"Shuffling dataset with buffer size {buffer_size}")
    tf_dataset = tf_dataset.shuffle(buffer_size=buffer_size)
    logger.info(f"Batching dataset with batch size {batch_size}")
    tf_dataset = tf_dataset.batch(batch_size) # Apply batching AFTER shuffling
    logger.info("Applying prefetching to dataset")
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    # --- End pipeline enhancements ---

    # Build Model
    input_shape=(TARGET_IMG_HEIGHT, TARGET_IMG_WIDTH, 3); num_classes=1 # Placeholder
    # TODO: Adapt num_classes based on actual detection task
    model = build_simple_cnn(input_shape=input_shape, num_classes=num_classes)
    model.summary(print_fn=logger.info)

    # Compile Model
    logger.info("Compiling model...")
    # TODO: Choose appropriate loss (e.g., Focal Loss for detection) and metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', # Placeholder loss
                  metrics=['accuracy']) # Placeholder metrics

    # Train Model (Placeholder - actual call commented out)
    logger.info(f"Placeholder: Starting model training for {epochs} epochs.")
    # try:
    #     history = model.fit(tf_dataset, epochs=epochs) # Actual training call
    #     logger.info("Model training finished.")
    #     # TODO: Process history
    # except Exception as train_err:
    #     logger.exception(f"Error during model training: {train_err}")
    #     return {"status": "Failed: Error during model.fit", "model_path": None}

    # Save Model (Placeholder)
    model_save_path = MODEL_SAVE_DIR / "simple_cnn_placeholder.keras"
    logger.info(f"Placeholder: Saving model to {model_save_path}")
    # model.save(model_save_path) # Actual save call

    logger.info("--- Model Training Script Finished (Final Pipeline, Placeholder Fit) ---")
    final_status = "Training script finished (Final Pipeline, Placeholder Fit)"
    return {"status": final_status, "model_path": str(model_save_path)}

# End of file backend/models/train.py
