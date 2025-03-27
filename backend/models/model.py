# backend/models/model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_simple_cnn(input_shape=(256, 256, 3), num_classes=1):
    """
    Builds a very simple placeholder CNN model.
    NOTE: This is a basic structure and needs significant refinement
          for actual object detection on RarePlanes (e.g., using pre-trained
          models, object detection heads like RetinaNet/YOLO/Faster R-CNN).
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            # Placeholder output layer - needs adjustment for actual task
            # For binary classification (plane/no plane), num_classes=1, activation='sigmoid'
            # For object detection, the output structure is much more complex.
            layers.Dense(num_classes, activation="sigmoid"),
        ]
    )
    print("Simple CNN model built (placeholder).")
    model.summary() # Print model summary to console
    return model

# You might add functions here later for loading pre-trained models,
# specific object detection architectures, etc.

