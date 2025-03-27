# backend/models/model.py
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Define MAX_BOXES consistently (can be moved to a config later)
MAX_BOXES = 100

def build_simple_cnn(input_shape, num_classes): # num_classes arg is unused now
    """
    Builds a basic CNN model.
    Modified to output a fixed number (MAX_BOXES) of bounding boxes.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            # Output MAX_BOXES * 4 values, then reshape
            layers.Dense(MAX_BOXES * 4, activation="sigmoid"), # Sigmoid keeps coords 0-1
            layers.Reshape((MAX_BOXES, 4)) # Reshape to (batch, MAX_BOXES, 4)
        ]
    )
    print(f"Simple CNN model built (outputting fixed {MAX_BOXES} boxes).")
    return model

# TODO: Implement a proper object detection model architecture
