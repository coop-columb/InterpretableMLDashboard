# backend/models/model.py
import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_simple_cnn(input_shape, num_classes):
    """
    Builds a basic CNN model.
    For detection, num_classes isn't quite right. We'll output 4 coords for now.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            # layers.Dropout(0.5), # Keep dropout? Maybe remove for regression. Let's remove for now.
            # Output 4 values (e.g., ymin, xmin, ymax, xmax) instead of num_classes/1
            layers.Dense(4, activation="sigmoid"), # Use sigmoid to keep outputs between 0 and 1
        ]
    )
    print("Simple CNN model built (modified for 4 outputs).")
    return model

# TODO: Implement a proper object detection model architecture
# (e.g., using KerasCV, TensorFlow Object Detection API, or building manually)

